"""Dense control: the same pipeline with a standard FFN instead of experts.

The baseline the MoE stages are compared against. Architecture and data match
Stage 2 — same frozen CLIP tower, same Stage 1 connector, same COCO captions,
same loss — but the language model is stock Mistral, so its dense FFN trains in
place of two routed experts. Self-attention and the FFN are unfrozen; the
connector and the vision tower are not.

Run from the repo root::

    torchrun --nproc_per_node=4 training_scripts/train_dense.py

A single-process CPU run works too: FSDP, mixed precision and FlashAttention are
skipped when CUDA is unavailable. Paths come from
``configs/training_config.yaml`` unless ``MOE_CONFIG`` points elsewhere.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM

from data import COCO_Loader
from models.utils.common import (
    full_state_dict_context,
    get_attn_implementation,
    get_model_dtype,
    load_config,
)
from training_scripts._lib import (
    RunContext,
    broadcast_flag,
    build_backbones,
    build_distributed_loaders,
    build_run_context,
    build_vision_connector,
    combine_sequence,
    shifted_caption_loss,
    teardown,
    wrap_with_fsdp,
)

logger = logging.getLogger(__name__)


@dataclass
class DenseSetup:
    """Everything the epoch loop needs, assembled once before training."""

    llm: nn.Module
    vision_encoder: nn.Module
    vision_connector: nn.Module
    train_loader: DataLoader
    val_loader: DataLoader
    train_sampler: DistributedSampler
    optimizer: optim.Optimizer
    scheduler: CosineAnnealingLR
    scaler: GradScaler
    loss_fn: nn.Module
    trainable_params: list[torch.Tensor]
    accumulation_steps: int
    vocab_size: int


def build_dense_llm(paths: dict[str, Any], ctx: RunContext) -> nn.Module:
    """Load stock Mistral and unfreeze self-attention plus the dense FFN.

    This is the control: the same trainable surface as Stage 3 minus the router,
    so any difference in the results is attributable to the MoE layer rather
    than to what was left trainable.
    """
    if ctx.is_main:
        logger.info("Loading dense Mistral from %s...", paths["mistral_local_path"])

    llm = AutoModelForCausalLM.from_pretrained(
        paths["mistral_local_path"],
        torch_dtype=get_model_dtype(),
        attn_implementation=get_attn_implementation(),
        low_cpu_mem_usage=True,
    )

    for param in llm.parameters():
        param.requires_grad = False
    for layer in llm.model.layers:
        layer.self_attn.requires_grad_(True)
        layer.mlp.requires_grad_(True)

    if ctx.is_main:
        logger.info("Dense model prepared: self-attention and FFN unfrozen.")
    return llm


def resume_from_checkpoint(
    setup: DenseSetup, checkpoint_dir: str, ctx: RunContext
) -> tuple[int, float]:
    """Restore model, optimiser and scheduler state if a checkpoint exists."""
    latest_path = os.path.join(checkpoint_dir, "dense_latest.pth")
    if not broadcast_flag(os.path.exists(latest_path), ctx):
        if ctx.is_main:
            logger.info("No checkpoint found. Starting dense training from scratch.")
        return 0, float("inf")

    if ctx.is_main:
        logger.debug("💾 Resuming dense training from %s", latest_path)

    checkpoint = torch.load(latest_path, map_location="cpu")
    with full_state_dict_context(setup.llm, rank0_only=False):
        setup.llm.load_state_dict(checkpoint["model_state_dict"])
    setup.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    setup.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    if ctx.is_main:
        logger.info("✅ Resumed from epoch %d (best val loss %.4f).", start_epoch, best_val_loss)
    return start_epoch, best_val_loss


def _forward(
    setup: DenseSetup,
    ctx: RunContext,
    images: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Run the frozen front end and the trainable LLM.

    The vision tower, the connector and the token embedding are all frozen here,
    so the whole front end runs under ``no_grad``.
    """
    with torch.no_grad():
        patch_embeddings = setup.vision_encoder(images).last_hidden_state
        visual_soft_tokens = setup.vision_connector(patch_embeddings)
        text_embeddings = setup.llm.model.embed_tokens(input_ids)

    combined_embeddings, combined_attention_mask = combine_sequence(
        visual_soft_tokens, text_embeddings, attention_mask, ctx
    )
    outputs = setup.llm(inputs_embeds=combined_embeddings, attention_mask=combined_attention_mask)
    return outputs.logits, visual_soft_tokens.shape[1]


def train_one_epoch(setup: DenseSetup, ctx: RunContext, epoch: int, num_epochs: int) -> float:
    """Run one training epoch and return the mean loss per batch."""
    setup.train_sampler.set_epoch(epoch)
    setup.llm.train()
    total_train_loss = 0.0
    setup.optimizer.zero_grad()

    for i, (images, input_ids, attention_mask) in enumerate(setup.train_loader):
        images = images.to(ctx.device)
        input_ids = input_ids.to(ctx.device)
        attention_mask = attention_mask.to(ctx.device)

        with autocast(device_type=ctx.amp_device, dtype=torch.bfloat16, enabled=ctx.on_gpu):
            logits, num_visual_tokens = _forward(setup, ctx, images, input_ids, attention_mask)
            loss = shifted_caption_loss(
                setup.loss_fn,
                logits,
                num_visual_tokens,
                input_ids,
                setup.vocab_size,
                device=ctx.device,
            )
            loss = loss / setup.accumulation_steps

        setup.scaler.scale(loss).backward()
        if loss.item() > 0:
            total_train_loss += loss.item() * setup.accumulation_steps

        if (i + 1) % setup.accumulation_steps == 0 or (i + 1) == len(setup.train_loader):
            setup.scaler.unscale_(setup.optimizer)
            torch.nn.utils.clip_grad_norm_(setup.trainable_params, max_norm=1.0)
            setup.scaler.step(setup.optimizer)
            setup.scaler.update()
            setup.optimizer.zero_grad()
            setup.scheduler.step()

    avg_train_loss = total_train_loss / len(setup.train_loader)
    if ctx.is_main:
        logger.info("Epoch [%d/%d] - Training Loss: %.4f", epoch + 1, num_epochs, avg_train_loss)
    return avg_train_loss


def run_validation(setup: DenseSetup, ctx: RunContext, epoch: int, num_epochs: int) -> float:
    """Validate and average the loss across ranks."""
    setup.llm.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for images, input_ids, attention_mask in setup.val_loader:
            images = images.to(ctx.device)
            input_ids = input_ids.to(ctx.device)
            attention_mask = attention_mask.to(ctx.device)

            with autocast(device_type=ctx.amp_device, dtype=torch.bfloat16, enabled=ctx.on_gpu):
                logits, num_visual_tokens = _forward(setup, ctx, images, input_ids, attention_mask)
                loss = shifted_caption_loss(
                    setup.loss_fn,
                    logits,
                    num_visual_tokens,
                    input_ids,
                    setup.vocab_size,
                    device=ctx.device,
                )
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(setup.val_loader)
    val_loss_tensor = torch.tensor(avg_val_loss).to(ctx.device)
    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
    avg_val_loss = val_loss_tensor.item()

    if ctx.is_main:
        logger.info("Epoch [%d/%d] - Validation Loss: %.4f", epoch + 1, num_epochs, avg_val_loss)
    return avg_val_loss


def save_checkpoints(
    setup: DenseSetup,
    ctx: RunContext,
    checkpoint_dir: str,
    epoch: int,
    avg_val_loss: float,
    best_val_loss: float,
) -> float:
    """Save the latest and, if improved, the best checkpoint. Returns the best."""
    # Gathering sharded parameters is collective — every rank must enter.
    with full_state_dict_context(setup.llm, rank0_only=True):
        llm_state_dict = setup.llm.state_dict()

    if ctx.is_main:
        # Update the best before writing either file — see train_stage_2.py.
        improved = avg_val_loss < best_val_loss
        if improved:
            best_val_loss = avg_val_loss
        checkpoint = {
            "model_state_dict": llm_state_dict,
            "optimizer_state_dict": setup.optimizer.state_dict(),
            "scheduler_state_dict": setup.scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "current_val_loss": avg_val_loss,
        }
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(checkpoint, os.path.join(checkpoint_dir, "dense_latest.pth"))

        if improved:
            best_path = os.path.join(checkpoint_dir, "dense_best.pth")
            torch.save(checkpoint, best_path)
            logger.info("🏆 New best model! Val loss: %.4f. Saved to %s", avg_val_loss, best_path)

    dist.barrier()
    return best_val_loss


def build_setup(
    paths: dict[str, Any],
    train_params: dict[str, Any],
    loader_params: dict[str, Any],
    num_epochs: int,
    ctx: RunContext,
) -> DenseSetup:
    """Assemble every component the epoch loop needs, in dependency order."""
    output_dir = paths["output_dir"]

    vision_encoder, clip_processor, tokenizer, _ = build_backbones(paths, ctx)
    llm = build_dense_llm(paths, ctx)
    vocab_size = llm.config.vocab_size

    vision_connector = build_vision_connector(vision_encoder, llm, ctx, trainable=False)

    # The dense baseline shards the embedding along with everything else — it
    # does not pass ignored_modules, unlike the MoE stages.
    llm = wrap_with_fsdp(llm, ctx, offload_params=True, ignore_embeddings=False)

    stage1_path = os.path.join(output_dir, "vision_connector_stage1_best.pth")
    if os.path.exists(stage1_path):
        logger.debug("💾 Loading Stage 1 connector weights from %s", stage1_path)
        vision_connector.load_state_dict(torch.load(stage1_path, map_location=ctx.device))
    dist.barrier()

    if ctx.is_main:
        logger.info("Creating datasets and dataloaders...")
    common = {
        "image_dir": paths["image_dir"],
        "annotations_file": paths["annotations_file"],
        "clip_processor": clip_processor,
        "tokenizer": tokenizer,
        "subset_fraction": train_params["subset_fraction"],
        "seed": loader_params.get("data_seed", 42),
    }
    train_loader, val_loader, train_sampler = build_distributed_loaders(
        COCO_Loader(split="train", **common),
        COCO_Loader(split="val", **common),
        train_params,
        loader_params,
        ctx,
    )

    accumulation_steps = train_params.get("gradient_accumulation_steps", 1)
    trainable_params = [p for p in llm.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=train_params["learning_rate"],
        weight_decay=train_params["weight_decay"],
        fused=True,
    )
    scaler = GradScaler(ctx.amp_device, enabled=ctx.on_gpu)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=(len(train_loader) // accumulation_steps) * num_epochs
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    return DenseSetup(
        llm=llm,
        vision_encoder=vision_encoder,
        vision_connector=vision_connector,
        train_loader=train_loader,
        val_loader=val_loader,
        train_sampler=train_sampler,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        loss_fn=loss_fn,
        trainable_params=trainable_params,
        accumulation_steps=accumulation_steps,
        vocab_size=vocab_size,
    )


def main() -> None:
    """Assemble the dense control run and drive the epoch loop."""
    config = load_config()
    paths = config["paths"]
    train_params = config["dense_control"]
    loader_params = config["dataloader"]
    num_epochs = train_params["num_epochs"]
    output_dir = paths["output_dir"]
    checkpoint_dir = os.path.join(output_dir, "dense_checkpoints")

    ctx = build_run_context(
        distributed=True,
        seed=loader_params.get("data_seed", 42),
        stage_name="the dense control",
    )
    if ctx.is_main:
        logger.info("--- Initializing Dense Control Model Training ---")

    setup = build_setup(paths, train_params, loader_params, num_epochs, ctx)
    start_epoch, best_val_loss = resume_from_checkpoint(setup, checkpoint_dir, ctx)

    metrics_path = os.path.join(output_dir, "training_metrics_dense.json")
    metrics_history: dict[str, list] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
    }
    if ctx.is_main and start_epoch > 0 and os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics_history = json.load(f)

    if ctx.is_main:
        logger.info("🚀 Starting dense control training...")

    start_time = time.time()
    for epoch in range(start_epoch, num_epochs):
        avg_train_loss = train_one_epoch(setup, ctx, epoch, num_epochs)
        avg_val_loss = run_validation(setup, ctx, epoch, num_epochs)

        if ctx.is_main:
            metrics_history["epoch"].append(epoch + 1)
            metrics_history["train_loss"].append(avg_train_loss)
            metrics_history["val_loss"].append(avg_val_loss)
            metrics_history["learning_rate"].append(setup.optimizer.param_groups[0]["lr"])
            with open(metrics_path, "w") as f:
                json.dump(metrics_history, f, indent=4)
            logger.info("✅ Metrics saved to %s", metrics_path)

        best_val_loss = save_checkpoints(
            setup, ctx, checkpoint_dir, epoch, avg_val_loss, best_val_loss
        )

    if ctx.is_main:
        duration = int(time.time() - start_time)
        logger.info(
            "--- Total Training Time: %dh %dm %ds ---",
            duration // 3600,
            (duration % 3600) // 60,
            duration % 60,
        )

    teardown()
    logger.info("Job finished.")


if __name__ == "__main__":
    main()
