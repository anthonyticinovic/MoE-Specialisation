"""Stage 2: specialise the two experts under a fixed, position-derived mask.

The stage that creates the expert specialisation the paper studies. Routing is
*hard*: visual tokens are forced to expert 0 and text tokens to expert 1 by a
mask built from token position, so no gate is trained and no routing decision is
learned. Only the experts are unfrozen; everything else, including the Stage 1
connector, stays frozen.

Run from the repo root::

    torchrun --nproc_per_node=4 training_scripts/train_stage_2.py

A single-process CPU run works too: FSDP, mixed precision and FlashAttention are
skipped when CUDA is unavailable. Paths come from
``configs/training_config.yaml`` unless ``MOE_CONFIG`` points elsewhere.
"""

from __future__ import annotations

import gc
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
    register_moe_model,
)
from training_scripts._lib import (
    RunContext,
    broadcast_flag,
    build_backbones,
    build_distributed_loaders,
    build_run_context,
    build_vision_connector,
    combine_sequence,
    load_matching_weights,
    shifted_caption_loss,
    teardown,
    wrap_with_fsdp,
)

logger = logging.getLogger(__name__)


@dataclass
class Stage2Setup:
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
    accumulation_steps: int
    vocab_size: int


def build_moe_llm(paths: dict[str, Any], ctx: RunContext) -> nn.Module:
    """Load the Stage 0 MoE model and unfreeze only its experts."""
    if ctx.is_main:
        logger.info("Loading custom MoE model from %s...", paths["moe_model_path"])

    llm = AutoModelForCausalLM.from_pretrained(
        paths["moe_model_path"],
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=get_model_dtype(),
        attn_implementation=get_attn_implementation(),
    )

    for param in llm.parameters():
        param.requires_grad = False
    for layer in llm.model.layers:
        if hasattr(layer.mlp, "experts"):
            for expert in layer.mlp.experts:
                for param in expert.parameters():
                    param.requires_grad = True

    if ctx.is_main:
        logger.info("✅ Custom MoE model loaded, experts unfrozen.")
    return llm


def apply_hard_routing_mask(
    llm: nn.Module, num_visual: tuple[int, int], num_text: tuple[int, int], ctx: RunContext
) -> None:
    """Force visual tokens to expert 0 and text tokens to expert 1.

    The mask is derived from position, not learned, and is not stored in the
    model — it must be set on every ``MoELayer`` before each forward pass.
    """
    routing_mask = torch.cat(
        [
            torch.zeros(num_visual, dtype=torch.long, device=ctx.device),
            torch.ones(num_text, dtype=torch.long, device=ctx.device),
        ],
        dim=1,
    )
    for layer in llm.model.layers:
        layer.mlp.routing_mask = routing_mask


def resume_from_checkpoint(
    llm: nn.Module, checkpoint_dir: str, ctx: RunContext
) -> tuple[int, float, bool]:
    """Restore model weights from the latest checkpoint if one exists.

    Returns the epoch to resume from, the best validation loss so far, and
    whether a checkpoint was found (the optimiser state is loaded later, once
    the optimiser exists).
    """
    latest_path = os.path.join(checkpoint_dir, "llm_stage2_latest.pth")
    found = broadcast_flag(os.path.exists(latest_path), ctx)

    latest_epoch, best_val_loss = 0, float("inf")
    if not found:
        if ctx.is_main:
            logger.info("No checkpoint found. Starting training from scratch.")
    else:
        with full_state_dict_context(llm):
            if ctx.is_main:
                checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
                state_dict_to_load = checkpoint["model_state_dict"]
                latest_epoch = checkpoint["epoch"]
                best_val_loss = checkpoint["best_val_loss"]
                logger.info(
                    "✅ Resumed from epoch %d. Previous best validation loss: %.4f",
                    latest_epoch,
                    best_val_loss,
                )
                del checkpoint
                gc.collect()
            else:
                state_dict_to_load = {}
            # A partial load is expected here: rank0_only gives the other ranks
            # an empty dict on purpose. An empty *overlap* on rank 0 is not.
            load_matching_weights(llm, state_dict_to_load, source=latest_path)

    state_data = [latest_epoch, best_val_loss]
    dist.broadcast_object_list(state_data, src=0)
    dist.barrier()
    return int(state_data[0]), state_data[1], found


def restore_optimizer_state(setup: Stage2Setup, checkpoint_dir: str, ctx: RunContext) -> None:
    """Load optimiser and scheduler state from the latest checkpoint."""
    if ctx.is_main:
        latest_path = os.path.join(checkpoint_dir, "llm_stage2_latest.pth")
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        if "optimizer_state_dict" in checkpoint:
            setup.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info(
                "  ✅ Loaded optimizer state (learning_rate: %.2e)",
                setup.optimizer.param_groups[0]["lr"],
            )
        if "scheduler_state_dict" in checkpoint:
            setup.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logger.info("  ✅ Loaded scheduler state (last_epoch: %d)", setup.scheduler.last_epoch)
        del checkpoint
        gc.collect()
    dist.barrier()


def train_one_epoch(setup: Stage2Setup, ctx: RunContext, epoch: int, num_epochs: int) -> float:
    """Run one training epoch under hard routing and return the mean loss."""
    setup.train_sampler.set_epoch(epoch)
    setup.llm.train()
    total_train_loss = 0.0
    setup.optimizer.zero_grad()

    for i, (images, input_ids, attention_mask) in enumerate(setup.train_loader):
        images = images.to(ctx.device)
        input_ids = input_ids.to(ctx.device)
        attention_mask = attention_mask.to(ctx.device)

        with autocast(device_type=ctx.amp_device, dtype=torch.bfloat16, enabled=ctx.on_gpu):
            # The encoder and the connector are both frozen in this stage, so
            # they run under no_grad; the token embedding does not, matching the
            # published Stage 2.
            with torch.no_grad():
                patch_embeddings = setup.vision_encoder(images).last_hidden_state
                visual_soft_tokens = setup.vision_connector(patch_embeddings)

            text_embeddings = setup.llm.model.embed_tokens(input_ids)
            combined_embeddings, combined_attention_mask = combine_sequence(
                visual_soft_tokens, text_embeddings, attention_mask, ctx
            )
            # Nothing upstream requires grad, so the LLM's input must become the
            # graph's leaf or the experts receive no gradient at all.
            combined_embeddings.requires_grad_(True)

            apply_hard_routing_mask(
                setup.llm, visual_soft_tokens.shape[:2], text_embeddings.shape[:2], ctx
            )

            outputs = setup.llm(
                inputs_embeds=combined_embeddings, attention_mask=combined_attention_mask
            )
            loss = shifted_caption_loss(
                setup.loss_fn,
                outputs.logits,
                visual_soft_tokens.shape[1],
                input_ids,
                setup.vocab_size,
                device=ctx.device,
            )
            loss = loss / setup.accumulation_steps

        setup.scaler.scale(loss).backward()
        if loss.item() > 0:
            total_train_loss += loss.item() * setup.accumulation_steps

        if (i + 1) % setup.accumulation_steps == 0 or (i + 1) == len(setup.train_loader):
            # Note: Stage 2 deliberately does not clip gradients.
            setup.scaler.step(setup.optimizer)
            setup.scaler.update()
            setup.optimizer.zero_grad()
            setup.scheduler.step()

        if ctx.is_main and (i + 1) % 100 == 0:
            logger.info("  Epoch %d, Batch [%d/%d]", epoch + 1, i + 1, len(setup.train_loader))

    avg_train_loss = total_train_loss / len(setup.train_loader)
    if ctx.is_main:
        logger.info("Epoch [%d/%d] - Training Loss: %.4f", epoch + 1, num_epochs, avg_train_loss)
    return avg_train_loss


def run_validation(setup: Stage2Setup, ctx: RunContext, epoch: int, num_epochs: int) -> float:
    """Validate under the same hard routing, averaged across ranks."""
    setup.llm.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for images, input_ids, attention_mask in setup.val_loader:
            images = images.to(ctx.device)
            input_ids = input_ids.to(ctx.device)
            attention_mask = attention_mask.to(ctx.device)

            with autocast(device_type=ctx.amp_device, dtype=torch.bfloat16, enabled=ctx.on_gpu):
                patch_embeddings = setup.vision_encoder(images).last_hidden_state
                visual_soft_tokens = setup.vision_connector(patch_embeddings)
                text_embeddings = setup.llm.model.embed_tokens(input_ids)
                combined_embeddings, combined_attention_mask = combine_sequence(
                    visual_soft_tokens, text_embeddings, attention_mask, ctx
                )
                apply_hard_routing_mask(
                    setup.llm, visual_soft_tokens.shape[:2], text_embeddings.shape[:2], ctx
                )
                outputs = setup.llm(
                    inputs_embeds=combined_embeddings, attention_mask=combined_attention_mask
                )
                loss = shifted_caption_loss(
                    setup.loss_fn,
                    outputs.logits,
                    visual_soft_tokens.shape[1],
                    input_ids,
                    setup.vocab_size,
                    device=ctx.device,
                )
                total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(setup.val_loader)
    # Every rank validates its own shard; average so all ranks agree on the
    # number used for model selection.
    val_loss_tensor = torch.tensor(avg_val_loss).to(ctx.device)
    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
    avg_val_loss = val_loss_tensor.item()

    if ctx.is_main:
        logger.info("Epoch [%d/%d] - Validation Loss: %.4f", epoch + 1, num_epochs, avg_val_loss)
    return avg_val_loss


def save_checkpoints(
    setup: Stage2Setup,
    ctx: RunContext,
    checkpoint_dir: str,
    epoch: int,
    avg_val_loss: float,
    best_val_loss: float,
) -> float:
    """Save the latest and, if improved, the best checkpoint. Returns the best."""
    # Gathering sharded parameters is collective — every rank must enter.
    with full_state_dict_context(setup.llm):
        cpu_state_dict = setup.llm.state_dict()

    if ctx.is_main:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Update the best *before* writing either file: the latest checkpoint is
        # what resumption reads, so recording a stale best there would let the
        # next epoch overwrite a better model.
        improved = avg_val_loss < best_val_loss
        if improved:
            best_val_loss = avg_val_loss
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": cpu_state_dict,
            "optimizer_state_dict": setup.optimizer.state_dict(),
            "scheduler_state_dict": setup.scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "current_val_loss": avg_val_loss,
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, "llm_stage2_latest.pth"))

        if improved:
            best_path = os.path.join(checkpoint_dir, "llm_stage2_best.pth")
            torch.save(checkpoint, best_path)
            logger.info(
                "🏆 New best model found! Validation loss: %.4f. Saved to %s",
                avg_val_loss,
                best_path,
            )
    return best_val_loss


def build_setup(
    paths: dict[str, Any],
    train_params: dict[str, Any],
    loader_params: dict[str, Any],
    num_epochs: int,
    ctx: RunContext,
) -> tuple[Stage2Setup, int, float, bool]:
    """Assemble every component the epoch loop needs, in dependency order."""
    output_dir = paths["output_dir"]
    checkpoint_dir = os.path.join(output_dir, "stage2_checkpoints")

    vision_encoder, clip_processor, tokenizer, _ = build_backbones(paths, ctx)
    llm = build_moe_llm(paths, ctx)

    vocab_size = llm.config.vocab_size
    # Stage 2 shards with parameter offloading.
    llm = wrap_with_fsdp(llm, ctx, offload_params=True)

    latest_epoch, best_val_loss, resumed = resume_from_checkpoint(llm, checkpoint_dir, ctx)
    llm.gradient_checkpointing_enable()

    vision_connector = build_vision_connector(vision_encoder, llm, ctx, trainable=False)
    stage1_path = os.path.join(output_dir, "vision_connector_stage1_best.pth")
    if os.path.exists(stage1_path):
        # device is a CUDA ordinal under FSDP and the string "cpu" otherwise.
        map_loc = f"cuda:{ctx.device}" if ctx.use_fsdp else ctx.device
        vision_connector.load_state_dict(torch.load(stage1_path, map_location=map_loc))
    elif ctx.is_main:
        logger.warning("Stage 1 connector weights not found at %s.", stage1_path)

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
    if ctx.is_main:
        logger.info("Using gradient accumulation with %d steps.", accumulation_steps)
        logger.info(
            "Effective batch size: %d",
            train_params["batch_size"] * accumulation_steps * dist.get_world_size(),
        )

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

    if ctx.is_main:
        logger.info("Optimizing %d trainable parameters.", sum(p.numel() for p in trainable_params))

    setup = Stage2Setup(
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
        accumulation_steps=accumulation_steps,
        vocab_size=vocab_size,
    )
    return setup, latest_epoch, best_val_loss, resumed


def main() -> None:
    """Assemble the Stage 2 run and drive the epoch loop."""
    register_moe_model()

    config = load_config()
    paths = config["paths"]
    train_params = config["training_stage2"]
    loader_params = config["dataloader"]
    num_epochs = train_params["num_epochs"]
    output_dir = paths["output_dir"]
    checkpoint_dir = os.path.join(output_dir, "stage2_checkpoints")

    ctx = build_run_context(
        distributed=True,
        seed=loader_params.get("data_seed", 42),
        stage_name="Stage 2",
    )
    if ctx.is_main:
        logger.info("--- Initializing Stage 2 Training ---")
        logger.info("PyTorch: %s", torch.__version__)
    logger.info("--- Rank %d --- Using device: %s", ctx.local_rank, ctx.device)

    setup, latest_epoch, best_val_loss, resumed = build_setup(
        paths, train_params, loader_params, num_epochs, ctx
    )
    if resumed:
        restore_optimizer_state(setup, checkpoint_dir, ctx)

    metrics_path = os.path.join(output_dir, "training_metrics_stage2.json")
    metrics_history: dict[str, list] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
    }
    if ctx.is_main and latest_epoch > 0 and os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics_history = json.load(f)

    if ctx.is_main:
        logger.info(
            "🚀 Starting Stage 2 training from epoch %d for %d total epochs...",
            latest_epoch,
            num_epochs,
        )

    start_time = time.time()
    for epoch in range(latest_epoch, num_epochs):
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
