"""Stage 1: train the vision→language connector.

The first training stage and the only single-GPU one. CLIP and the LLM are
frozen; the sole trainable module is the ``VisionLanguageConnector`` that
projects CLIP patch embeddings into the LLM's embedding space. Everything
downstream loads the connector this stage produces.

Run from the repo root::

    python training_scripts/train_stage_1.py

A CPU run works too — mixed precision, 8-bit loading and ``torch.compile`` are
skipped when CUDA is unavailable. Paths come from
``configs/training_config.yaml`` unless ``MOE_CONFIG`` points elsewhere.
"""

from __future__ import annotations

import gc
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import MistralForCausalLM

from data import COCO_Loader
from models.utils.common import get_model_dtype, load_config
from training_scripts._lib import (
    RunContext,
    build_backbones,
    build_run_context,
    build_vision_connector,
    combine_sequence,
)

logger = logging.getLogger(__name__)


@dataclass
class Stage1Setup:
    """Everything the epoch loop needs, assembled once before training."""

    llm: nn.Module
    vision_encoder: nn.Module
    vision_connector: nn.Module
    train_loader: DataLoader
    val_loader: DataLoader
    optimizer: optim.Optimizer
    scheduler: CosineAnnealingLR
    scaler: GradScaler
    loss_fn: nn.Module
    accumulation_steps: int
    vocab_size: int


def build_frozen_llm(paths: dict[str, Any], ctx: RunContext) -> nn.Module:
    """Load the base Mistral, frozen — only its embeddings and logits are used.

    8-bit loading needs bitsandbytes and CUDA, so a CPU run falls back to
    float32.
    """
    load_kwargs: dict[str, Any] = (
        {"load_in_8bit": True, "dtype": torch.bfloat16}
        if ctx.on_gpu
        else {"dtype": get_model_dtype()}
    )
    llm = MistralForCausalLM.from_pretrained(paths["mistral_local_path"], **load_kwargs)
    if not ctx.on_gpu:
        # 8-bit loading places the model itself; the float32 path does not.
        llm = llm.to(ctx.device)

    for param in llm.parameters():
        param.requires_grad = False
    logger.info("✅ Models loaded and frozen.")
    return llm


def build_datasets(
    paths: dict[str, Any],
    train_params: dict[str, Any],
    loader_params: dict[str, Any],
    clip_processor: Any,
    tokenizer: Any,
) -> tuple[Any, Any]:
    """Build the COCO caption datasets. The same seed keeps the split stable."""
    common = {
        "image_dir": paths["image_dir"],
        "annotations_file": paths["annotations_file"],
        "clip_processor": clip_processor,
        "tokenizer": tokenizer,
        "subset_fraction": train_params["subset_fraction"],
        "seed": loader_params.get("data_seed", 42),
    }
    return COCO_Loader(split="train", **common), COCO_Loader(split="val", **common)


def build_loaders(
    train_dataset: Any,
    val_dataset: Any,
    train_params: dict[str, Any],
    loader_params: dict[str, Any],
    ctx: RunContext,
) -> tuple[DataLoader, DataLoader]:
    """Plain (non-distributed) loaders — Stage 1 is single-process.

    persistent_workers requires num_workers > 0, and pinned memory only helps
    host→GPU copies.
    """
    num_workers = loader_params["num_workers_s1"]
    loader_kwargs = {
        "batch_size": train_params["batch_size"],
        "num_workers": num_workers,
        "pin_memory": ctx.on_gpu,
        "persistent_workers": num_workers > 0,
    }
    return (
        DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        DataLoader(val_dataset, shuffle=False, **loader_kwargs),
    )


def _connector_loss(
    setup: Stage1Setup, logits: torch.Tensor, num_visual_tokens: int, input_ids: torch.Tensor
) -> torch.Tensor:
    """Next-token loss over the caption.

    The slice starts one position earlier than in Stages 2/2.5/dense: the last
    visual position predicts the first caption token, so every caption token is
    scored rather than all but the first. Preserved as-is — it is the alignment
    the published Stage 1 was trained with.
    """
    text_logits = logits[:, num_visual_tokens - 1 : -1, :].contiguous()
    text_labels = input_ids.contiguous()
    return setup.loss_fn(text_logits.view(-1, setup.vocab_size), text_labels.view(-1))


def _forward_batch(
    setup: Stage1Setup,
    ctx: RunContext,
    images: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Project the image, prepend it to the text and run the frozen LLM.

    The connector is the only trainable module, so it stays outside ``no_grad``
    while the encoder and the token embedding do not.
    """
    visual_soft_tokens = setup.vision_connector(patch_embeddings_of(setup, images))
    text_embeddings = setup.llm.model.embed_tokens(input_ids)
    combined_embeddings, combined_attention_mask = combine_sequence(
        visual_soft_tokens, text_embeddings, attention_mask, ctx
    )
    outputs = setup.llm(inputs_embeds=combined_embeddings, attention_mask=combined_attention_mask)
    return outputs.logits, visual_soft_tokens.shape[1]


def patch_embeddings_of(setup: Stage1Setup, images: torch.Tensor) -> torch.Tensor:
    """CLIP patch embeddings, always under no_grad — the tower is frozen."""
    with torch.no_grad():
        return setup.vision_encoder(images).last_hidden_state


def train_one_epoch(setup: Stage1Setup, ctx: RunContext, epoch: int, num_epochs: int) -> float:
    """Run one training epoch and return the mean loss per batch."""
    setup.vision_connector.train()
    total_train_loss = 0.0
    setup.optimizer.zero_grad()

    for i, (images, input_ids, attention_mask) in enumerate(setup.train_loader):
        images = images.to(ctx.device)
        input_ids = input_ids.to(ctx.device)
        attention_mask = attention_mask.to(ctx.device)

        # The frozen encoder and embedding run outside autocast, matching the
        # published Stage 1: only the trainable connector and the LLM forward
        # are done in reduced precision.
        with torch.no_grad():
            patch_embeddings = setup.vision_encoder(images).last_hidden_state
            text_embeddings = setup.llm.model.embed_tokens(input_ids)

        with autocast(ctx.amp_device, dtype=torch.bfloat16, enabled=ctx.on_gpu):
            visual_soft_tokens = setup.vision_connector(patch_embeddings)
            num_visual_tokens = visual_soft_tokens.shape[1]
            combined_embeddings, combined_attention_mask = combine_sequence(
                visual_soft_tokens, text_embeddings, attention_mask, ctx
            )
            outputs = setup.llm(
                inputs_embeds=combined_embeddings, attention_mask=combined_attention_mask
            )
            loss = _connector_loss(setup, outputs.logits, num_visual_tokens, input_ids)
            loss = loss / setup.accumulation_steps

        setup.scaler.scale(loss).backward()

        if (i + 1) % setup.accumulation_steps == 0 or (i + 1) == len(setup.train_loader):
            setup.scaler.unscale_(setup.optimizer)
            torch.nn.utils.clip_grad_norm_(setup.vision_connector.parameters(), 1.0)
            setup.scaler.step(setup.optimizer)
            setup.scaler.update()
            setup.scheduler.step()
            setup.optimizer.zero_grad()

        total_train_loss += loss.item() * setup.accumulation_steps

        if (i + 1) % 100 == 0:
            logger.info("  Epoch %d, Batch [%d/%d]", epoch + 1, i + 1, len(setup.train_loader))

        del images, input_ids, attention_mask, patch_embeddings, text_embeddings
        del visual_soft_tokens, combined_embeddings, outputs, loss
        gc.collect()
        if ctx.on_gpu:
            torch.cuda.empty_cache()

    avg_train_loss = total_train_loss / len(setup.train_loader)
    logger.info("Epoch [%d/%d] - Training Loss: %.4f", epoch + 1, num_epochs, avg_train_loss)
    return avg_train_loss


def run_validation(setup: Stage1Setup, ctx: RunContext, epoch: int, num_epochs: int) -> float:
    """Validate the connector and return the mean loss."""
    setup.vision_connector.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for images, input_ids, attention_mask in setup.val_loader:
            images = images.to(ctx.device)
            input_ids = input_ids.to(ctx.device)
            attention_mask = attention_mask.to(ctx.device)

            with autocast(ctx.amp_device, dtype=torch.bfloat16, enabled=ctx.on_gpu):
                logits, num_visual_tokens = _forward_batch(
                    setup, ctx, images, input_ids, attention_mask
                )
                loss = _connector_loss(setup, logits, num_visual_tokens, input_ids)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(setup.val_loader)
    logger.info("Epoch [%d/%d] - Validation Loss: %.4f", epoch + 1, num_epochs, avg_val_loss)
    return avg_val_loss


def save_checkpoints(
    setup: Stage1Setup, output_dir: str, avg_val_loss: float, best_val_loss: float
) -> float:
    """Save the latest connector and, if improved, the best. Returns the best."""
    # Unwrap torch.compile's OptimizedModule so the saved state dict loads into
    # a plain VisionLanguageConnector.
    state = getattr(setup.vision_connector, "_orig_mod", setup.vision_connector).state_dict()

    torch.save(state, os.path.join(output_dir, "vision_connector_stage1_latest.pth"))
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_path = os.path.join(output_dir, "vision_connector_stage1_best.pth")
        torch.save(state, best_path)
        logger.info("🏆 New best validation loss! Model saved to %s", best_path)
    return best_val_loss


def build_setup(
    paths: dict[str, Any],
    train_params: dict[str, Any],
    loader_params: dict[str, Any],
    num_epochs: int,
    ctx: RunContext,
) -> Stage1Setup:
    """Assemble every component the epoch loop needs, in dependency order."""
    vision_encoder, clip_processor, tokenizer, _ = build_backbones(paths, ctx)
    llm = build_frozen_llm(paths, ctx)

    logger.info("Creating datasets and dataloaders...")
    train_dataset, val_dataset = build_datasets(
        paths, train_params, loader_params, clip_processor, tokenizer
    )
    train_loader, val_loader = build_loaders(
        train_dataset, val_dataset, train_params, loader_params, ctx
    )

    vision_connector = build_vision_connector(vision_encoder, llm, ctx, trainable=True)

    accumulation_steps = train_params.get("gradient_accumulation_steps", 1)
    optimizer = optim.AdamW(
        vision_connector.parameters(),
        lr=train_params["learning_rate"],
        weight_decay=train_params.get("weight_decay", 0.01),
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=(len(train_loader) // accumulation_steps) * num_epochs
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    # Loss scaling is GPU-only; on CPU the loop runs in plain float32.
    scaler = GradScaler(ctx.amp_device, enabled=ctx.on_gpu)

    return Stage1Setup(
        llm=llm,
        vision_encoder=vision_encoder,
        vision_connector=vision_connector,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        loss_fn=loss_fn,
        accumulation_steps=accumulation_steps,
        vocab_size=llm.config.vocab_size,
    )


def main() -> None:
    """Assemble the Stage 1 run and drive the epoch loop."""
    config = load_config()
    paths = config["paths"]
    train_params = config["training_stage1"]
    loader_params = config["dataloader"]
    num_epochs = train_params["num_epochs"]
    output_dir = paths["output_dir"]

    ctx = build_run_context(
        distributed=False,
        seed=loader_params.get("data_seed", 42),
        stage_name="Stage 1",
    )
    logger.info("--- Initializing Stage 1: Vision Connector Training ---")
    logger.info("Using device: %s", ctx.device)

    setup = build_setup(paths, train_params, loader_params, num_epochs, ctx)

    os.makedirs(output_dir, exist_ok=True)
    latest_path = os.path.join(output_dir, "vision_connector_stage1_latest.pth")
    if os.path.exists(latest_path):
        logger.debug("💾 Loading saved weights from %s", latest_path)
        setup.vision_connector.load_state_dict(torch.load(latest_path, map_location=ctx.device))

    # torch.compile is skipped on CPU, where warm-up costs more than it saves.
    if ctx.on_gpu:
        setup.vision_connector = torch.compile(setup.vision_connector)

    metrics_path = os.path.join(output_dir, "loss_history_stage1.json")
    metrics_history: dict[str, list] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
    }
    best_val_loss = float("inf")

    logger.info("🚀 Starting training...")
    for epoch in range(num_epochs):
        avg_train_loss = train_one_epoch(setup, ctx, epoch, num_epochs)
        avg_val_loss = run_validation(setup, ctx, epoch, num_epochs)
        best_val_loss = save_checkpoints(setup, output_dir, avg_val_loss, best_val_loss)

        metrics_history["epoch"].append(epoch + 1)
        metrics_history["train_loss"].append(avg_train_loss)
        metrics_history["val_loss"].append(avg_val_loss)
        metrics_history["learning_rate"].append(setup.optimizer.param_groups[0]["lr"])
        with open(metrics_path, "w") as f:
            json.dump(metrics_history, f, indent=4)
        logger.info("✅ Metrics saved to %s", metrics_path)

    logger.info("✅ Training complete.")


if __name__ == "__main__":
    main()
