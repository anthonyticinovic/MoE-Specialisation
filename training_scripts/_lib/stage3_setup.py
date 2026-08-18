"""Model, data and checkpoint construction for Stage 3.

Stage 3 is the largest of the five scripts because it is the only one that
loads from two earlier stages, switches routing mode, configures three separate
dropouts and resumes from a portable checkpoint. Those steps are here so
``train_stage_3.py`` reads as the training loop it is.

This lives in ``_lib`` under its "or large enough to crowd a stage script"
clause rather than the shared-between-stages one, and is imported directly
rather than re-exported from ``_lib/__init__``: nothing here is shared, and the
other four stages should not see these names.
"""

from __future__ import annotations

import gc
import logging
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM

from data import COCO_Loader, LLaVA_Loader
from models.utils.checkpoints import load_matching_weights, state_dict_from
from models.utils.common import (
    full_state_dict_context,
    get_attn_implementation,
    get_model_dtype,
)
from training_scripts._lib.runtime import RunContext

logger = logging.getLogger(__name__)


@dataclass
class TrainingSetup:
    """Everything the epoch loop needs, assembled once before training."""

    llm: nn.Module
    vision_encoder: nn.Module
    vision_connector: nn.Module
    embed_tokens_layer: nn.Module
    train_loader: DataLoader
    val_loader: DataLoader
    train_sampler: DistributedSampler
    optimizer: optim.Optimizer
    scheduler: CosineAnnealingLR
    scaler: GradScaler
    loss_fn: nn.Module
    trainable_params: list[torch.Tensor]
    vocab_size: int
    num_visual_tokens: int
    accumulation_steps: int


def build_llm(paths: dict[str, Any], train_params: dict[str, Any], ctx: RunContext) -> nn.Module:
    """Load the MoE model and put it into soft routing with Stage 3 dropout.

    Gradient checkpointing is deliberately left off: under FSDP it interacts
    badly with the MoE layer's per-rank dummy expert pass and destabilises
    activations.
    """
    llm = AutoModelForCausalLM.from_pretrained(
        paths["moe_model_path"],
        trust_remote_code=True,
        local_files_only=True,
        dtype=get_model_dtype(),
        attn_implementation=get_attn_implementation(),
        low_cpu_mem_usage=True,
    )

    if ctx.is_main:
        logger.info("Setting MoE layers to 'soft' routing mode for Stage 3.")
    for layer in llm.model.layers:
        if hasattr(layer.mlp, "routing_mode"):
            layer.mlp.routing_mode = "soft"

    attention_dropout = train_params.get("attention_dropout", 0.0)
    expert_dropout = train_params.get("expert_dropout", 0.0)

    for layer in llm.model.layers:
        # Mistral names this `attention_dropout`; other variants use `dropout`.
        if hasattr(layer.self_attn, "attention_dropout"):
            layer.self_attn.attention_dropout = attention_dropout
        if hasattr(layer.self_attn, "dropout") and isinstance(
            layer.self_attn.dropout, (int, float)
        ):
            layer.self_attn.dropout = attention_dropout

        if hasattr(layer.mlp, "experts"):
            if not hasattr(layer.mlp, "expert_dropout"):
                layer.mlp.expert_dropout = nn.Dropout(expert_dropout)
            else:
                layer.mlp.expert_dropout.p = expert_dropout

    if ctx.is_main:
        logger.info("  ✅ Attention dropout: %s", attention_dropout)
        logger.info("  ✅ Expert dropout: %s", expert_dropout)
        logger.info("  ✅ Router dropout: 0.1 (pre-configured in MoE layer)")

    return llm


def configure_trainable_parameters(
    llm: nn.Module, vision_encoder: nn.Module, ctx: RunContext
) -> None:
    """Unfreeze self-attention, the router gate and the experts; freeze the rest.

    The vision connector stays frozen at its Stage 1 weights and the CLIP tower
    is never trained, so Stage 3 only moves the language-side parameters.
    """
    if ctx.is_main:
        logger.info("Preparing model for Stage 3: unfreezing self-attn, router and experts.")

    for param in llm.parameters():
        param.requires_grad = False

    for name, param in llm.named_parameters():
        if any(token in name for token in ("self_attn", "mlp.gate", "mlp.experts")):
            param.requires_grad = True
            if ctx.is_main and "layers.0" in name:
                logger.info("  Unfrozen: %s", name)

    for param in vision_encoder.parameters():
        param.requires_grad = False

    if ctx.is_main:
        trainable = sum(p.numel() for p in llm.parameters() if p.requires_grad)
        total = sum(p.numel() for p in llm.parameters())
        logger.info(
            "LLM: %s / %s parameters trainable (%.1f%%)",
            f"{trainable:,}",
            f"{total:,}",
            100 * trainable / total,
        )


def load_stage2_experts(llm: nn.Module, checkpoint_dir: str, ctx: RunContext) -> None:
    """Load the specialised experts produced by Stage 2.

    Rank 0 decides whether the file exists and broadcasts that decision, so no
    rank can take a different branch and desynchronise the collectives.
    """
    checkpoint_found = torch.tensor(0.0, device=ctx.device)
    checkpoint_path = os.path.join(checkpoint_dir, "llm_stage2_best.pth")

    if ctx.is_main and os.path.exists(checkpoint_path):
        checkpoint_found.fill_(1.0)
    dist.broadcast(checkpoint_found, src=0)

    if checkpoint_found.item() != 1.0:
        raise FileNotFoundError(
            f"Stage 2 checkpoint not found: {checkpoint_path}. "
            "Stage 3 cannot start without the trained experts."
        )

    if ctx.is_main:
        logger.debug("💾 Loading Stage 2 (expert) checkpoint: %s", checkpoint_path)

    # state_dict_from accepts both checkpoint shapes; load_matching_weights
    # raises rather than leaving the model on its Stage 0 weights.
    state_dict = state_dict_from(checkpoint_path)
    # rank0_only keeps loading agnostic to the number of GPUs.
    with full_state_dict_context(llm, rank0_only=True):
        missing_keys, _ = load_matching_weights(llm, state_dict, source=checkpoint_path)
        if ctx.is_main:
            if missing_keys:
                logger.debug("  ⚠️  Missing keys (%d): %s...", len(missing_keys), missing_keys[:5])
            else:
                logger.info("  ✅ All keys matched perfectly!")

    del state_dict
    gc.collect()
    dist.barrier()

    if ctx.is_main:
        logger.info("✅ Stage 2 checkpoint loaded successfully on all ranks.")


def load_stage1_connector(vision_connector: nn.Module, output_dir: str, ctx: RunContext) -> None:
    """Load the Stage 1 vision connector, warning loudly if it is absent."""
    connector_found = torch.tensor(0.0, device=ctx.device)
    weights_path = os.path.join(output_dir, "vision_connector_stage1_best.pth")

    if ctx.is_main and os.path.exists(weights_path):
        connector_found.fill_(1.0)
    dist.broadcast(connector_found, src=0)

    if connector_found.item() != 1.0:
        if ctx.is_main:
            logger.warning(
                "Stage 1 connector not found at %s — proceeding with a randomly "
                "initialised connector, which is not recommended for Stage 3.",
                weights_path,
            )
        return

    # device is a CUDA ordinal under FSDP and the string "cpu" otherwise.
    map_location = f"cuda:{ctx.device}" if ctx.use_fsdp else ctx.device
    vision_connector.load_state_dict(torch.load(weights_path, map_location=map_location))
    if ctx.is_main:
        logger.info("✅ Vision connector weights loaded successfully on all ranks.")


def build_datasets(
    paths: dict[str, Any],
    train_params: dict[str, Any],
    loader_params: dict[str, Any],
    clip_processor: Any,
    tokenizer: Any,
    ctx: RunContext,
) -> tuple[Any, Any]:
    """Build the train/val datasets for the configured dataset type."""
    dataset_type = train_params.get("dataset", "coco")
    seed = loader_params.get("data_seed", 42)

    if dataset_type == "llava":
        if ctx.is_main:
            logger.info("📚 Using LLaVA-Instruct-150K dataset (all Q&A pairs, multi-turn)")
        common = dict(
            annotations_file=paths["llava_annotations_file"],
            image_dir=paths["llava_image_dir"],
            clip_processor=clip_processor,
            tokenizer=tokenizer,
            val_fraction=0.2,
            seed=seed,
        )
        train_dataset = LLaVA_Loader(
            split="train", subset_fraction=train_params["subset_fraction"], **common
        )
        val_dataset = LLaVA_Loader(
            split="val",
            subset_fraction=train_params.get("val_subset_fraction", 1.0),
            **common,
        )
        return train_dataset, val_dataset

    if ctx.is_main:
        logger.info("📚 Using COCO captions dataset")
    common = dict(
        image_dir=paths["image_dir"],
        annotations_file=paths["annotations_file"],
        clip_processor=clip_processor,
        tokenizer=tokenizer,
        subset_fraction=train_params["subset_fraction"],
        seed=seed,
    )
    train_dataset = COCO_Loader(split="train", **common)
    val_dataset = COCO_Loader(
        split="val",
        val_subset_fraction=train_params.get("val_subset_fraction", 0.2),
        **common,
    )
    return train_dataset, val_dataset


def maybe_resume(
    llm: nn.Module,
    vision_connector: nn.Module,
    checkpoint_dir: str,
    ctx: RunContext,
) -> int:
    """Resume from the portable checkpoint if one exists; return the next epoch.

    Only model weights are restored. The optimiser and scheduler start fresh so
    the run can change dataset size, learning rate or GPU count without hitting
    state-shape mismatches.
    """
    portable_path = os.path.join(checkpoint_dir, "llm_stage3_latest_portable.pth")
    should_resume = 1.0 if ctx.is_main and os.path.exists(portable_path) else 0.0
    resume_tensor = torch.tensor([should_resume], dtype=torch.float32).to(ctx.device)
    dist.broadcast(resume_tensor, src=0)

    if resume_tensor.item() != 1.0:
        if ctx.is_main:
            logger.info("🏁 No portable checkpoint found. Starting training from scratch.")
        return 0

    if ctx.is_main:
        logger.debug("💾 Resuming Stage 3 from portable checkpoint: %s", portable_path)

    checkpoint = torch.load(portable_path, map_location="cpu")
    with full_state_dict_context(llm, rank0_only=True):
        load_matching_weights(llm, checkpoint["model_state_dict"], source=portable_path)
    vision_connector.load_state_dict(checkpoint["connector_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    if ctx.is_main:
        logger.info(
            "   📊 Checkpoint was at epoch %s with val_loss %.4f",
            checkpoint["epoch"],
            checkpoint.get("val_loss", float("inf")),
        )
        logger.info("   🔄 Optimizer and scheduler initialised fresh from the current config.")

    del checkpoint
    gc.collect()
    dist.barrier()

    if ctx.is_main:
        logger.info("✅ Resumed with fresh optimiser. Starting from epoch %d.", start_epoch)
    return start_epoch


def log_configuration(
    setup: TrainingSetup,
    ctx: RunContext,
    train_params: dict[str, Any],
    num_epochs: int,
    start_epoch: int,
) -> None:
    """Print the run's configuration once, from rank 0."""
    if not ctx.is_main:
        return

    world_size = dist.get_world_size()
    steps_per_epoch = len(setup.train_loader) // setup.accumulation_steps
    batch_size = train_params["batch_size"]

    logger.info("\n%s", "=" * 70)
    logger.info("🚀 STAGE 3 TRAINING CONFIGURATION")
    logger.info("%s", "=" * 70)
    logger.info("Dataset:               %s", train_params.get("dataset", "coco").upper())
    logger.info("Epochs:                %d (starting from epoch %d)", num_epochs, start_epoch)
    logger.info("Training batches:      %d", len(setup.train_loader))
    logger.info("Validation batches:    %d", len(setup.val_loader))
    logger.info("Batch size per GPU:    %d", batch_size)
    logger.info("Gradient accumulation: %d", setup.accumulation_steps)
    logger.info(
        "Effective batch size:  %d (batch_size × %d ranks × accum_steps)",
        batch_size * world_size * setup.accumulation_steps,
        world_size,
    )
    logger.info("Steps per epoch:       %d", steps_per_epoch)
    logger.info("Learning rate:         %.2e", train_params["learning_rate"])
    logger.info("Weight decay:          %s", train_params["weight_decay"])
    logger.info("Label smoothing:       %s", train_params.get("label_smoothing", 0.0))
    logger.info("Attention dropout:     %s", train_params.get("attention_dropout", 0.0))
    logger.info("Expert dropout:        %s", train_params.get("expert_dropout", 0.0))
    logger.info("Mixed precision:       %s", "bfloat16" if ctx.on_gpu else "disabled (CPU)")
    logger.info("Sharding:              %s", "FSDP" if ctx.on_gpu else "none (single process)")
    logger.info("%s\n", "=" * 70)
