"""Stage 3: end-to-end fine-tuning of self-attention, router and experts.

The final training stage. Starting from the Stage 2 experts and the Stage 1
vision connector, this trains self-attention, the router gate and the experts
together under learned soft routing on LLaVA-Instruct data, and records the
per-layer routing metrics the paper's analysis is built on.

Run from the repo root::

    torchrun --nproc_per_node=4 training_scripts/train_stage_3.py

A single-process CPU run (used by the demo) works too: FSDP, mixed precision
and FlashAttention are all skipped when CUDA is unavailable. Paths come from
``configs/training_config.yaml`` unless ``MOE_CONFIG`` points elsewhere.
"""

from __future__ import annotations

import datetime
import gc
import json
import logging
import os
import time
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.utils.common import (
    full_state_dict_context,
    load_config,
    register_moe_model,
    unwrap_model,
)
from training_scripts._lib import (
    ExpertUsageTracker,
    RunContext,
    build_backbones,
    build_distributed_loaders,
    build_run_context,
    build_vision_connector,
    combine_sequence,
    move_batch,
    save_expert_metrics,
    teardown,
    wrap_with_fsdp,
)

# Stage-3-only construction code. Imported from the module rather than through
# `_lib/__init__` because none of it is shared with the other four stages.
from training_scripts._lib.stage3_setup import (
    TrainingSetup,
    build_datasets,
    build_llm,
    configure_trainable_parameters,
    load_stage1_connector,
    load_stage2_experts,
    log_configuration,
    maybe_resume,
)

logger = logging.getLogger(__name__)

# NCCL default is 10 minutes, which is too short for FSDP checkpoint loading.
DIST_TIMEOUT = datetime.timedelta(minutes=60)
# Validation is capped so an epoch stays a sensible length on the full dataset.
MAX_VAL_BATCHES = 300
NUM_EXPERTS = 2


def _forward(
    setup: TrainingSetup,
    ctx: RunContext,
    images: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Project the image, prepend it to the text embeddings and run the LLM.

    The vision tower is frozen, so its forward runs under no_grad in both
    training and validation — building a graph through it would waste memory
    without producing any gradient.
    """
    with torch.no_grad():
        patch_embeddings = setup.vision_encoder(images).last_hidden_state
    visual_soft_tokens = setup.vision_connector(patch_embeddings)
    # The cached embedding layer avoids touching FSDP internals in the loop.
    text_embeddings = setup.embed_tokens_layer(input_ids)

    combined_embeddings, combined_attention_mask = combine_sequence(
        visual_soft_tokens, text_embeddings, attention_mask, ctx
    )
    outputs = setup.llm(inputs_embeds=combined_embeddings, attention_mask=combined_attention_mask)
    return outputs.logits, visual_soft_tokens.shape[1]


def _sequence_loss(
    setup: TrainingSetup,
    logits: torch.Tensor,
    num_visual_tokens: int,
    input_ids: torch.Tensor,
    labels: torch.Tensor | None,
) -> torch.Tensor:
    """Next-token cross-entropy over the text span only.

    Visual tokens are dropped from the logits first. With LLaVA labels the
    question tokens are already masked to -100, so only answer tokens
    contribute; with COCO every text token does.
    """
    if labels is not None:
        text_logits = logits[..., num_visual_tokens:, :].contiguous()
        text_logits = text_logits[..., :-1, :].contiguous()
        text_labels = labels.contiguous()[..., 1:].contiguous()
    else:
        text_logits = logits[..., num_visual_tokens:-1, :].contiguous()
        text_labels = input_ids[..., 1:].contiguous()

    return setup.loss_fn(text_logits.view(-1, setup.vocab_size), text_labels.view(-1))


def train_one_epoch(setup: TrainingSetup, ctx: RunContext, epoch: int, num_epochs: int) -> float:
    """Run one training epoch and return the mean loss per optimiser step."""
    setup.train_sampler.set_epoch(epoch)

    # Every rank must draw the same Gumbel noise for routing, or FSDP sees
    # divergent execution order across ranks.
    torch.manual_seed(42 + epoch)
    torch.cuda.manual_seed_all(42 + epoch)

    setup.llm.train()
    # The connector is frozen, so it stays in eval mode.
    setup.vision_connector.eval()

    total_train_loss = 0.0
    setup.optimizer.zero_grad()
    epoch_start_time = time.time()

    if ctx.is_main:
        logger.info("\n%s", "=" * 70)
        logger.info("📚 Starting Epoch %d/%d", epoch + 1, num_epochs)
        logger.info("%s", "=" * 70)

    steps_per_epoch = len(setup.train_loader) // setup.accumulation_steps

    for i, batch in enumerate(setup.train_loader):
        moved, has_labels = move_batch(batch, ctx.device)
        if has_labels:
            images, input_ids, attention_mask, labels = moved
        else:
            images, input_ids, attention_mask = moved
            labels = None

        with autocast(device_type=ctx.amp_device, dtype=torch.bfloat16, enabled=ctx.on_gpu):
            logits, num_visual_tokens = _forward(setup, ctx, images, input_ids, attention_mask)
            ce_loss = _sequence_loss(setup, logits, num_visual_tokens, input_ids, labels)
            loss = ce_loss / setup.accumulation_steps

        setup.scaler.scale(loss).backward()

        if (i + 1) % setup.accumulation_steps == 0 or (i + 1) == len(setup.train_loader):
            # .item() forces a host sync, so it is only called here — once per
            # optimiser step — rather than on every micro-batch, where it would
            # skew timing between ranks.
            total_train_loss += loss.item() * setup.accumulation_steps

            setup.scaler.unscale_(setup.optimizer)
            torch.nn.utils.clip_grad_norm_(setup.trainable_params, max_norm=1.0)
            setup.scaler.step(setup.optimizer)
            setup.scaler.update()
            setup.optimizer.zero_grad()
            setup.scheduler.step()

            if ctx.is_main:
                steps_done = (i + 1) // setup.accumulation_steps
                if steps_done % 50 == 0 or steps_done == 1 or steps_done == steps_per_epoch:
                    elapsed = time.time() - epoch_start_time
                    steps_per_sec = steps_done / elapsed if elapsed > 0 else 0.0
                    eta_minutes = int(
                        ((steps_per_epoch - steps_done) / steps_per_sec / 60)
                        if steps_per_sec > 0
                        else 0
                    )
                    logger.info(
                        "  [Step %4d/%d] Loss: %.4f | LR: %.2e | Speed: %.2f steps/s | ETA: %dmin",
                        steps_done,
                        steps_per_epoch,
                        total_train_loss / (i + 1),
                        setup.scheduler.get_last_lr()[0],
                        steps_per_sec,
                        eta_minutes,
                    )

    # Averaged over optimiser steps, not micro-batches.
    avg_train_loss = total_train_loss / steps_per_epoch
    if ctx.is_main:
        logger.info(
            "\n✅ Training complete: Avg Loss = %.4f | Time: %.2f min",
            avg_train_loss,
            (time.time() - epoch_start_time) / 60,
        )
    return avg_train_loss


def run_validation(setup: TrainingSetup, ctx: RunContext) -> tuple[float, dict | None, int]:
    """Validate and, on rank 0, collect the routing metrics.

    Every rank runs the loop so FSDP collectives stay matched, but each computes
    its own average and only rank 0's is used for model selection.
    """
    setup.llm.eval()
    setup.vision_connector.eval()
    total_val_loss = 0.0
    val_steps = 0

    expert_tracker = None
    if ctx.is_main:
        # Sized from the loaded model rather than the 7B/ViT-L constants, so the
        # tracker is also correct for the tiny models used by the CPU demo.
        expert_tracker = ExpertUsageTracker(
            num_layers=setup.llm.config.num_hidden_layers,
            num_experts=NUM_EXPERTS,
            visual_token_end=setup.num_visual_tokens - 1,
        )
        logger.debug("\n📊 Running validation (all ranks, max %d batches)...", MAX_VAL_BATCHES)

    with torch.no_grad():
        for i, batch in enumerate(setup.val_loader):
            if i >= MAX_VAL_BATCHES:
                break

            moved, has_labels = move_batch(batch, ctx.device)
            if has_labels:
                images, input_ids, attention_mask, labels = moved
            else:
                images, input_ids, attention_mask = moved
                labels = None

            with autocast(device_type=ctx.amp_device, dtype=torch.bfloat16, enabled=ctx.on_gpu):
                logits, num_visual_tokens = _forward(setup, ctx, images, input_ids, attention_mask)

                if expert_tracker is not None:
                    _collect_routing_metrics(setup, ctx, expert_tracker, logits)

                loss = _sequence_loss(setup, logits, num_visual_tokens, input_ids, labels)

            total_val_loss += loss.item()
            val_steps += 1

            if ctx.is_main and (i + 1) % 25 == 0:
                logger.info(
                    "  Validation progress: %d/%d batches | Avg Loss: %.4f",
                    i + 1,
                    MAX_VAL_BATCHES,
                    total_val_loss / val_steps,
                )

    avg_val_loss = total_val_loss / val_steps if val_steps > 0 else float("inf")
    expert_metrics = expert_tracker.compute_metrics() if expert_tracker is not None else None
    return avg_val_loss, expert_metrics, val_steps


def _collect_routing_metrics(
    setup: TrainingSetup, ctx: RunContext, tracker: ExpertUsageTracker, logits: torch.Tensor
) -> None:
    """Read each MoE layer's stored router logits into the tracker.

    MoELayer caches ``_last_router_logits`` on every forward, so the routing
    distribution can be recovered without a second pass.
    """
    model = unwrap_model(setup.llm)
    # Shape from the batch in hand — the final batch of an epoch may be short.
    batch_size, seq_len = logits.shape[:2]
    positions = torch.arange(seq_len, device=ctx.device).unsqueeze(0).expand(batch_size, -1)

    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, "_last_router_logits"):
            router_probs = torch.softmax(layer.mlp._last_router_logits, dim=-1)
            tracker.update(layer_idx, router_probs, positions)


def save_checkpoints(
    setup: TrainingSetup,
    ctx: RunContext,
    checkpoint_dir: str,
    epoch: int,
    avg_val_loss: float,
    best_val_loss: float,
) -> float:
    """Save the latest (and, if improved, best) checkpoint. Returns the new best.

    Two variants are written: a full checkpoint carrying optimiser and scheduler
    state, which is tied to the GPU count, and a portable one holding only model
    weights, which is what resumption uses.
    """
    dist.barrier()
    gc.collect()
    if ctx.on_gpu:
        torch.cuda.empty_cache()

    # Gathering the sharded parameters is collective — every rank must enter.
    with full_state_dict_context(setup.llm, rank0_only=True):
        llm_state_dict = setup.llm.state_dict()
    connector_state_dict = setup.vision_connector.state_dict()

    if ctx.is_main:
        checkpoint_start = time.time()
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Update the best before writing either file — see train_stage_2.py.
        improved = avg_val_loss < best_val_loss
        if improved:
            best_val_loss = avg_val_loss
        full_checkpoint = {
            "model_state_dict": llm_state_dict,
            "connector_state_dict": connector_state_dict,
            "optimizer_state_dict": setup.optimizer.state_dict(),
            "scheduler_state_dict": setup.scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "current_val_loss": avg_val_loss,
            "world_size": dist.get_world_size(),
        }
        portable_checkpoint = {
            "model_state_dict": llm_state_dict,
            "connector_state_dict": connector_state_dict,
            "epoch": epoch,
            "val_loss": avg_val_loss,
        }

        torch.save(full_checkpoint, os.path.join(checkpoint_dir, "llm_stage3_latest.pth"))
        torch.save(
            portable_checkpoint,
            os.path.join(checkpoint_dir, "llm_stage3_latest_portable.pth"),
        )
        logger.info("  ✅ Saved latest checkpoint (%.1fs)", time.time() - checkpoint_start)

        if improved:
            torch.save(full_checkpoint, os.path.join(checkpoint_dir, "llm_stage3_best.pth"))
            torch.save(
                portable_checkpoint,
                os.path.join(checkpoint_dir, "llm_stage3_best_portable.pth"),
            )
            logger.info("\n  🏆 NEW BEST MODEL! Val loss improved: %.4f", avg_val_loss)
        else:
            logger.debug("  ℹ️  Best val loss remains: %.4f", best_val_loss)

        del llm_state_dict, connector_state_dict, full_checkpoint, portable_checkpoint

    gc.collect()
    if ctx.on_gpu:
        torch.cuda.empty_cache()
    dist.barrier()
    return best_val_loss


def build_setup(
    paths: dict[str, Any],
    train_params: dict[str, Any],
    loader_params: dict[str, Any],
    num_epochs: int,
    ctx: RunContext,
) -> TrainingSetup:
    """Assemble every component the epoch loop needs, in dependency order.

    Order matters: the connector's dimensions come from the loaded backbones,
    the vocab size and embedding layer must be cached before FSDP wrapping, and
    the Stage 2 experts must be loaded after wrapping so the state dict is
    gathered through FSDP rather than applied to an unsharded model.
    """
    output_dir = paths["output_dir"]
    stage2_checkpoint_dir = os.path.join(output_dir, "stage2_checkpoints")

    vision_encoder, clip_processor, tokenizer, num_visual_tokens = build_backbones(paths, ctx)
    llm = build_llm(paths, train_params, ctx)
    configure_trainable_parameters(llm, vision_encoder, ctx)

    vision_connector = build_vision_connector(vision_encoder, llm, ctx, trainable=False)

    # Cache both before wrapping: reading llm.config or the embedding layer
    # through FSDP can trigger collectives on a single rank.
    vocab_size = llm.config.vocab_size
    embed_tokens_layer = llm.model.embed_tokens

    # Stage 3 shards without parameter offloading.
    llm = wrap_with_fsdp(llm, ctx, offload_params=None)
    load_stage2_experts(llm, stage2_checkpoint_dir, ctx)
    load_stage1_connector(vision_connector, output_dir, ctx)
    dist.barrier()

    if ctx.is_main:
        logger.info("Creating datasets and dataloaders...")
    train_dataset, val_dataset = build_datasets(
        paths, train_params, loader_params, clip_processor, tokenizer, ctx
    )
    train_loader, val_loader, train_sampler = build_distributed_loaders(
        train_dataset, val_dataset, train_params, loader_params, ctx, drop_last_train=True
    )

    accumulation_steps = train_params.get("gradient_accumulation_steps", 1)
    trainable_params = [p for p in llm.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=train_params["learning_rate"],
        weight_decay=train_params["weight_decay"],
        fused=False,
    )
    # GradScaler applies to float16 only; this stage trains in bfloat16, which
    # does not need loss scaling.
    scaler = GradScaler(ctx.amp_device, enabled=False)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=(len(train_loader) // accumulation_steps) * num_epochs
    )
    label_smoothing = train_params.get("label_smoothing", 0.0)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=label_smoothing)
    if ctx.is_main and label_smoothing > 0:
        logger.info("  ✅ Label smoothing: %s", label_smoothing)

    return TrainingSetup(
        llm=llm,
        vision_encoder=vision_encoder,
        vision_connector=vision_connector,
        embed_tokens_layer=embed_tokens_layer,
        train_loader=train_loader,
        val_loader=val_loader,
        train_sampler=train_sampler,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        loss_fn=loss_fn,
        trainable_params=trainable_params,
        vocab_size=vocab_size,
        num_visual_tokens=num_visual_tokens,
        accumulation_steps=accumulation_steps,
    )


def main() -> None:
    """Assemble the Stage 3 run and drive the epoch loop."""
    register_moe_model()

    config = load_config()
    paths = config["paths"]
    train_params = config["training_stage3"]
    loader_params = config["dataloader"]
    num_epochs = train_params["num_epochs"]
    output_dir = paths["output_dir"]
    stage3_checkpoint_dir = os.path.join(output_dir, "stage3_checkpoints")

    ctx = build_run_context(
        distributed=True,
        seed=loader_params.get("data_seed", 42),
        timeout=DIST_TIMEOUT,
        stage_name="Stage 3",
    )

    if ctx.is_main:
        logger.info("--- Initializing Stage 3 Training (End-to-End) ---")

    setup = build_setup(paths, train_params, loader_params, num_epochs, ctx)

    start_epoch = maybe_resume(setup.llm, setup.vision_connector, stage3_checkpoint_dir, ctx)
    best_val_loss = float("inf")

    metrics_path = os.path.join(output_dir, "training_metrics_stage3.json")
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
        logger.info(
            "Optimizing %d trainable parameters.",
            sum(p.numel() for p in setup.trainable_params),
        )
    log_configuration(setup, ctx, train_params, num_epochs, start_epoch)

    start_time = time.time()
    for epoch in range(start_epoch, num_epochs):
        avg_train_loss = train_one_epoch(setup, ctx, epoch, num_epochs)
        avg_val_loss, expert_metrics, val_steps = run_validation(setup, ctx)

        if ctx.is_main:
            if expert_metrics is not None:
                save_expert_metrics(expert_metrics, output_dir, epoch, setup.num_visual_tokens)

            logger.info("\n%s", "=" * 70)
            logger.info("Epoch [%d/%d] Complete", epoch + 1, num_epochs)
            logger.info("  Training Loss:   %.4f", avg_train_loss)
            logger.info("  Validation Loss: %.4f (rank 0, %d batches)", avg_val_loss, val_steps)
            logger.info("  Learning Rate:   %.2e", setup.scheduler.get_last_lr()[0])
            logger.info("%s\n", "=" * 70)

            metrics_history["epoch"].append(epoch + 1)
            metrics_history["train_loss"].append(avg_train_loss)
            metrics_history["val_loss"].append(avg_val_loss)
            metrics_history["learning_rate"].append(setup.optimizer.param_groups[0]["lr"])
            with open(metrics_path, "w") as f:
                json.dump(metrics_history, f, indent=4)
            logger.info("✅ Metrics saved to %s", metrics_path)

        best_val_loss = save_checkpoints(
            setup, ctx, stage3_checkpoint_dir, epoch, avg_val_loss, best_val_loss
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
