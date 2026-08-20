"""Stage 2.5: train the router, and nothing else.

The bridging stage. Stage 2 specialises the experts under a fixed,
position-derived mask — there is no learned router. Stage 3 needs one. Going
straight from a hard mask to end-to-end soft routing collapses routing onto a
single expert, so this stage trains only the gate, with the experts frozen,
until soft routing is stable.

Three things shape the objective: Gumbel-Softmax temperature annealed from 2.0
towards 1.0, a load-balancing term, and an entropy bonus that decays over
epochs — early exploration first, specialisation later.

Run from the repo root::

    torchrun --nproc_per_node=4 training_scripts/train_stage_2.5.py

A single-process CPU run works too. Paths come from
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
    unwrap_model,
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
    state_dict_from,
    teardown,
    wrap_with_fsdp,
)

logger = logging.getLogger(__name__)

# Validation is capped so an epoch stays a sensible length on the full dataset.
MAX_VAL_BATCHES = 75
# The router's gradients are clipped loosely: the gate is tiny and a tight bound
# stalls it before it escapes the uniform-routing fixed point.
ROUTER_GRAD_CLIP = 10.0
ROUTER_MONITOR_EVERY = 500


@dataclass
class Stage25Setup:
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
    hidden_size: int
    num_moe_layers: int
    load_balancing_coeff: float


def build_moe_llm(paths: dict[str, Any], ctx: RunContext) -> nn.Module:
    """Load the MoE model, switch it to soft routing, and freeze all but the gate."""
    llm = AutoModelForCausalLM.from_pretrained(
        paths["moe_model_path"],
        trust_remote_code=True,
        local_files_only=True,
        dtype=get_model_dtype(),
        attn_implementation=get_attn_implementation(),
        low_cpu_mem_usage=True,
    )

    if ctx.is_main:
        logger.info("Setting MoE layers to 'soft' routing mode for Stage 2.5.")
    for layer in llm.model.layers:
        if hasattr(layer.mlp, "routing_mode"):
            layer.mlp.routing_mode = "soft"

    if ctx.is_main:
        logger.info("Freezing experts, unfreezing routers.")
    for name, param in llm.named_parameters():
        param.requires_grad = "mlp.gate" in name

    return llm


def load_stage2_experts(llm: nn.Module, checkpoint_dir: str, ctx: RunContext) -> None:
    """Load the specialised experts produced by Stage 2."""
    checkpoint_path = os.path.join(checkpoint_dir, "llm_stage2_best.pth")
    if not broadcast_flag(os.path.exists(checkpoint_path), ctx):
        if ctx.is_main:
            logger.warning(
                "Stage 2 best checkpoint not found at %s. Starting from base weights.",
                checkpoint_path,
            )
        return

    if ctx.is_main:
        logger.debug("💾 Loading Stage 2 expert weights from %s", checkpoint_path)

    # state_dict_from accepts both checkpoint shapes; load_matching_weights
    # raises rather than leaving the model on its Stage 0 weights.
    state_dict = state_dict_from(checkpoint_path)
    with full_state_dict_context(llm, rank0_only=False):
        load_matching_weights(llm, state_dict, source=checkpoint_path)

    del state_dict
    gc.collect()
    dist.barrier()

    if ctx.is_main:
        logger.info("✅ Stage 2 'best' state loaded successfully.")


def reinitialise_router_gates(llm: nn.Module, ctx: RunContext, std: float = 0.1) -> None:
    """Reset every gate from scratch after loading the Stage 2 checkpoint.

    That checkpoint carries gate tensors that were never trained — Stage 2 uses
    hard routing — so they are discarded rather than fine-tuned; keeping them
    collapses routing onto a single expert. This is a deliberate design choice,
    not a workaround for a load bug.

    ``std`` defaults to 0.1, wider than ``MoELayer``'s construction-time 0.05:
    a wider init gives each gate a stronger initial modality preference, which
    empirically helps the routers escape the uniform-routing fixed point.
    Override it with ``training_stage2.5.router_init_std``.

    The per-layer work is ``MoELayer.initialize_gate``. This function used to
    inline its own copy, which is how the two came to disagree on the standard
    deviation and how the copy in the model file ended up called by nothing.
    """
    if ctx.is_main:
        logger.info("--- Re-initialising router gates before soft routing (std=%s) ---", std)

    for layer in unwrap_model(llm).model.layers:
        if hasattr(layer.mlp, "gate"):
            layer.mlp.initialize_gate(std=std, device=ctx.device)
            layer.mlp.gate.weight.requires_grad = True


def router_temperature(epoch: int) -> float:
    """Gumbel-Softmax temperature for this epoch.

    Starts at 2.0 and decays towards a floor of 1.0: high temperature softens
    the routing distribution and encourages exploration early on.
    """
    return max(1.0, 2.0 * (0.9**epoch))


def apply_router_temperature(llm: nn.Module, temperature: float) -> None:
    """Push the temperature onto every soft-routing layer.

    ``MoELayer.forward`` reads ``_forward_temperature``; ``temperature`` is set
    too because the analysis scripts read that name.
    """
    for layer in unwrap_model(llm).model.layers:
        if hasattr(layer.mlp, "temperature"):
            layer.mlp.temperature = temperature
        if getattr(layer.mlp, "routing_mode", None) == "soft":
            layer.mlp._forward_temperature = temperature


def router_regularisers(
    setup: Stage25Setup, combined_embeddings: torch.Tensor
) -> tuple[torch.Tensor | int, torch.Tensor | int]:
    """Sum the load-balancing loss and the entropy bonus across layers.

    Both are summed, not averaged — that is what enters the objective. The
    reported entropy divides by the layer count so it stays a per-layer mean
    within [0, ln(num_experts)].
    """
    total_load_balancing_loss: torch.Tensor | int = 0
    total_entropy_bonus: torch.Tensor | int = 0

    for layer in unwrap_model(setup.llm).model.layers:
        if hasattr(layer.mlp, "load_balancing_loss"):
            total_load_balancing_loss += layer.mlp.load_balancing_loss

        if hasattr(layer.mlp, "gate"):
            gate_logits = layer.mlp.gate(combined_embeddings.view(-1, setup.hidden_size))
            gate_probs = torch.softmax(gate_logits, dim=-1)
            entropy = -(gate_probs * torch.log(gate_probs + 1e-10)).sum(dim=-1).mean()
            total_entropy_bonus += entropy

    return total_load_balancing_loss, total_entropy_bonus


def clip_router_gradients(llm: nn.Module) -> None:
    """Clip the gate gradients, loosely."""
    router_params = [
        param
        for name, param in llm.named_parameters()
        if "mlp.gate" in name and param.requires_grad
    ]
    if router_params:
        torch.nn.utils.clip_grad_norm_(router_params, max_norm=ROUTER_GRAD_CLIP)


def log_router_analysis(
    setup: Stage25Setup,
    combined_embeddings: torch.Tensor,
    input_ids: torch.Tensor,
    num_visual: int,
    epoch: int,
    batch: int,
) -> None:
    """Compare how a sample visual and text token route, at a few layers.

    The specialisation score is the gap between the two — near zero means the
    router is ignoring modality, which is exactly the collapse this stage exists
    to prevent.
    """
    layers = unwrap_model(setup.llm).model.layers
    # First, middle and last layer of whatever depth the model actually has.
    sample_layers = sorted({0, len(layers) // 2, len(layers) - 1})

    with torch.no_grad():
        visual_idx = num_visual // 2
        visual_hidden = combined_embeddings[0:1, visual_idx : visual_idx + 1, :].float()
        text_idx = num_visual + (input_ids.shape[1] // 2)
        text_hidden = combined_embeddings[0:1, text_idx : text_idx + 1, :].float()

        logger.info("\n%s", "=" * 60)
        logger.info("Router Analysis - Batch %d | Epoch %d", batch, epoch + 1)
        logger.info("%s", "=" * 60)

        for layer_idx in sample_layers:
            gate = getattr(layers[layer_idx].mlp, "gate", None)
            if gate is None:
                continue
            visual_probs = torch.softmax(gate(visual_hidden.view(-1, setup.hidden_size)), dim=-1)
            text_probs = torch.softmax(gate(text_hidden.view(-1, setup.hidden_size)), dim=-1)
            specialisation = abs(visual_probs[0, 0] - text_probs[0, 0]).item()

            logger.info("\nLayer %2d:", layer_idx)
            logger.info(
                "  Visual token -> E0: %.3f, E1: %.3f", visual_probs[0, 0], visual_probs[0, 1]
            )
            logger.info("  Text   token -> E0: %.3f, E1: %.3f", text_probs[0, 0], text_probs[0, 1])
            logger.info(
                "  Specialisation score: %.3f (%s)",
                specialisation,
                "GOOD" if specialisation > 0.3 else "WEAK",
            )
        logger.info("%s\n", "=" * 60)


def train_one_epoch(
    setup: Stage25Setup, ctx: RunContext, epoch: int, num_epochs: int
) -> tuple[float, float, float, float, float]:
    """Train the router for one epoch.

    Returns the mean total, cross-entropy, load-balancing and entropy values,
    plus the temperature used.
    """
    setup.train_sampler.set_epoch(epoch)
    setup.llm.train()

    temperature = router_temperature(epoch)
    apply_router_temperature(setup.llm, temperature)
    if ctx.is_main:
        logger.info("  Router temperature for epoch %d: %.3f", epoch + 1, temperature)

    total_train_loss = total_ce_loss = total_lb_loss = 0.0
    entropy_sum, entropy_batches = 0.0, 0
    setup.optimizer.zero_grad()

    # Decays over epochs: explore first, allow specialisation later.
    entropy_coeff = 0.001 * (0.95**epoch)

    for i, (images, input_ids, attention_mask) in enumerate(setup.train_loader):
        images = images.to(ctx.device)
        input_ids = input_ids.to(ctx.device)
        attention_mask = attention_mask.to(ctx.device)

        with autocast(device_type=ctx.amp_device, dtype=torch.bfloat16, enabled=ctx.on_gpu):
            # Everything upstream of the router is frozen in this stage.
            with torch.no_grad():
                patch_embeddings = setup.vision_encoder(images).last_hidden_state
                visual_soft_tokens = setup.vision_connector(patch_embeddings)
                text_embeddings = setup.llm.model.embed_tokens(input_ids)

            combined_embeddings, combined_attention_mask = combine_sequence(
                visual_soft_tokens, text_embeddings, attention_mask, ctx
            )
            apply_router_temperature(setup.llm, temperature)

            outputs = setup.llm(
                inputs_embeds=combined_embeddings, attention_mask=combined_attention_mask
            )
            ce_loss = shifted_caption_loss(
                setup.loss_fn,
                outputs.logits,
                visual_soft_tokens.shape[1],
                input_ids,
                setup.vocab_size,
                device=ctx.device,
            )
            load_balancing_loss, entropy_bonus = router_regularisers(setup, combined_embeddings)

            loss = (
                ce_loss
                + setup.load_balancing_coeff * load_balancing_loss
                - entropy_coeff * entropy_bonus  # negative: the bonus is maximised
            ) / setup.accumulation_steps

        setup.scaler.scale(loss).backward()

        if (i + 1) % setup.accumulation_steps == 0 or (i + 1) == len(setup.train_loader):
            setup.scaler.unscale_(setup.optimizer)
            clip_router_gradients(setup.llm)
            setup.scaler.step(setup.optimizer)
            setup.scaler.update()
            setup.optimizer.zero_grad()
            setup.scheduler.step()

        if loss.item() > 0:
            total_train_loss += loss.item() * setup.accumulation_steps
            total_ce_loss += ce_loss.item()
            if isinstance(load_balancing_loss, torch.Tensor):
                total_lb_loss += load_balancing_loss.item()
            if isinstance(entropy_bonus, torch.Tensor):
                # Summed over layers for the loss; divide so the reported value
                # is a per-layer mean bounded by ln(num_experts).
                entropy_sum += entropy_bonus.item() / setup.num_moe_layers
                entropy_batches += 1

        if ctx.is_main and (i + 1) % ROUTER_MONITOR_EVERY == 0:
            log_router_analysis(
                setup,
                combined_embeddings,
                input_ids,
                visual_soft_tokens.shape[1],
                epoch,
                i + 1,
            )
        if ctx.is_main and (i + 1) % 100 == 0:
            logger.info("  Epoch %d, Batch [%d/%d]", epoch + 1, i + 1, len(setup.train_loader))

    batches = len(setup.train_loader)
    avg_entropy = entropy_sum / entropy_batches if entropy_batches else 0.0
    if ctx.is_main:
        logger.info(
            "Epoch [%d/%d] - Training Loss: %.4f | CE Loss: %.4f | LB Loss: %.4f | "
            "Entropy: %.4f | Temp: %.3f",
            epoch + 1,
            num_epochs,
            total_train_loss / batches,
            total_ce_loss / batches,
            total_lb_loss / batches,
            avg_entropy,
            temperature,
        )
    return (
        total_train_loss / batches,
        total_ce_loss / batches,
        total_lb_loss / batches,
        avg_entropy,
        temperature,
    )


def run_validation(setup: Stage25Setup, ctx: RunContext, epoch: int, num_epochs: int) -> float:
    """Validate on a capped number of batches; each rank keeps its own average."""
    setup.llm.eval()
    total_val_loss = 0.0
    val_steps = 0

    if ctx.is_main:
        logger.info("  Starting validation (max %d batches per GPU)...", MAX_VAL_BATCHES)

    with torch.no_grad():
        for i, (images, input_ids, attention_mask) in enumerate(setup.val_loader):
            if i >= MAX_VAL_BATCHES:
                break

            images = images.to(ctx.device)
            input_ids = input_ids.to(ctx.device)
            attention_mask = attention_mask.to(ctx.device)

            try:
                with autocast(device_type=ctx.amp_device, dtype=torch.bfloat16, enabled=ctx.on_gpu):
                    patch_embeddings = setup.vision_encoder(images).last_hidden_state
                    visual_soft_tokens = setup.vision_connector(patch_embeddings)
                    text_embeddings = setup.llm.model.embed_tokens(input_ids)
                    combined_embeddings, combined_attention_mask = combine_sequence(
                        visual_soft_tokens, text_embeddings, attention_mask, ctx
                    )
                    outputs = setup.llm(
                        inputs_embeds=combined_embeddings,
                        attention_mask=combined_attention_mask,
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
                val_steps += 1
            except Exception as exc:  # noqa: BLE001 - a bad batch must not kill the run
                if ctx.is_main:
                    logger.debug("⚠️  Validation batch %d failed: %s", i, exc)
                break

    avg_val_loss = total_val_loss / val_steps if val_steps > 0 else float("inf")
    if ctx.is_main:
        logger.info(
            "Epoch [%d/%d] - Validation Loss: %.4f (rank 0, %d batches)",
            epoch + 1,
            num_epochs,
            avg_val_loss,
            val_steps,
        )
    return avg_val_loss


def save_checkpoints(
    setup: Stage25Setup,
    ctx: RunContext,
    checkpoint_dir: str,
    epoch: int,
    avg_val_loss: float,
    best_val_loss: float,
) -> float:
    """Save the trained router weights. Returns the best validation loss.

    Only the trainable parameters are stored: the experts are unchanged from
    Stage 2, so repeating them would triple the checkpoint for nothing.
    """
    # Gathering sharded parameters is collective — every rank must enter.
    with full_state_dict_context(setup.llm, rank0_only=True):
        full_state_dict = setup.llm.state_dict()

    if ctx.is_main:
        router_weights = {
            name: weight
            for name, weight in full_state_dict.items()
            if setup.llm.get_parameter(name).requires_grad
        }
        # Update the best before writing either file — see train_stage_2.py.
        improved = avg_val_loss < best_val_loss
        if improved:
            best_val_loss = avg_val_loss
        checkpoint = {
            "model_state_dict": router_weights,
            "optimizer_state_dict": setup.optimizer.state_dict(),
            "scheduler_state_dict": setup.scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "current_val_loss": avg_val_loss,
        }

        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(checkpoint, os.path.join(checkpoint_dir, "llm_stage2_5_latest.pth"))

        if improved:
            best_path = os.path.join(checkpoint_dir, "llm_stage2_5_best.pth")
            torch.save(checkpoint, best_path)
            logger.info("🏆 New best model! Val loss: %.4f. Saved to %s", avg_val_loss, best_path)
    return best_val_loss


def resume_from_checkpoint(
    setup: Stage25Setup, checkpoint_dir: str, ctx: RunContext
) -> tuple[int, float]:
    """Restore router, optimiser and scheduler state if a checkpoint exists."""
    latest_path = os.path.join(checkpoint_dir, "llm_stage2_5_latest.pth")
    if not broadcast_flag(os.path.exists(latest_path), ctx):
        if ctx.is_main:
            logger.info("🏁 No 'latest' checkpoint found. Starting training from scratch.")
        return 0, float("inf")

    if ctx.is_main:
        logger.debug("💾 Resuming training from %s", latest_path)

    checkpoint = torch.load(latest_path, map_location="cpu")
    with full_state_dict_context(setup.llm, rank0_only=False):
        # Only the router weights are stored, so most keys are legitimately
        # missing — but not all of them.
        load_matching_weights(setup.llm, checkpoint["model_state_dict"], source=latest_path)
    setup.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    setup.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    del checkpoint
    gc.collect()
    dist.barrier()

    if ctx.is_main:
        logger.info("✅ Resumed successfully. Starting from epoch %d.", start_epoch)
    return start_epoch, best_val_loss


def build_setup(
    paths: dict[str, Any],
    train_params: dict[str, Any],
    loader_params: dict[str, Any],
    num_epochs: int,
    ctx: RunContext,
) -> Stage25Setup:
    """Assemble every component the epoch loop needs, in dependency order."""
    output_dir = paths["output_dir"]
    stage2_checkpoint_dir = os.path.join(output_dir, "stage2_checkpoints")

    vision_encoder, clip_processor, tokenizer, _ = build_backbones(paths, ctx)
    llm = build_moe_llm(paths, ctx)

    vocab_size = llm.config.vocab_size
    hidden_size = llm.config.hidden_size
    num_moe_layers = sum(1 for layer in llm.model.layers if hasattr(layer.mlp, "gate"))

    # Stage 2.5 shards without parameter offloading.
    llm = wrap_with_fsdp(llm, ctx, offload_params=None)
    load_stage2_experts(llm, stage2_checkpoint_dir, ctx)
    reinitialise_router_gates(llm, ctx, std=train_params.get("router_init_std", 0.1))

    # Gradient checkpointing is intentionally left disabled: combined with FSDP
    # and the MoE layer's per-rank dummy expert pass it produced unstable
    # activations, and only the small gate trains here anyway.
    vision_connector = build_vision_connector(vision_encoder, llm, ctx, trainable=False)
    stage1_path = os.path.join(output_dir, "vision_connector_stage1_best.pth")
    vision_connector.load_state_dict(torch.load(stage1_path, map_location=ctx.device))

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
    )
    scaler = GradScaler(ctx.amp_device, enabled=ctx.on_gpu)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=(len(train_loader) // accumulation_steps) * num_epochs
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    if ctx.is_main:
        logger.info(
            "Optimizing %d trainable router parameters.",
            sum(p.numel() for p in trainable_params),
        )

    return Stage25Setup(
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
        hidden_size=hidden_size,
        num_moe_layers=num_moe_layers,
        load_balancing_coeff=train_params.get("load_balancing_coeff", 0.01),
    )


def main() -> None:
    """Assemble the Stage 2.5 run and drive the epoch loop."""
    register_moe_model()

    config = load_config()
    paths = config["paths"]
    train_params = config["training_stage2.5"]
    loader_params = config["dataloader"]
    num_epochs = train_params["num_epochs"]
    output_dir = paths["output_dir"]
    checkpoint_dir = os.path.join(output_dir, "stage2_5_checkpoints")

    ctx = build_run_context(
        distributed=True,
        seed=loader_params.get("data_seed", 42),
        stage_name="Stage 2.5",
    )
    if ctx.is_main:
        logger.info("--- Initializing Stage 2.5 Training (Training the Router) ---")

    setup = build_setup(paths, train_params, loader_params, num_epochs, ctx)
    start_epoch, best_val_loss = resume_from_checkpoint(setup, checkpoint_dir, ctx)

    metrics_path = os.path.join(output_dir, "training_metrics_stage2.5.json")
    metrics_history: dict[str, list] = {
        "epoch": [],
        "train_loss": [],
        "train_ce_loss": [],
        "train_lb_loss": [],
        "val_loss": [],
        "learning_rate": [],
        "entropy": [],
        "temperature": [],
    }
    if ctx.is_main and start_epoch > 0 and os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics_history = json.load(f)

    if ctx.is_main:
        logger.info("🚀 Starting Stage 2.5 training from epoch %d...", start_epoch)

    for epoch in range(start_epoch, num_epochs):
        train_loss, ce_loss, lb_loss, entropy, temperature = train_one_epoch(
            setup, ctx, epoch, num_epochs
        )
        avg_val_loss = run_validation(setup, ctx, epoch, num_epochs)

        if ctx.is_main:
            metrics_history["epoch"].append(epoch + 1)
            metrics_history["train_loss"].append(train_loss)
            metrics_history["train_ce_loss"].append(ce_loss)
            metrics_history["train_lb_loss"].append(lb_loss)
            metrics_history["val_loss"].append(avg_val_loss)
            metrics_history["learning_rate"].append(setup.optimizer.param_groups[0]["lr"])
            metrics_history["entropy"].append(entropy)
            metrics_history["temperature"].append(temperature)
            with open(metrics_path, "w") as f:
                json.dump(metrics_history, f, indent=4)
            logger.info("✅ Metrics saved to %s", metrics_path)

        best_val_loss = save_checkpoints(
            setup, ctx, checkpoint_dir, epoch, avg_val_loss, best_val_loss
        )

    teardown()
    logger.info("Job finished.")


if __name__ == "__main__":
    main()
