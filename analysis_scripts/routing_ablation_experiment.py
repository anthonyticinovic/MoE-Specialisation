#!/usr/bin/env python3
"""
Expert Routing Ablation Study for Stage 2 (Hard Routing)

Experiment: Evaluate whether expert specialisation is meaningful by comparing
performance with normal routing vs. flipped routing.

Expected: Normal routing (vision=Expert 0, text=Expert 1) should have lower loss
than flipped routing (vision=Expert 1, text=Expert 0).
"""

import argparse
import logging
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from analysis_scripts._lib import load_stage2_models, load_training_config
from models.utils.common import get_device, setup_logging

logger = logging.getLogger(__name__)


def load_stage2_model(checkpoint_path, config, device=None):
    """Load Stage 2 model with hard routing via the shared _lib loader."""
    logger.info("Loading Stage 2 model...")
    models = load_stage2_models(config, device, stage2_checkpoint=checkpoint_path)
    return (
        models.vision_encoder,
        models.vision_connector,
        models.llm,
        models.tokenizer,
        models.clip_processor,
    )


def set_routing_mask(
    llm, vision_expert_id, text_expert_id, num_visual_tokens, num_text_tokens, device=None
):
    """
    Set routing masks for all MoE layers.

    Args:
        vision_expert_id: Expert ID for visual tokens (0 or 1)
        text_expert_id: Expert ID for text tokens (0 or 1)
        num_visual_tokens: Number of visual tokens
        num_text_tokens: Number of text tokens
    """
    # Create routing mask with batch dimension: [batch_size=1, total_seq_len]
    # Match the format from train_stage_2.py: shape [batch, seq_len]
    routing_mask = torch.cat(
        [
            torch.full((1, num_visual_tokens), vision_expert_id, dtype=torch.long, device=device),
            torch.full((1, num_text_tokens), text_expert_id, dtype=torch.long, device=device),
        ],
        dim=1,
    )

    # Set for all MoE layers
    for layer in llm.model.layers:
        layer.mlp.routing_mask = routing_mask

    return routing_mask


def compute_loss_single_example(
    clip_model,
    vision_connector,
    llm,
    pixel_values,
    input_ids,
    vision_expert_id,
    text_expert_id,
    device=None,
):
    """
    Compute loss for a single example with specified routing.

    Args:
        pixel_values: Preprocessed image tensor [1, 3, 224, 224]
        input_ids: Tokenized caption [1, seq_len]

    Returns:
        loss (float): Cross-entropy loss
    """
    with torch.no_grad():
        # Ensure inputs are on correct device and have batch dimension
        pixel_values = (
            pixel_values.unsqueeze(0).to(device)
            if pixel_values.dim() == 3
            else pixel_values.to(device)
        )
        input_ids = (
            input_ids.unsqueeze(0).to(device) if input_ids.dim() == 1 else input_ids.to(device)
        )

        # Get visual features. The projection is cast to the LLM's own dtype
        # (bfloat16 on GPU, float32 on CPU) rather than to a hardcoded
        # bfloat16, which made the whole script GPU-only.
        visual_outputs = clip_model(pixel_values=pixel_values)
        visual_features = visual_outputs.last_hidden_state  # (1, 257, 1024)
        visual_embeds = vision_connector(visual_features).to(llm.dtype)  # (1, 257, 4096)

        # Get text embeddings
        text_embeds = llm.get_input_embeddings()(input_ids)  # (1, seq_len, 4096)

        # Combine embeddings
        combined_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        # Create attention mask
        attention_mask = torch.ones(combined_embeds.shape[:2], device=device)

        # Set routing mask
        num_visual = visual_embeds.shape[1]
        num_text = text_embeds.shape[1]
        set_routing_mask(llm, vision_expert_id, text_expert_id, num_visual, num_text)

        # Create labels (mask visual tokens)
        labels = torch.cat(
            [
                torch.full((1, num_visual), -100, dtype=torch.long, device=device),  # Ignore visual
                input_ids,  # Predict text
            ],
            dim=1,
        )

        # Forward pass
        outputs = llm(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )

        return outputs.loss.item()


def run_routing_ablation(
    checkpoint_path,
    data_path,
    num_samples=100,
    device=None,
    training_config=None,
    image_dir=None,
    annotations_file=None,
    output_dir=None,
):
    """
    Run routing ablation experiment.

    Args:
        checkpoint_path: Path to Stage 2 checkpoint (None → default from config)
        data_path: Path to COCO root directory (expects val2017/ and annotations/)
        num_samples: Number of examples to evaluate
        device: Compute device (None → CUDA when available, else CPU)
        training_config: Path to training_config.yaml (None → MOE_CONFIG, then
            the repo default)
        image_dir: Image directory, overriding the COCO-root convention
        annotations_file: COCO-format captions JSON, overriding the convention
        output_dir: Where to write the JSON and the plot
    """
    config = load_training_config(training_config)
    device = device or get_device()

    logger.info("=" * 80)
    logger.info("EXPERT ROUTING ABLATION STUDY - STAGE 2")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {checkpoint_path or 'default (from training_config.yaml)'}")
    logger.info(f"Num samples: {num_samples}")
    logger.info(f"Device: {device}")
    logger.info("")

    # Load model
    clip_model, vision_connector, llm, tokenizer, processor = load_stage2_model(
        checkpoint_path, config, device
    )

    # Put models in eval mode
    clip_model.eval()
    vision_connector.eval()
    llm.eval()

    # Load data. By convention data_path is a COCO root holding val2017/ and
    # annotations/, defaulting to the parent of the training image_dir.
    # image_dir/annotations_file override that convention outright, which is
    # what lets the demo point this at its own fixtures.
    if data_path is None:
        data_path = str(Path(config["paths"]["image_dir"]).parent)
    coco_root = Path(data_path)
    image_dir = image_dir or str(coco_root / "val2017")
    annotations_file = annotations_file or str(coco_root / "annotations" / "captions_val2017.json")
    logger.info(f"Loading data from {image_dir}...")
    from torch.utils.data import DataLoader

    from data.COCO_loader import COCO_Loader

    val_dataset = COCO_Loader(
        image_dir=image_dir,
        annotations_file=annotations_file,
        clip_processor=processor,
        tokenizer=tokenizer,
        subset_fraction=1.0,
        split="val",
    )

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Collect losses
    normal_losses = []  # vision=0, text=1 (trained configuration)
    flipped_losses = []  # vision=1, text=0 (ablation)

    logger.info(f"\nEvaluating {num_samples} samples...")
    logger.info("  Normal routing: vision → Expert 0, text → Expert 1")
    logger.info("  Flipped routing: vision → Expert 1, text → Expert 0")
    logger.info("")

    for i, batch in enumerate(tqdm(val_loader, total=num_samples)):
        if i >= num_samples:
            break

        # COCO_Loader returns (image_processed, input_ids, attention_mask)
        pixel_values, input_ids, attention_mask = batch

        # Normal routing (vision=0, text=1)
        loss_normal = compute_loss_single_example(
            clip_model,
            vision_connector,
            llm,
            pixel_values,
            input_ids,
            vision_expert_id=0,
            text_expert_id=1,
            device=device,
        )
        normal_losses.append(loss_normal)

        # Flipped routing (vision=1, text=0)
        loss_flipped = compute_loss_single_example(
            clip_model,
            vision_connector,
            llm,
            pixel_values,
            input_ids,
            vision_expert_id=1,
            text_expert_id=0,
            device=device,
        )
        flipped_losses.append(loss_flipped)

    # Compute statistics
    normal_mean = np.mean(normal_losses)
    normal_std = np.std(normal_losses)
    flipped_mean = np.mean(flipped_losses)
    flipped_std = np.std(flipped_losses)

    delta = flipped_mean - normal_mean
    delta_percent = (delta / normal_mean) * 100

    # Results
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS")
    logger.info("=" * 80)
    logger.info("\nNormal Routing (vision=0, text=1):")
    logger.info(f"  Mean Loss: {normal_mean:.4f} ± {normal_std:.4f}")
    logger.info("\nFlipped Routing (vision=1, text=0):")
    logger.info(f"  Mean Loss: {flipped_mean:.4f} ± {flipped_std:.4f}")
    logger.info(f"\nΔ Loss (Flipped - Normal): {delta:+.4f} ({delta_percent:+.1f}%)")

    if delta > 0:
        logger.info(f"\nVALIDATION: Flipped routing has {delta_percent:.1f}% higher loss!")
        logger.info("   This confirms that expert specialisation is meaningful.")
    else:
        logger.warning("\n UNEXPECTED: Flipped routing has lower loss!")
        logger.info("   This suggests experts may not be specialised as expected.")

    # Save results
    results = {
        "num_samples": num_samples,
        "normal_routing": {
            "mean": float(normal_mean),
            "std": float(normal_std),
            "losses": [float(l) for l in normal_losses],
        },
        "flipped_routing": {
            "mean": float(flipped_mean),
            "std": float(flipped_std),
            "losses": [float(l) for l in flipped_losses],
        },
        "delta": {"absolute": float(delta), "percent": float(delta_percent)},
    }

    output_dir = Path(output_dir or "results/routing_ablation")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "routing_ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_dir / 'routing_ablation_results.json'}")

    # Create visualisation
    create_visualization(normal_losses, flipped_losses, output_dir)

    return results


def create_visualization(normal_losses, flipped_losses, output_dir):
    """Create clean box plot visualisation comparing normal vs flipped routing."""

    import numpy as np

    # Calculate statistics for display
    normal_mean = np.mean(normal_losses)
    normal_std = np.std(normal_losses)
    flipped_mean = np.mean(flipped_losses)
    flipped_std = np.std(flipped_losses)
    delta = flipped_mean - normal_mean
    delta_percent = (delta / normal_mean) * 100

    # Create single box plot with whiskers
    fig, ax = plt.subplots(figsize=(8, 6))

    box_data = [normal_losses, flipped_losses]
    bp = ax.boxplot(
        box_data,
        tick_labels=["Normal\n(Vision→E0, Text→E1)", "Flipped\n(Vision→E1, Text→E0)"],
        patch_artist=True,
        widths=0.6,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=8),
    )

    # Color the boxes
    bp["boxes"][0].set_facecolor("#3498db")
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor("#e74c3c")
    bp["boxes"][1].set_alpha(0.7)

    # Customize appearance
    for element in ["whiskers", "fliers", "caps"]:
        plt.setp(bp[element], color="#2c3e50", linewidth=1.5)
    plt.setp(bp["medians"], color="#2c3e50", linewidth=2)

    # Add grid
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Labels and title
    ax.set_ylabel("Cross-Entropy Loss", fontsize=13, fontweight="bold")
    ax.set_title("Expert Routing Swap Study", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / "routing_ablation_comparison.png", dpi=300, bbox_inches="tight")
    logger.info(f"Visualisation saved to: {output_dir / 'routing_ablation_comparison.png'}")


def main(argv: list[str] | None = None) -> dict:
    """Parse arguments and run the ablation. Returns the results dict.

    Split out of the ``__main__`` block so a test can call it: the training
    scripts were given a ``main()`` for the same reason, and this is the one
    analysis script the CPU demo executes.
    """
    parser = argparse.ArgumentParser(description="Expert routing ablation study")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to Stage 2 checkpoint (default: llm_stage2_best.pth from training_config.yaml)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to COCO root directory (default: parent of paths.image_dir in config)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=100, help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device (default: cuda when available, else cpu)"
    )
    parser.add_argument(
        "--training-config",
        type=str,
        default=None,
        help="Path to training config (default: $MOE_CONFIG, else configs/training_config.yaml)",
    )
    parser.add_argument(
        "--image-dir", type=str, default=None, help="Image directory (default: <data>/val2017)"
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default=None,
        help="COCO captions JSON (default: <data>/annotations/captions_val2017.json)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Default: results/routing_ablation"
    )

    args = parser.parse_args(argv)

    return run_routing_ablation(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        num_samples=args.num_samples,
        device=args.device,
        training_config=args.training_config,
        image_dir=args.image_dir,
        annotations_file=args.annotations,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    setup_logging()
    main()
