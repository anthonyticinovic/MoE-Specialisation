#!/usr/bin/env python3
"""
Expert Routing Ablation Study for Stage 2 (Hard Routing)

Experiment: Evaluate whether expert specialization is meaningful by comparing
performance with normal routing vs. flipped routing.

Expected: Normal routing (vision=Expert 0, text=Expert 1) should have lower loss
than flipped routing (vision=Expert 1, text=Expert 0).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel

from models.custom_mistral import MistralMoEForCausalLM
from models.vl_connector import VisionLanguageConnector


def load_stage2_model(checkpoint_path, device="cuda"):
    """Load Stage 2 model with hard routing."""
    print("Loading Stage 2 model...")

    # Load CLIP vision encoder
    clip_model = CLIPVisionModel.from_pretrained("YOUR_PATH_HERE/models/clip-vit-large-patch14").to(
        device
    )

    # Load vision connector
    vision_connector = VisionLanguageConnector()
    connector_state = torch.load(
        "YOUR_PATH_HERE/outputs/vision_connector_stage1_best.pth", map_location="cpu"
    )
    vision_connector.load_state_dict(connector_state)
    vision_connector = vision_connector.to(device)

    # Load LLM
    llm = MistralMoEForCausalLM.from_pretrained(
        "YOUR_PATH_HERE/models/Mistral-7B-MoE", torch_dtype=torch.bfloat16, device_map=device
    )

    # Load checkpoint - exactly matching cross_concept_similarity_matrix.py pattern
    print(f"   Loading Stage 2 expert weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Check if this is a full checkpoint (dict with keys) or direct state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # FULL checkpoint format (with training state)
        print(f"   Detected FULL checkpoint format (epoch {checkpoint.get('epoch', 'unknown')})")
        llm.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        # PORTABLE/DIRECT state_dict format
        print("   Detected PORTABLE checkpoint format (state_dict only)")
        llm.load_state_dict(checkpoint, strict=False)

    print("   ✅ Stage 2 checkpoint loaded successfully")
    llm.eval()

    # Force hard routing mode (Stage 2 uses hard routing)
    print("   Setting MoE layers to hard routing mode...")
    for layer in llm.model.layers:
        if hasattr(layer.mlp, "routing_mode"):
            layer.mlp.routing_mode = "hard"

    # Load tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(
        "YOUR_PATH_HERE/models/Mistral-7B-v0.3", use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token for COCO_Loader

    processor = CLIPImageProcessor.from_pretrained("YOUR_PATH_HERE/models/clip-vit-large-patch14")

    return clip_model, vision_connector, llm, tokenizer, processor


def set_routing_mask(
    llm, vision_expert_id, text_expert_id, num_visual_tokens, num_text_tokens, device="cuda"
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
    device="cuda",
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

        # Get visual features
        visual_outputs = clip_model(pixel_values=pixel_values)
        visual_features = visual_outputs.last_hidden_state  # (1, 257, 1024)
        visual_embeds = vision_connector(visual_features).to(torch.bfloat16)  # (1, 257, 4096)

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


def run_routing_ablation(checkpoint_path, data_path, num_samples=100, device="cuda"):
    """
    Run routing ablation experiment.

    Args:
        checkpoint_path: Path to Stage 2 checkpoint
        data_path: Path to COCO validation data
        num_samples: Number of examples to evaluate
    """
    print("=" * 80)
    print("EXPERT ROUTING ABLATION STUDY - STAGE 2")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Num samples: {num_samples}")
    print(f"Device: {device}")
    print()

    # Load model
    clip_model, vision_connector, llm, tokenizer, processor = load_stage2_model(
        checkpoint_path, device
    )

    # Put models in eval mode
    clip_model.eval()
    vision_connector.eval()
    llm.eval()

    # Load data
    print(f"Loading data from {data_path}...")
    from torch.utils.data import DataLoader

    from data.COCO_loader import COCO_Loader

    val_dataset = COCO_Loader(
        image_dir="YOUR_PATH_HERE/datasets/coco/val2017",
        annotations_file="YOUR_PATH_HERE/datasets/coco/annotations/captions_val2017.json",
        clip_processor=processor,
        tokenizer=tokenizer,
        subset_fraction=1.0,
        split="val",
    )

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Collect losses
    normal_losses = []  # vision=0, text=1 (trained configuration)
    flipped_losses = []  # vision=1, text=0 (ablation)

    print(f"\nEvaluating {num_samples} samples...")
    print("  Normal routing: vision → Expert 0, text → Expert 1")
    print("  Flipped routing: vision → Expert 1, text → Expert 0")
    print()

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
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print("\nNormal Routing (vision=0, text=1):")
    print(f"  Mean Loss: {normal_mean:.4f} ± {normal_std:.4f}")
    print("\nFlipped Routing (vision=1, text=0):")
    print(f"  Mean Loss: {flipped_mean:.4f} ± {flipped_std:.4f}")
    print(f"\nΔ Loss (Flipped - Normal): {delta:+.4f} ({delta_percent:+.1f}%)")

    if delta > 0:
        print(f"\n✅ VALIDATION: Flipped routing has {delta_percent:.1f}% higher loss!")
        print("   This confirms that expert specialization is meaningful.")
    else:
        print("\n⚠️  UNEXPECTED: Flipped routing has lower loss!")
        print("   This suggests experts may not be specialized as expected.")

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

    output_dir = Path("results/routing_ablation")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "routing_ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {output_dir / 'routing_ablation_results.json'}")

    # Create visualization
    create_visualization(normal_losses, flipped_losses, output_dir)

    return results


def create_visualization(normal_losses, flipped_losses, output_dir):
    """Create clean box plot visualization comparing normal vs flipped routing."""

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
    print(f"📊 Visualization saved to: {output_dir / 'routing_ablation_comparison.png'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Expert routing ablation study")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="YOUR_PATH_HERE/outputs/stage2_checkpoints/llm_stage2_best.pth",
        help="Path to Stage 2 checkpoint",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="YOUR_PATH_HERE/datasets/coco",
        help="Path to COCO data directory",
    )
    parser.add_argument(
        "--num_samples", type=int, default=100, help="Number of samples to evaluate"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")

    args = parser.parse_args()

    results = run_routing_ablation(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        num_samples=args.num_samples,
        device=args.device,
    )
