"""
Create side-by-side comparison visualizations of Stage 2 vs Stage 3 similarity matrices.

Reads the JSON outputs from cross_concept_similarity_matrix.py for both stages
and creates comparison plots with independent color scales for better interpretability.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_matrix_and_labels(stage_dir: str, layer: int):
    """Load similarity matrix and labels from JSON files."""
    matrix_file = Path(stage_dir) / f"similarity_matrix_layer{layer}.json"
    labels_file = Path(stage_dir) / f"labels_layer{layer}.json"

    with open(matrix_file) as f:
        matrix_data = json.load(f)
    with open(labels_file) as f:
        labels_data = json.load(f)

    matrix = np.array(matrix_data["matrix"])
    labels = labels_data["labels"]

    return matrix, labels


def create_comparison_plot(
    stage2_dir: str, stage3_dir: str, layer: int, output_dir: str, temperature: float = 0.01
):
    """Create side-by-side comparison of Stage 2 vs Stage 3 cross-modal similarities."""

    # Load data
    stage2_matrix, stage2_labels = load_matrix_and_labels(stage2_dir, layer)
    stage3_matrix, stage3_labels = load_matrix_and_labels(stage3_dir, layer)

    # Verify labels match
    if stage2_labels != stage3_labels:
        print("⚠️  Warning: Labels don't match between stages!")

    n = stage2_matrix.shape[0]
    half_n = n // 2

    # Extract cross-modal submatrices (txt rows × img columns)
    stage2_cross = stage2_matrix[half_n:, :half_n]
    stage3_cross = stage3_matrix[half_n:, :half_n]

    # Extract labels
    img_labels = [l.replace("img:", "") for l in stage2_labels[:half_n]]
    txt_labels = [l.replace("txt:", "") for l in stage2_labels[half_n:]]

    # Compute statistics
    print(f"\n📊 Layer {layer} Statistics:")
    print(f"   Stage 2 range: [{stage2_cross.min():.3f}, {stage2_cross.max():.3f}]")
    print(f"   Stage 3 range: [{stage3_cross.min():.3f}, {stage3_cross.max():.3f}]")
    print(f"   Stage 2 mean: {stage2_cross.mean():.3f}")
    print(f"   Stage 3 mean: {stage3_cross.mean():.3f}")

    # Create side-by-side figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

    # Compute INDEPENDENT color scales for maximum sensitivity per stage
    stage2_vmin = stage2_cross.min() - 0.005
    stage2_vmax = stage2_cross.max() + 0.005
    stage3_vmin = stage3_cross.min() - 0.005
    stage3_vmax = stage3_cross.max() + 0.005

    # Plot Stage 2 with its own scale
    sns.heatmap(
        stage2_cross,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=stage2_vmin,
        vmax=stage2_vmax,
        xticklabels=img_labels,
        yticklabels=txt_labels,
        ax=ax1,
        cbar_kws={"label": "Cosine Similarity (Stage 2)"},
        square=True,
        linewidths=0.5,
        linecolor="lightgray",
    )
    ax1.set_title(f"Stage 2: Hard Routing\nLayer {layer}", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Image Concepts", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Text Concepts", fontsize=11, fontweight="bold")

    # Plot Stage 3 with its own scale
    sns.heatmap(
        stage3_cross,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=stage3_vmin,
        vmax=stage3_vmax,
        xticklabels=img_labels,
        yticklabels=txt_labels,
        ax=ax2,
        cbar_kws={"label": "Cosine Similarity (Stage 3)"},
        square=True,
        linewidths=0.5,
        linecolor="lightgray",
    )
    ax2.set_title(
        f"Stage 3: Soft Routing (T={temperature})\nLayer {layer}", fontsize=12, fontweight="bold"
    )
    ax2.set_xlabel("Image Concepts", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Text Concepts", fontsize=11, fontweight="bold")

    # Overall title
    fig.suptitle(
        f"Compositional Case Study: Stage 2 vs Stage 3 (Layer {layer})",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plot_path = Path(output_dir) / f"comparison_cross_modal_layer{layer}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"   ✓ Saved comparison plot to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create Stage 2 vs Stage 3 comparison visualizations"
    )
    parser.add_argument(
        "--stage2-dir", type=str, required=True, help="Directory with Stage 2 results"
    )
    parser.add_argument(
        "--stage3-dir", type=str, required=True, help="Directory with Stage 3 results"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[0, 16, 31],
        help="Layers to compare (default: 0 16 31)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory for comparison plots"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.01,
        help="Temperature used for Stage 3 (for plot title)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Stage 2 vs Stage 3 Comparison Visualization")
    print("=" * 80)
    print(f"Stage 2 directory: {args.stage2_dir}")
    print(f"Stage 3 directory: {args.stage3_dir}")
    print(f"Layers: {args.layers}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)

    for layer in args.layers:
        print(f"\n{'=' * 80}")
        print(f"Processing Layer {layer}")
        print(f"{'=' * 80}")

        create_comparison_plot(
            stage2_dir=args.stage2_dir,
            stage3_dir=args.stage3_dir,
            layer=layer,
            output_dir=args.output_dir,
            temperature=args.temperature,
        )

    print(f"\n{'=' * 80}")
    print("✅ All comparison plots created!")
    print(f"📁 Saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
