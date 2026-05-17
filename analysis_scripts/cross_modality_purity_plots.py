"""Plotting helpers for cross_modality_purity.

Extracted from ``cross_modality_purity.py`` (verbatim) to keep the analyzer
module focused on analysis. These are pure functions — they take data and an
output directory and write PNGs.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_metric(
    results: dict,
    concepts: list[str],
    layers: list[int],
    metric_key: str,
    output_dir: str,
    ylabel: str,
    title: str,
    filename: str,
    ylim: tuple = None,
):
    """Generic plotting function for metrics."""
    plt.figure(figsize=(12, 8))

    for concept in concepts:
        values = [results[metric_key][concept].get(f"layer_{layer}", 0.0) for layer in layers]
        plt.plot(layers, values, marker="o", linewidth=2, markersize=8, label=concept)

    if "Cosine" in ylabel:
        plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    plt.xlabel("Layer", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f"{title}\n(Vision vs Text Expert)", fontweight="bold")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    if ylim:
        plt.ylim(ylim)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    print(f"  ✓ Saved {filename}")


def plot_purity_matrices(matrices: dict, target_layers: list[int], output_dir: str):
    """Plot purity matrices as heatmaps for comparison across layers."""
    # Dynamically create subplots based on number of matrices actually computed
    num_matrices = len(matrices)
    fig, axes = plt.subplots(1, num_matrices, figsize=(6 * num_matrices, 5))

    # Handle case where only 1 matrix (axes won't be an array)
    if num_matrices == 1:
        axes = [axes]

    for idx, layer in enumerate(target_layers):
        if layer not in matrices:
            continue  # Skip if this layer wasn't computed
        matrix, labels = matrices[layer]

        # Create heatmap
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=-1,
            vmax=1,
            xticklabels=labels,
            yticklabels=labels,
            ax=axes[idx],
            cbar_kws={"label": "Cosine Similarity"},
            square=True,
        )
        axes[idx].set_title(f"Layer {layer}", fontsize=14, fontweight="bold")

    plt.suptitle(
        "Cross-Concept Purity Matrix (Mean-Pooled)", fontsize=16, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    output_path = os.path.join(output_dir, "purity_matrix_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved purity_matrix_comparison.png")


def plot_clip_connector_comparison(
    clip_matrix: np.ndarray,
    connector_matrix: np.ndarray,
    labels: list[str],
    output_dir: str,
    pooling: str = "mean",
):
    """
    Plot side-by-side comparison of CLIP vs connector similarity matrices.

    Args:
        clip_matrix: 2×2 similarity matrix for CLIP embeddings
        connector_matrix: 2×2 similarity matrix for connector embeddings
        labels: List of concept names
        output_dir: Directory to save plot
        pooling: "mean" or "cls" for plot labeling and filename
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Determine titles and filename based on pooling
    if pooling == "cls":
        clip_title = "Raw CLIP Embeddings\n(1024-dim, CLS token)"
        connector_title = "Post-Connector Embeddings\n(4096-dim, CLS token)"
        filename = "clip_vs_connector_comparison_cls.png"
        suptitle = "CLIP vs Vision Connector: CLS Token Comparison"
    else:  # mean pooling
        clip_title = "Raw CLIP Embeddings\n(1024-dim, mean-pooled)"
        connector_title = "Post-Connector Embeddings\n(4096-dim, mean-pooled)"
        filename = "clip_vs_connector_comparison.png"
        suptitle = "CLIP vs Vision Connector: Mean-Pooled Comparison"

    # Plot raw CLIP embeddings
    sns.heatmap(
        clip_matrix,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[0],
        cbar_kws={"label": "Cosine Similarity"},
        square=True,
    )
    axes[0].set_title(clip_title, fontsize=13, fontweight="bold")

    # Plot post-connector embeddings
    sns.heatmap(
        connector_matrix,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[1],
        cbar_kws={"label": "Cosine Similarity"},
        square=True,
    )
    axes[1].set_title(connector_title, fontsize=13, fontweight="bold")

    plt.suptitle(suptitle, fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {filename}")


def plot_token_variance(variance_results: dict, labels: list[str], output_dir: str):
    """
    Plot Level 1: Token-level variance analysis.

    Shows the standard deviation of pairwise token similarities within each image,
    comparing CLIP vs connector for both concepts.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    concepts = list(variance_results.keys())
    x = np.arange(len(concepts))
    width = 0.35

    # Extract standard deviations
    clip_stds = [variance_results[c]["clip"]["std"] for c in concepts]
    connector_stds = [variance_results[c]["connector"]["std"] for c in concepts]

    # Create grouped bar chart
    bars1 = ax.bar(x - width / 2, clip_stds, width, label="CLIP", color="#3498db", alpha=0.8)
    bars2 = ax.bar(
        x + width / 2, connector_stds, width, label="Connector", color="#e74c3c", alpha=0.8
    )

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    ax.set_xlabel("Concept", fontsize=12, fontweight="bold")
    ax.set_ylabel("Token Similarity Std Dev", fontsize=12, fontweight="bold")
    ax.set_title(
        "Level 1: Internal Token Diversity\n(Higher = More Diverse Spatial Structure)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(concepts, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    # Generate filename with concept names
    concept_suffix = "_".join(concepts)
    filename = f"token_variance_analysis_{concept_suffix}.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {filename}")


def plot_position_specific_similarity(position_results: dict, output_dir: str):
    """
    Plot Level 2: Position-specific similarity analysis.

    Shows cat-car similarity at each of the 257 token positions for both
    CLIP and connector, revealing if certain positions maintain better separation.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    positions = np.arange(257)
    clip_sims = position_results["clip_similarities"]
    connector_sims = position_results["connector_similarities"]
    labels = position_results["labels"]

    # Plot both lines
    ax.plot(positions, clip_sims, linewidth=2, label="CLIP", color="#3498db", alpha=0.8)
    ax.plot(positions, connector_sims, linewidth=2, label="Connector", color="#e74c3c", alpha=0.8)

    # Highlight CLS token (position 0)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5, linewidth=1.5)
    ax.text(
        0,
        ax.get_ylim()[1] * 0.95,
        "CLS",
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Add horizontal reference line at 0.5
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3)

    ax.set_xlabel("Token Position", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"Cosine Similarity ({labels[0]} vs {labels[1]})", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Level 2: Position-Specific Concept Similarity\n"
        f"({labels[0]} vs {labels[1]} across 257 tokens)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11, loc="best")
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    plt.tight_layout()

    # Generate filename with concept names
    concept_suffix = "_".join(labels)
    filename = f"position_specific_similarity_{concept_suffix}.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {filename}")


def plot_alignment_curves(
    curves: dict[str, dict[int, float]], output_dir: str, title_suffix: str = ""
):
    """Plot layer-by-layer alignment curves for multiple concept pairs.

    Args:
        curves: Dict mapping concept_name to {layer: similarity} dict
        output_dir: Directory to save plot
        title_suffix: Optional suffix for plot title (e.g., "Stage 3")
    """
    plt.figure(figsize=(14, 8))

    # Plot each concept's alignment curve
    for concept_name, similarities in curves.items():
        layers = sorted(similarities.keys())
        values = [similarities[l] for l in layers]
        plt.plot(
            layers,
            values,
            marker="o",
            linewidth=2.5,
            markersize=8,
            label=concept_name,
            alpha=0.8,
        )

    # Add reference lines
    plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1.5)
    plt.axhline(y=0.5, color="lightgray", linestyle=":", alpha=0.4)
    plt.axvline(x=-1, color="lightblue", linestyle="--", alpha=0.3, linewidth=1)

    # Annotations
    plt.text(
        -1,
        plt.ylim()[1] * 0.95,
        "Embedding",
        ha="center",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3),
    )

    plt.xlabel("Layer", fontsize=13, fontweight="bold")
    plt.ylabel("Cosine Similarity (Vision ↔ Text)", fontsize=13, fontweight="bold")
    title = "Cross-Modal Concept Alignment by Layer"
    if title_suffix:
        title += f" ({title_suffix})"
    plt.title(title, fontsize=15, fontweight="bold")

    plt.legend(loc="best", fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "stage3_alignment_curves.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved stage3_alignment_curves.png")
