"""Plotting and reporting for layer_clustering_analysis (extracted verbatim)."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_clustering_analysis(
    layer_idx: int,
    df: pd.DataFrame,
    coords_2d: np.ndarray,
    output_dir: str,
    expert_confidence_threshold: float = 0.6,
):
    """
    Generate 3 scatter plots with identical layout but different coloring.

    Args:
        layer_idx: Layer index being visualized
        df: DataFrame with metadata (concept, modality, expert_choice)
        coords_2d: 2D coordinates from dimensionality reduction [n_samples, 2]
        output_dir: Directory to save plots
    """
    print(f"   📊 Generating plots for Layer {layer_idx}...")

    os.makedirs(output_dir, exist_ok=True)

    # Common plot settings
    figsize = (10, 8)
    alpha = 0.6
    s = 50  # Point size

    # Plot 1: Color by Concept
    fig, ax = plt.subplots(figsize=figsize)
    concepts = df["concept"].unique()
    concept_colors = sns.color_palette("tab10", n_colors=len(concepts))

    for i, concept in enumerate(concepts):
        mask = df["concept"] == concept
        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            c=[concept_colors[i]],
            label=concept,
            alpha=alpha,
            s=s,
            edgecolors="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Dimension 1", fontsize=12)
    ax.set_ylabel("Dimension 2", fontsize=12)
    ax.set_title(f"Layer {layer_idx} - Colored by Concept", fontsize=14, fontweight="bold")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    concept_path = os.path.join(output_dir, f"layer_{layer_idx}_concept.png")
    plt.savefig(concept_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      ✅ Saved: {concept_path}")

    # Plot 2: Color by Modality
    fig, ax = plt.subplots(figsize=figsize)
    modalities = df["modality"].unique()
    modality_colors = {
        "vision": "#1f77b4",
        "text": "#ff7f0e",
    }  # Blue for vision, orange for text

    for modality in modalities:
        mask = df["modality"] == modality
        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            c=modality_colors[modality],
            label=modality,
            alpha=alpha,
            s=s,
            edgecolors="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Dimension 1", fontsize=12)
    ax.set_ylabel("Dimension 2", fontsize=12)
    ax.set_title(f"Layer {layer_idx} - Colored by Modality", fontsize=14, fontweight="bold")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    modality_path = os.path.join(output_dir, f"layer_{layer_idx}_modality.png")
    plt.savefig(modality_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      ✅ Saved: {modality_path}")

    # Plot 3: Color by Expert Choice with Confidence
    fig, ax = plt.subplots(figsize=figsize)

    # Base colors for each expert (will be modulated by confidence)
    base_colors = {
        "Expert 0": np.array([44, 160, 44]) / 255.0,  # Green
        "Expert 1": np.array([214, 39, 40]) / 255.0,  # Red
        "mixed": np.array([148, 103, 189]) / 255.0,  # Purple
        "unknown": np.array([127, 127, 127]) / 255.0,  # Gray
    }

    # Create scatter plot with confidence-modulated colors
    experts = df["expert_choice"].unique()

    for expert in experts:
        mask = df["expert_choice"] == expert
        base_color = base_colors.get(expert, base_colors["unknown"])

        # Get confidence values for this expert's points
        confidences = df.loc[mask, "routing_confidence"].values

        # Modulate color intensity by confidence
        # Low confidence (threshold) -> lighter, High confidence (1.0) -> full color
        colors = []
        threshold = expert_confidence_threshold
        conf_range = 1.0 - threshold

        for conf in confidences:
            # Blend between white (low confidence) and base color (high confidence)
            # Scale confidence from [threshold, 1.0] to [0.0, 1.0] for better visual range
            scaled_conf = (conf - threshold) / conf_range if conf >= threshold else 0.0
            scaled_conf = np.clip(scaled_conf, 0.0, 1.0)

            # Interpolate: (1-scaled_conf)*white + scaled_conf*base_color
            color = (1 - scaled_conf) * np.array([1.0, 1.0, 1.0]) + scaled_conf * base_color
            colors.append(color)

        scatter = ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            c=colors,
            label=expert,
            s=s,
            edgecolors="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Dimension 1", fontsize=12)
    ax.set_ylabel("Dimension 2", fontsize=12)
    ax.set_title(
        f"Layer {layer_idx} - Expert Selection with Routing Confidence",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper left", framealpha=0.9, title="Expert Choice")
    ax.grid(True, alpha=0.3)

    # Add colorbar for confidence
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    # Create a colormap that goes from white to gray (for visualization reference)
    # This represents the confidence gradient
    # Use the actual confidence threshold from the config
    norm = Normalize(vmin=expert_confidence_threshold, vmax=1.0)
    sm = ScalarMappable(cmap=plt.cm.Greys, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    cbar.set_label("Routing Confidence", rotation=270, labelpad=20, fontsize=11)

    plt.tight_layout()

    expert_path = os.path.join(output_dir, f"layer_{layer_idx}_expert.png")
    plt.savefig(expert_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      ✅ Saved: {expert_path}")


def generate_clustering_report(
    layer_idx: int,
    df: pd.DataFrame,
    metrics: dict[str, dict[str, float]],
    output_dir: str,
):
    """
    Generate markdown report with statistics and metrics.

    Args:
        layer_idx: Layer index
        df: DataFrame with all sample data
        metrics: Dict mapping label_type -> metrics dict
        output_dir: Directory to save report
    """
    print(f"   📝 Generating report for Layer {layer_idx}...")

    report_path = os.path.join(output_dir, f"layer_{layer_idx}_report.md")

    with open(report_path, "w") as f:
        f.write(f"# Layer {layer_idx} - Clustering Analysis Report\n\n")

        # Summary statistics
        f.write("## Summary Statistics\n\n")
        f.write(f"- **Total samples**: {len(df)}\n")
        f.write(f"- **Modalities**: {df['modality'].value_counts().to_dict()}\n")
        f.write(f"- **Concepts**: {df['concept'].value_counts().to_dict()}\n")
        f.write(f"- **Expert choices**: {df['expert_choice'].value_counts().to_dict()}\n\n")

        # Average routing confidence
        avg_confidence = df["routing_confidence"].mean()
        f.write(f"- **Average routing confidence**: {avg_confidence:.3f}\n\n")

        # Clustering metrics
        f.write("## Clustering Quality Metrics\n\n")
        f.write(
            "| Label Type | Silhouette Score | Davies-Bouldin Index | # Clusters | # Samples |\n"
        )
        f.write(
            "|------------|------------------|----------------------|------------|-----------|\n"
        )

        for label_type in ["concept", "modality", "expert"]:
            m = metrics[label_type]
            f.write(
                f"| {label_type.capitalize()} | {m['silhouette_score']:.4f} | {m['davies_bouldin_index']:.4f} | {m['n_clusters']} | {m['n_samples']} |\n"
            )

        f.write("\n")

        # Metric interpretation
        f.write("### Metric Interpretation\n\n")
        f.write(
            "- **Silhouette Score**: Measures how well samples are clustered (-1 to 1, higher is better)\n"
        )
        f.write("  - Score > 0.5: Strong clustering\n")
        f.write("  - Score 0.2-0.5: Moderate clustering\n")
        f.write("  - Score < 0.2: Weak clustering\n\n")
        f.write("- **Davies-Bouldin Index**: Measures cluster separation (lower is better)\n")
        f.write("  - Score < 1.0: Well-separated clusters\n")
        f.write("  - Score 1.0-2.0: Moderate separation\n")
        f.write("  - Score > 2.0: Poorly separated clusters\n\n")

        # Expert-Modality alignment analysis
        f.write("## Expert-Modality Alignment Analysis\n\n")

        # Create contingency table
        contingency = pd.crosstab(df["modality"], df["expert_choice"], normalize="index") * 100
        f.write("### Routing Pattern by Modality (% of samples)\n\n")

        # Try to use markdown format, fallback to string representation
        try:
            f.write(contingency.to_markdown())
        except ImportError:
            # Fallback if tabulate is not available
            f.write("```\n")
            f.write(str(contingency))
            f.write("\n```")
        f.write("\n\n")

        # Interpretation
        f.write("### Interpretation\n\n")
        if "Expert 0" in contingency.columns and "Expert 1" in contingency.columns:
            vision_to_0 = (
                contingency.loc["vision", "Expert 0"] if "vision" in contingency.index else 0
            )
            vision_to_1 = (
                contingency.loc["vision", "Expert 1"] if "vision" in contingency.index else 0
            )
            text_to_0 = contingency.loc["text", "Expert 0"] if "text" in contingency.index else 0
            text_to_1 = contingency.loc["text", "Expert 1"] if "text" in contingency.index else 0

            f.write(
                f"- Vision tokens: {vision_to_0:.1f}% → Expert 0, {vision_to_1:.1f}% → Expert 1\n"
            )
            f.write(f"- Text tokens: {text_to_0:.1f}% → Expert 0, {text_to_1:.1f}% → Expert 1\n\n")

            # Determine specialization pattern
            if vision_to_0 > 70 and text_to_1 > 70:
                f.write(
                    "**Strong modality specialization detected**: Vision → Expert 0, Text → Expert 1\n"
                )
            elif vision_to_1 > 70 and text_to_0 > 70:
                f.write(
                    "**Strong modality specialization detected**: Vision → Expert 1, Text → Expert 0\n"
                )
            elif max(vision_to_0, vision_to_1) < 60 or max(text_to_0, text_to_1) < 60:
                f.write(
                    "**Weak modality specialization**: Routing is relatively balanced across experts\n"
                )
            else:
                f.write(
                    "**Moderate modality specialization**: Some preference but not strongly separated\n"
                )

    print(f"      ✅ Saved: {report_path}")
