"""Plotting helpers for attention_routing_analysis (extracted verbatim)."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np


def plot_attention_routing_evolution(
    layer_metrics: dict[int, dict[str, list[float]]], output_dir: str
):
    """
    Generate line plots showing how attention metrics evolve across layers.

    Args:
        layer_metrics: Dict mapping layer_idx -> metric_name -> values
        output_dir: Directory to save plots
    """
    print("\n📊 Generating attention-routing evolution plots...")
    os.makedirs(output_dir, exist_ok=True)

    num_layers = len(layer_metrics)
    layers = np.arange(num_layers)

    # Compute mean and std for each metric at each layer
    def get_mean_std(metric_name):
        means = []
        stds = []
        for layer_idx in range(num_layers):
            values = layer_metrics[layer_idx][metric_name]
            if len(values) > 0:
                means.append(np.mean(values))
                stds.append(np.std(values))
            else:
                means.append(0.0)
                stds.append(0.0)
        return np.array(means), np.array(stds)

    # Plot 1: Attention Patterns (Cross-Modal + Intra-Modal)
    fig, ax = plt.subplots(figsize=(12, 6))

    # Cross-modal attention
    means_t2v, stds_t2v = get_mean_std("text_to_vision_attention")
    ax.plot(
        layers,
        means_t2v,
        linewidth=2.5,
        label="Text → Vision (cross-modal)",
        color="#1f77b4",
        linestyle="-",
    )
    ax.fill_between(layers, means_t2v - stds_t2v, means_t2v + stds_t2v, alpha=0.3, color="#1f77b4")

    # Intra-modal attention (text only)
    means_t2t, stds_t2t = get_mean_std("text_to_text_attention")
    ax.plot(
        layers,
        means_t2t,
        linewidth=2,
        label="Text → Text (intra-modal)",
        color="#ff7f0e",
        linestyle="--",
    )
    ax.fill_between(layers, means_t2t - stds_t2t, means_t2t + stds_t2t, alpha=0.3, color="#ff7f0e")

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Mean Attention Mass", fontsize=12)
    ax.set_title("Attention Patterns Across Layers", fontsize=14, fontweight="bold")
    ax.legend(loc="best", framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, num_layers - 1)
    # Auto-scale y-axis with margin, minimum at 0
    y_max = max(max(means_t2v + stds_t2v), max(means_t2t + stds_t2t))
    ax.set_ylim(0, min(1.0, y_max * 1.1))  # Cap at 1.0

    # Add note about self-attention
    ax.text(
        0.98,
        0.02,
        "Note: Intra-modal excludes self-attention",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        alpha=0.7,
        style="italic",
    )

    plt.tight_layout()

    plot_path = os.path.join(output_dir, "attention_patterns.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved: {plot_path}")

    # Plot 3: Attention Focus (Entropy)
    fig, ax = plt.subplots(figsize=(12, 6))

    means_text_ent, stds_text_ent = get_mean_std("text_attention_entropy")
    means_vision_ent, stds_vision_ent = get_mean_std("vision_attention_entropy")

    ax.plot(layers, means_text_ent, linewidth=2, label="Text Attention Entropy", color="#d62728")
    ax.fill_between(
        layers,
        means_text_ent - stds_text_ent,
        means_text_ent + stds_text_ent,
        alpha=0.3,
        color="#d62728",
    )

    ax.plot(
        layers, means_vision_ent, linewidth=2, label="Vision Attention Entropy", color="#9467bd"
    )
    ax.fill_between(
        layers,
        means_vision_ent - stds_vision_ent,
        means_vision_ent + stds_vision_ent,
        alpha=0.3,
        color="#9467bd",
    )

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Attention Entropy (nats)", fontsize=12)
    ax.set_title("Attention Focus Across Layers", fontsize=14, fontweight="bold")
    ax.legend(loc="best", framealpha=0.9, title="Lower = more focused")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, num_layers - 1)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "attention_focus.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved: {plot_path}")

    print(f"\n✅ All plots saved to {output_dir}")


def plot_expert_attention_correlation(
    layer_token_data: dict[int, dict[str, list[float]]], output_dir: str
):
    """
    Generate plot showing token-level attention patterns grouped by expert routing.

    Plots mean text→vision attention for Expert 0 vs Expert 1 across all layers.

    Args:
        layer_token_data: Dict mapping layer_idx -> {'expert0_attentions': [...], 'expert1_attentions': [...]}
        output_dir: Directory to save plot
    """
    print("\n📊 Generating expert-attention correlation plot (token-level)...")

    num_layers = len(layer_token_data)
    layers = np.arange(num_layers)

    # For each layer, compute mean and std of token-level attention by expert
    expert0_means = []
    expert0_stds = []
    expert0_counts = []

    expert1_means = []
    expert1_stds = []
    expert1_counts = []

    for layer_idx in range(num_layers):
        # Expert 0 statistics
        expert0_attention = layer_token_data[layer_idx]["expert0_attentions"]
        if len(expert0_attention) > 0:
            expert0_means.append(np.mean(expert0_attention))
            expert0_stds.append(np.std(expert0_attention))
            expert0_counts.append(len(expert0_attention))
        else:
            expert0_means.append(np.nan)
            expert0_stds.append(np.nan)
            expert0_counts.append(0)

        # Expert 1 statistics
        expert1_attention = layer_token_data[layer_idx]["expert1_attentions"]
        if len(expert1_attention) > 0:
            expert1_means.append(np.mean(expert1_attention))
            expert1_stds.append(np.std(expert1_attention))
            expert1_counts.append(len(expert1_attention))
        else:
            expert1_means.append(np.nan)
            expert1_stds.append(np.nan)
            expert1_counts.append(0)

    # Convert to arrays
    expert0_means = np.array(expert0_means)
    expert0_stds = np.array(expert0_stds)
    expert0_counts = np.array(expert0_counts)

    expert1_means = np.array(expert1_means)
    expert1_stds = np.array(expert1_stds)
    expert1_counts = np.array(expert1_counts)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Helper function to plot continuous segments
    def plot_continuous_segments(layers, means, stds, color, linestyle, label):
        """Plot line in continuous segments, skipping NaN values."""
        valid = ~np.isnan(means)

        # Find continuous segments
        segments = []
        start = None
        for i in range(len(valid)):
            if valid[i]:
                if start is None:
                    start = i
            else:
                if start is not None:
                    segments.append((start, i))
                    start = None
        if start is not None:
            segments.append((start, len(valid)))

        # Plot each continuous segment
        for seg_start, seg_end in segments:
            seg_layers = layers[seg_start:seg_end]
            seg_means = means[seg_start:seg_end]
            seg_stds = stds[seg_start:seg_end]

            # Plot line with uniform style
            ax.plot(
                seg_layers,
                seg_means,
                color=color,
                linestyle=linestyle,
                linewidth=2.5,
                solid_capstyle="round",
            )

            # Plot error band for this segment
            ax.fill_between(
                seg_layers, seg_means - seg_stds, seg_means + seg_stds, color=color, alpha=0.2
            )

    # Plot Expert 0 (blue, solid)
    plot_continuous_segments(layers, expert0_means, expert0_stds, "#1f77b4", "-", "Expert 0")

    # Plot Expert 1 (red, dashed)
    plot_continuous_segments(layers, expert1_means, expert1_stds, "#d62728", "--", "Expert 1")

    # Add legend with dummy lines (full opacity for legend)
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="#1f77b4", linewidth=2.5, linestyle="-", label="Expert 0"),
        Line2D([0], [0], color="#d62728", linewidth=2.5, linestyle="--", label="Expert 1"),
    ]
    ax.legend(handles=legend_elements, loc="best", framealpha=0.9, fontsize=11)

    # Labels and formatting
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Mean Text → Vision Attention", fontsize=12)
    ax.set_title("Expert-Specific Attention Patterns Across Layers", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, num_layers - 1)
    ax.set_ylim(0, 1.0)

    # Add note about token-level analysis
    ax.text(
        0.98,
        0.02,
        "Note: Token-level analysis. Each point = one text token's attention to vision.",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        alpha=0.7,
        style="italic",
    )

    plt.tight_layout()

    plot_path = os.path.join(output_dir, "expert_attention_correlation.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved: {plot_path}")

    # Print summary statistics
    print("\n📈 Expert-Attention Correlation Summary (Token-Level):")
    print(f"   Expert 0 tokens per layer: {expert0_counts.mean():.0f} ± {expert0_counts.std():.0f}")
    print(f"   Expert 1 tokens per layer: {expert1_counts.mean():.0f} ± {expert1_counts.std():.0f}")
    print(f"   Total tokens analyzed: {expert0_counts.sum() + expert1_counts.sum():.0f}")

    # Report layers with zero tokens for either expert
    zero_e0_layers = np.where(expert0_counts == 0)[0]
    zero_e1_layers = np.where(expert1_counts == 0)[0]
    if len(zero_e0_layers) > 0:
        print(f"   ⚠️  Layers with ZERO Expert 0 tokens: {list(zero_e0_layers)}")
    if len(zero_e1_layers) > 0:
        print(f"   ⚠️  Layers with ZERO Expert 1 tokens: {list(zero_e1_layers)}")

    # Report expert balance
    total_e0 = expert0_counts.sum()
    total_e1 = expert1_counts.sum()
    if total_e0 + total_e1 > 0:
        e0_fraction = total_e0 / (total_e0 + total_e1)
        e1_fraction = total_e1 / (total_e0 + total_e1)
        print(f"   Expert routing balance: E0={e0_fraction:.1%}, E1={e1_fraction:.1%}")

    # Find layers with largest difference (only layers with data for both experts)
    valid_e0 = ~np.isnan(expert0_means)
    valid_e1 = ~np.isnan(expert1_means)
    valid_both = valid_e0 & valid_e1
    if valid_both.any():
        differences = np.abs(expert0_means[valid_both] - expert1_means[valid_both])
        max_diff_idx = layers[valid_both][np.argmax(differences)]
        max_diff = differences.max()
        print(f"   Largest attention difference: Layer {max_diff_idx} ({max_diff:.3f})")
        print(f"      Expert 0: {expert0_means[max_diff_idx]:.3f}")
        print(f"      Expert 1: {expert1_means[max_diff_idx]:.3f}")
