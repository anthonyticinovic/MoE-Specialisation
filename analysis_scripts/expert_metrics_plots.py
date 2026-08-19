# \!/usr/bin/env python3
"""Per-layer plots for ``plot_expert_metrics``.

How expert load, routing entropy, confidence and the visual/text split vary
across the model's layers within an epoch. The across-epoch views and the text
report live in ``expert_metrics_evolution_plots``.

Pure functions: take loaded metrics plus an output directory, write files.
"""

import logging
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from analysis_scripts._lib import set_publication_rcparams

logger = logging.getLogger(__name__)

set_publication_rcparams()


def _plot_load_single_epoch(ax, all_metrics, epochs, selected_layers, x_positions):
    """One epoch: overlapping bars, with a legend naming just the two experts."""
    epoch = epochs[0]
    metrics = all_metrics[epoch]
    expert_0_loads = []
    expert_1_loads = []
    for layer_idx in selected_layers:
        layer_data = metrics["per_layer"][layer_idx]
        load_dist = layer_data["expert_load_distribution"]
        expert_0_loads.append(load_dist.get("expert_0", 0))
        expert_1_loads.append(load_dist.get("expert_1", 0))

    # Overlapping bars: draw Expert 1 (behind) then Expert 0 (front).
    # Make front bar slightly narrower so the behind bar remains visible.
    width_back = 0.62
    width_front = 0.48
    ax.bar(
        x_positions,
        expert_1_loads,
        width_back,
        label="Expert 1",
        color="#ff7f0e",
        alpha=0.75,
        hatch="//",
        edgecolor="k",
        linewidth=0.6,
        zorder=2,
    )
    ax.bar(
        x_positions,
        expert_0_loads,
        width_front,
        label="Expert 0",
        color="#1f77b4",
        alpha=0.8,
        edgecolor="k",
        linewidth=0.8,
        zorder=3,
    )

    # Compact, vertically stacked legend for single-epoch view (inside axes)
    # Use representative patches to ensure compact layout
    p0 = mpatches.Patch(facecolor="#1f77b4", edgecolor="k", label="Expert 0", alpha=0.8)
    p1 = mpatches.Patch(
        facecolor="#ff7f0e", edgecolor="k", hatch="//", label="Expert 1", alpha=0.75
    )
    ax.legend(
        handles=[p0, p1],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.95),
        ncol=1,
        fontsize=10,
        frameon=True,
        framealpha=0.9,
        handlelength=1.6,
        handletextpad=0.6,
        borderaxespad=0.5,
    )


def _plot_load_multi_epoch(ax, all_metrics, epochs, selected_layers, x_positions):
    """Several epochs: grouped bars coloured by epoch, one handle per expert."""
    width = 0.35 / max(1, len(epochs))
    colors = plt.cm.viridis(np.linspace(0, 1, len(epochs)))
    handles = []
    for epoch_idx, (epoch, color) in enumerate(zip(epochs, colors, strict=True)):
        metrics = all_metrics[epoch]
        expert_0_loads = []
        expert_1_loads = []

        for layer_idx in selected_layers:
            layer_data = metrics["per_layer"][layer_idx]
            load_dist = layer_data["expert_load_distribution"]
            expert_0_loads.append(load_dist.get("expert_0", 0))
            expert_1_loads.append(load_dist.get("expert_1", 0))

        # Offset bars for each epoch
        offset = width * (epoch_idx - len(epochs) / 2 + 0.5)
        ax.bar(x_positions + offset, expert_0_loads, width, color=color, alpha=0.7)
        ax.bar(x_positions + offset, expert_1_loads, width, color=color, alpha=0.4, hatch="//")
        # Keep only a single handle per expert to avoid a huge legend
        if epoch_idx == 0:
            handles.append(mpatches.Patch(color=color, label="Expert 0", alpha=0.7))
            handles.append(mpatches.Patch(color=color, label="Expert 1", alpha=0.4))

    # Deduplicate labels and present a compact legend
    # Use a small font and single column to keep legend compact
    if handles:
        # Use vertical stacked legend and only two items (Expert 0 / Expert 1)
        # Construct two representative patches with standard colours and place
        # the legend inside the axes (upper-right) as a single column.
        p0 = mpatches.Patch(color="#1f77b4", label="Expert 0", alpha=0.7)
        p1 = mpatches.Patch(color="#ff7f0e", label="Expert 1", alpha=0.7, hatch="//")
        ax.legend(
            handles=[p0, p1],
            loc="upper right",
            bbox_to_anchor=(0.98, 0.95),
            ncol=1,
            fontsize=9,
            handlelength=1.6,
            frameon=True,
            framealpha=0.9,
        )


def plot_expert_load_distribution(
    all_metrics, output_dir, selected_layers=None, selected_epochs=None
):
    """
    Plot expert load distribution for specific layers across all epochs.
    Shows how work is distributed between expert_0 and expert_1 at selected layers.

    Args:
        selected_layers: List of layer indices to plot. If None, plots all layers.
        selected_epochs: Optional list of epochs to plot. If None, all epochs are used.
    """
    if selected_layers is None:
        selected_layers = [0, 7, 15, 23, 31]
    if selected_epochs is None:
        epochs = sorted(all_metrics.keys())
    else:
        epochs = selected_epochs

    # If only one epoch is requested, draw a compact grouped bar chart with a
    # small legend that only shows Expert 0 and Expert 1 (not per-epoch entries).
    fig, ax = plt.subplots(figsize=(12, 6))
    x_positions = np.arange(len(selected_layers))

    if len(epochs) == 1:
        _plot_load_single_epoch(ax, all_metrics, epochs, selected_layers, x_positions)
    else:
        _plot_load_multi_epoch(ax, all_metrics, epochs, selected_layers, x_positions)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Expert Load (%)")
    ax.set_title("Expert Load Distribution Across Layers (Stage 3)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"L{layer}" for layer in selected_layers], rotation=45, ha="right")
    # Note: legend is created per-branch above (single-epoch or multi-epoch).
    # Do not call a generic legend() here which would override branch-specific layout.
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "expert_load_distribution.png"))
    plt.close()
    logger.info("  Saved: expert_load_distribution.png")


def plot_routing_entropy(all_metrics, output_dir, selected_layers=None, selected_epochs=None):
    """
    Plot routing entropy for specific layers across all epochs.
    Lower entropy = more decisive/confident routing.

    Args:
        selected_layers: List of layer indices to plot. If None, uses default selection.
    """
    if selected_layers is None:
        selected_layers = [0, 7, 15, 23, 31]
    if selected_epochs is None:
        epochs = sorted(all_metrics.keys())
    else:
        epochs = selected_epochs
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(epochs)))
    for epoch, color in zip(epochs, colors, strict=True):
        entropies_across_layers = []

        for layer_idx in selected_layers:
            metrics = all_metrics[epoch]
            layer_data = metrics["per_layer"][layer_idx]
            entropies_across_layers.append(layer_data["avg_routing_entropy"])

        ax.plot(
            selected_layers,
            entropies_across_layers,
            label=f"Epoch {epoch}",
            color=color,
            marker="o",
            markersize=8,
            linewidth=2.5,
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Average Routing Entropy")
    ax.set_title("Routing Entropy Across Layers\n(Lower = More Decisive Routing)")
    ax.legend(loc="best")
    ax.set_xticks(selected_layers)
    ax.set_xticklabels([f"L{layer}" for layer in selected_layers], rotation=45, ha="right")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "routing_entropy.png"))
    plt.close()
    logger.info("  Saved: routing_entropy.png")


def plot_high_confidence_fraction(
    all_metrics, output_dir, selected_layers=None, selected_epochs=None
):
    """
    Plot high confidence routing fraction for specific layers across all epochs.
    Shows what fraction of routing decisions are made with >70% confidence.

    Args:
        selected_layers: List of layer indices to plot. If None, uses default selection.
    """
    if selected_layers is None:
        selected_layers = [0, 7, 15, 23, 31]
    if selected_epochs is None:
        epochs = sorted(all_metrics.keys())
    else:
        epochs = selected_epochs
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(epochs)))
    for epoch, color in zip(epochs, colors, strict=True):
        high_conf_across_layers = []

        for layer_idx in selected_layers:
            metrics = all_metrics[epoch]
            layer_data = metrics["per_layer"][layer_idx]
            high_conf_across_layers.append(layer_data["high_confidence_fraction"])

        ax.plot(
            selected_layers,
            high_conf_across_layers,
            label=f"Epoch {epoch}",
            color=color,
            marker="o",
            markersize=8,
            linewidth=2.5,
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("High Confidence Fraction")
    ax.set_title(
        "High Confidence Routing Fraction Across Layers\n(Fraction of Decisions with >70% "
        "Confidence)"
    )
    ax.legend(loc="best")
    ax.set_xticks(selected_layers)
    ax.set_xticklabels([f"L{layer}" for layer in selected_layers], rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "high_confidence_fraction.png"))
    plt.close()
    logger.info("  Saved: high_confidence_fraction.png")


def plot_visual_vs_text_routing(
    all_metrics, output_dir, selected_layers=None, selected_epochs=None
):
    """
    Plot visual vs text token routing patterns for specific layers.
    Shows what % of visual tokens go to expert_1 vs % of text tokens go to expert_1.
    This reveals modality-specific specialisation patterns.

    Args:
        selected_layers: List of layer indices to plot. If None, uses default selection.
    """
    if selected_layers is None:
        selected_layers = [0, 7, 15, 23, 31]
    if selected_epochs is None:
        epochs = sorted(all_metrics.keys())
    else:
        epochs = selected_epochs
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(epochs)))
    for epoch, color in zip(epochs, colors, strict=True):
        visual_expert1_across_layers = []
        text_expert1_across_layers = []

        for layer_idx in selected_layers:
            metrics = all_metrics[epoch]
            layer_data = metrics["per_layer"][layer_idx]
            routing = layer_data["visual_vs_text_routing"]

            # Get % of visual tokens going to expert_1
            if "visual" in routing and "expert_1" in routing["visual"]:
                visual_expert1_across_layers.append(routing["visual"]["expert_1"])
            else:
                visual_expert1_across_layers.append(0)

            # Get % of text tokens going to expert_1
            if "text" in routing and "expert_1" in routing["text"]:
                text_expert1_across_layers.append(routing["text"]["expert_1"])
            else:
                text_expert1_across_layers.append(0)

        # Plot with different markers for visual vs text
        ax.plot(
            selected_layers,
            visual_expert1_across_layers,
            label=f"Epoch {epoch} - Visual",
            color=color,
            marker="o",
            markersize=8,
            linewidth=2.5,
            linestyle="-",
        )
        ax.plot(
            selected_layers,
            text_expert1_across_layers,
            label=f"Epoch {epoch} - Text",
            color=color,
            marker="s",
            markersize=8,
            linewidth=2.5,
            linestyle="--",
            alpha=0.7,
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("% Tokens Routed to Expert 1")
    ax.set_title("Visual vs Text Token Routing Across Layers\n(% Routed to Expert 1)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2)
    ax.set_xticks(selected_layers)
    ax.set_xticklabels([f"L{layer}" for layer in selected_layers], rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "visual_vs_text_routing.png"))
    plt.close()
    logger.info("  Saved: visual_vs_text_routing.png")
