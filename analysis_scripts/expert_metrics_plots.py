# \!/usr/bin/env python3
"""Plotting and reporting functions for plot_expert_metrics.

Extracted verbatim from ``plot_expert_metrics.py`` to keep the CLI/orchestration
module small. Pure functions: take loaded metrics + an output dir, write files.
"""

import json
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from analysis_scripts._lib import set_publication_rcparams

set_publication_rcparams()


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
        bar_back = ax.bar(
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
        bar_front = ax.bar(
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

    else:
        # Multi-epoch plotting -- keep behaviour but produce compact deduped legend
        width = 0.35 / max(1, len(epochs))
        colors = plt.cm.viridis(np.linspace(0, 1, len(epochs)))
        handles = []
        labels = []
        for epoch_idx, (epoch, color) in enumerate(zip(epochs, colors)):
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
            h0 = ax.bar(x_positions + offset, expert_0_loads, width, color=color, alpha=0.7)
            h1 = ax.bar(
                x_positions + offset, expert_1_loads, width, color=color, alpha=0.4, hatch="//"
            )
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

    ax.set_xlabel("Layer")
    ax.set_ylabel("Expert Load (%)")
    ax.set_title("Expert Load Distribution Across Layers (Stage 3)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"L{l}" for l in selected_layers], rotation=45, ha="right")
    # Note: legend is created per-branch above (single-epoch or multi-epoch).
    # Do not call a generic legend() here which would override branch-specific layout.
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "expert_load_distribution.png"))
    plt.close()
    print("  ✅ Saved: expert_load_distribution.png")


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
    for epoch, color in zip(epochs, colors):
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
    ax.set_xticklabels([f"L{l}" for l in selected_layers], rotation=45, ha="right")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "routing_entropy.png"))
    plt.close()
    print("  ✅ Saved: routing_entropy.png")


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
    for epoch, color in zip(epochs, colors):
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
        "High Confidence Routing Fraction Across Layers\n(Fraction of Decisions with >70% Confidence)"
    )
    ax.legend(loc="best")
    ax.set_xticks(selected_layers)
    ax.set_xticklabels([f"L{l}" for l in selected_layers], rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "high_confidence_fraction.png"))
    plt.close()
    print("  ✅ Saved: high_confidence_fraction.png")


def plot_visual_vs_text_routing(
    all_metrics, output_dir, selected_layers=None, selected_epochs=None
):
    """
    Plot visual vs text token routing patterns for specific layers.
    Shows what % of visual tokens go to expert_1 vs % of text tokens go to expert_1.
    This reveals modality-specific specialization patterns.

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
    for epoch, color in zip(epochs, colors):
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
    ax.set_xticklabels([f"L{l}" for l in selected_layers], rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "visual_vs_text_routing.png"))
    plt.close()
    print("  ✅ Saved: visual_vs_text_routing.png")


def plot_specialization_evolution(all_metrics, output_dir, selected_epochs=None):
    """
    Plot how expert specialization evolves across epochs.
    Shows aggregate % of visual/text tokens routed to each expert over training.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    if selected_epochs is None:
        epochs = sorted(all_metrics.keys())
    else:
        epochs = selected_epochs

    # Extract aggregate routing patterns
    visual_to_expert0 = []
    visual_to_expert1 = []
    text_to_expert0 = []
    text_to_expert1 = []

    for epoch in epochs:
        metrics = all_metrics[epoch]
        agg = metrics["aggregate"]

        if "visual_routing" in agg:
            visual_to_expert0.append(agg["visual_routing"].get("expert_0", 0))
            visual_to_expert1.append(agg["visual_routing"].get("expert_1", 0))
        else:
            visual_to_expert0.append(0)
            visual_to_expert1.append(0)

        if "text_routing" in agg:
            text_to_expert0.append(agg["text_routing"].get("expert_0", 0))
            text_to_expert1.append(agg["text_routing"].get("expert_1", 0))
        else:
            text_to_expert0.append(0)
            text_to_expert1.append(0)

    # Plot 1: Visual Token Routing Evolution
    ax1.plot(
        epochs,
        visual_to_expert0,
        label="Expert 0",
        marker="o",
        linewidth=2.5,
        color="#1f77b4",
        markersize=10,
    )
    ax1.plot(
        epochs,
        visual_to_expert1,
        label="Expert 1",
        marker="s",
        linewidth=2.5,
        color="#ff7f0e",
        markersize=10,
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("% Visual Tokens Routed to Expert")
    ax1.set_title("Visual Token Routing Evolution\n(Aggregate Across All Layers)")
    ax1.legend()
    ax1.set_ylim(0, 100)
    ax1.set_xticks(epochs)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Text Token Routing Evolution
    ax2.plot(
        epochs,
        text_to_expert0,
        label="Expert 0",
        marker="o",
        linewidth=2.5,
        color="#1f77b4",
        markersize=10,
    )
    ax2.plot(
        epochs,
        text_to_expert1,
        label="Expert 1",
        marker="s",
        linewidth=2.5,
        color="#ff7f0e",
        markersize=10,
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("% Text Tokens Routed to Expert")
    ax2.set_title("Text Token Routing Evolution\n(Aggregate Across All Layers)")
    ax2.legend()
    ax2.set_ylim(0, 100)
    ax2.set_xticks(epochs)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "specialization_evolution.png"))
    plt.close()
    print("  ✅ Saved: specialization_evolution.png")


def plot_aggregate_summary(all_metrics, output_dir):
    """
    Plot aggregate summary statistics for the latest epoch.
    Shows overall expert utilization patterns (simplified version).
    """
    # Use the latest epoch
    latest_epoch = max(all_metrics.keys())
    metrics = all_metrics[latest_epoch]
    agg = metrics["aggregate"]

    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)

    colors_bar = ["#1f77b4", "#ff7f0e"]

    # Plot 1: Expert Load Distribution (Aggregate)
    ax1 = fig.add_subplot(gs[0, 0])
    experts = list(agg["expert_load_distribution"].keys())
    loads = list(agg["expert_load_distribution"].values())
    ax1.bar(experts, loads, color=colors_bar, alpha=0.7, edgecolor="black", linewidth=1.5)
    ax1.set_ylabel("Load (%)", fontsize=12)
    ax1.set_title("Aggregate Expert Load Distribution", fontsize=13, fontweight="bold")
    ax1.set_ylim(0, 100)
    for i, (expert, load) in enumerate(zip(experts, loads)):
        ax1.text(
            i, load + 3, f"{load:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11
        )
    ax1.grid(True, alpha=0.3, axis="y")

    # Plot 2: Visual Routing (Aggregate)
    ax2 = fig.add_subplot(gs[0, 1])
    if "visual_routing" in agg:
        visual_experts = list(agg["visual_routing"].keys())
        visual_loads = list(agg["visual_routing"].values())
        ax2.bar(
            visual_experts,
            visual_loads,
            color=colors_bar,
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )
        ax2.set_ylabel("% Visual Tokens", fontsize=12)
        ax2.set_title("Visual Token Routing", fontsize=13, fontweight="bold")
        ax2.set_ylim(0, 100)
        for i, (expert, load) in enumerate(zip(visual_experts, visual_loads)):
            ax2.text(
                i,
                load + 3,
                f"{load:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11,
            )
        ax2.grid(True, alpha=0.3, axis="y")

    # Plot 3: Text Routing (Aggregate)
    ax3 = fig.add_subplot(gs[0, 2])
    if "text_routing" in agg:
        text_experts = list(agg["text_routing"].keys())
        text_loads = list(agg["text_routing"].values())
        ax3.bar(
            text_experts, text_loads, color=colors_bar, alpha=0.7, edgecolor="black", linewidth=1.5
        )
        ax3.set_ylabel("% Text Tokens", fontsize=12)
        ax3.set_title("Text Token Routing", fontsize=13, fontweight="bold")
        ax3.set_ylim(0, 100)
        for i, (expert, load) in enumerate(zip(text_experts, text_loads)):
            ax3.text(
                i,
                load + 3,
                f"{load:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11,
            )
        ax3.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Aggregate Expert Metrics Summary (Stage 3)", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
    plt.savefig(os.path.join(output_dir, "aggregate_summary.png"))
    plt.close()
    print("  ✅ Saved: aggregate_summary.png")


def generate_report(all_metrics, output_dir):
    """Generate a text report summarizing key findings."""
    report_path = os.path.join(output_dir, "expert_metrics_report.txt")

    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("EXPERT UTILIZATION METRICS REPORT\n")
        f.write("=" * 80 + "\n\n")

        for epoch in sorted(all_metrics.keys()):
            metrics = all_metrics[epoch]
            agg = metrics["aggregate"]

            f.write(f"\nEPOCH {epoch}\n")
            f.write("-" * 40 + "\n")

            f.write("\n1. Expert Load Distribution (Aggregate):\n")
            for expert, load in agg["expert_load_distribution"].items():
                f.write(f"   {expert}: {load:.2f}%\n")

            f.write(f"\n2. Routing Entropy: {agg['avg_routing_entropy']:.4f}\n")
            f.write("   (Lower = more decisive routing)\n")

            f.write(f"\n3. High Confidence Fraction: {agg['high_confidence_fraction']:.2%}\n")
            f.write("   (Fraction with >70% confidence)\n")

            f.write("\n4. Visual Token Routing:\n")
            if "visual_routing" in agg:
                for expert, load in agg["visual_routing"].items():
                    f.write(f"   {expert}: {load:.2f}%\n")

            f.write("\n5. Text Token Routing:\n")
            if "text_routing" in agg:
                for expert, load in agg["text_routing"].items():
                    f.write(f"   {expert}: {load:.2f}%\n")

            # Compute specialization score
            if "visual_routing" in agg and "text_routing" in agg:
                visual_e0 = agg["visual_routing"].get("expert_0", 50)
                text_e0 = agg["text_routing"].get("expert_0", 50)
                specialization_divergence = abs(visual_e0 - text_e0)
                f.write(
                    f"\n6. Modality Specialization Divergence: {specialization_divergence:.2f}%\n"
                )
                f.write("   (Difference in expert_0 preference between modalities)\n")
                if specialization_divergence > 30:
                    f.write("   ✓ Strong modality specialization detected!\n")
                elif specialization_divergence > 15:
                    f.write("   ✓ Moderate modality specialization\n")
                else:
                    f.write("   ⚠ Weak modality specialization\n")

            f.write("\n" + "=" * 80 + "\n")

    print("  ✅ Saved: expert_metrics_report.txt")


def plot_modality_specialization_divergence(all_metrics, output_dir, selected_epochs=None):
    """
    Plot modality specialization divergence over epochs.
    Shows |Visual_Expert0% - Text_Expert0%| to quantify how differently
    experts handle visual vs text tokens.

    This is THE KEY METRIC for understanding modality-specific specialization.
    """
    if selected_epochs is None:
        epochs = sorted(all_metrics.keys())
    else:
        epochs = selected_epochs

    divergence_values = []

    for epoch in epochs:
        metrics = all_metrics[epoch]
        agg = metrics["aggregate"]

        if "visual_routing" in agg and "text_routing" in agg:
            visual_e0 = agg["visual_routing"].get("expert_0", 50)
            text_e0 = agg["text_routing"].get("expert_0", 50)
            divergence = abs(visual_e0 - text_e0)
            divergence_values.append(divergence)
        else:
            divergence_values.append(0)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        epochs,
        divergence_values,
        label="Specialization Divergence",
        marker="o",
        linewidth=3,
        color="#e74c3c",
        markersize=10,
    )

    # Add horizontal reference lines
    ax.axhline(
        y=30,
        color="green",
        linestyle="--",
        alpha=0.5,
        linewidth=2,
        label="Strong Specialization (>30%)",
    )
    ax.axhline(
        y=15,
        color="orange",
        linestyle="--",
        alpha=0.5,
        linewidth=2,
        label="Moderate Specialization (>15%)",
    )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Specialization Divergence (%)", fontsize=12)
    ax.set_title(
        "Modality Specialization Divergence Over Training\n|Visual Expert 0% - Text Expert 0%|",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=10)
    ax.set_xticks(epochs)
    ax.set_ylim(0, max(divergence_values) * 1.1 if divergence_values else 50)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "specialization_divergence.png"))
    plt.close()
    print("  ✅ Saved: specialization_divergence.png")


def plot_routing_confidence_evolution(all_metrics, output_dir, selected_epochs=None):
    """
    Plot routing entropy evolution over epochs.
    Shows average routing entropy (lower = more decisive routing).

    Shows that the model is learning meaningful routing patterns.
    """
    if selected_epochs is None:
        epochs = sorted(all_metrics.keys())
    else:
        epochs = selected_epochs

    entropy_values = []

    for epoch in epochs:
        metrics = all_metrics[epoch]
        agg = metrics["aggregate"]

        entropy_values.append(agg.get("avg_routing_entropy", 0))

    fig, ax = plt.subplots(figsize=(12, 7))

    # Routing Entropy plot
    ax.plot(
        epochs,
        entropy_values,
        label="Routing Entropy",
        marker="s",
        linewidth=3,
        color="#3498db",
        markersize=10,
    )
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Average Routing Entropy", fontsize=13)
    ax.set_title(
        "Routing Entropy Evolution\n(Lower = More Decisive Routing)", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(epochs)
    ax.set_ylim(0, max(entropy_values) * 1.1 if entropy_values else 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "routing_confidence_evolution.png"))
    plt.close()
    print("  ✅ Saved: routing_confidence_evolution.png")


def plot_loss_and_specialization(all_metrics, output_dir, metrics_json_path, selected_epochs=None):
    """
    Dual-axis plot showing training/validation loss and specialization divergence.

    THE GOLD PLOT for papers: Shows that specialization emerges during training
    and correlates with loss improvement.

    Args:
        all_metrics: Expert metrics dict
        output_dir: Output directory
        metrics_json_path: Path to training_metrics_stage3.json
        selected_epochs: Optional list of epochs to plot
    """
    if selected_epochs is None:
        epochs = sorted(all_metrics.keys())
    else:
        epochs = selected_epochs

    # Load training metrics
    if not os.path.exists(metrics_json_path):
        print(f"  ⚠️  Warning: Training metrics not found at {metrics_json_path}")
        print("     Skipping loss_and_specialization plot")
        return

    with open(metrics_json_path) as f:
        training_metrics = json.load(f)

    # Extract loss values for selected epochs
    train_loss = []
    val_loss = []
    divergence_values = []

    for epoch in epochs:
        # Find corresponding epoch in training metrics
        if epoch in training_metrics["epoch"]:
            idx = training_metrics["epoch"].index(epoch)
            train_loss.append(training_metrics["train_loss"][idx])
            val_loss.append(training_metrics["val_loss"][idx])
        else:
            train_loss.append(None)
            val_loss.append(None)

        # Compute specialization divergence
        metrics = all_metrics[epoch]
        agg = metrics["aggregate"]

        if "visual_routing" in agg and "text_routing" in agg:
            visual_e0 = agg["visual_routing"].get("expert_0", 50)
            text_e0 = agg["text_routing"].get("expert_0", 50)
            divergence = abs(visual_e0 - text_e0)
            divergence_values.append(divergence)
        else:
            divergence_values.append(None)

    # Filter out None values for plotting
    valid_epochs = [
        e
        for i, e in enumerate(epochs)
        if train_loss[i] is not None and divergence_values[i] is not None
    ]
    valid_train_loss = [l for l in train_loss if l is not None]
    valid_val_loss = [l for l in val_loss if l is not None]
    valid_divergence = [d for d in divergence_values if d is not None]

    if not valid_epochs:
        print("  ⚠️  Warning: No valid data for loss_and_specialization plot")
        return

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Left Y-axis: Loss
    color_train = "#3498db"
    color_val = "#e74c3c"
    ax1.set_xlabel("Epoch", fontsize=13)
    ax1.set_ylabel("Loss", fontsize=13, color="black")

    line1 = ax1.plot(
        valid_epochs,
        valid_train_loss,
        label="Training Loss",
        marker="o",
        linewidth=3,
        color=color_train,
        markersize=8,
    )
    line2 = ax1.plot(
        valid_epochs,
        valid_val_loss,
        label="Validation Loss",
        marker="s",
        linewidth=3,
        color=color_val,
        markersize=8,
    )

    ax1.tick_params(axis="y", labelcolor="black")
    ax1.set_xticks(valid_epochs)
    ax1.grid(True, alpha=0.3, axis="y")

    # Right Y-axis: Specialization Divergence
    ax2 = ax1.twinx()
    color_spec = "#2ecc71"
    ax2.set_ylabel("Specialization Divergence (%)", fontsize=13, color=color_spec)

    line3 = ax2.plot(
        valid_epochs,
        valid_divergence,
        label="Specialization Divergence",
        marker="D",
        linewidth=3,
        color=color_spec,
        markersize=8,
        linestyle="--",
    )

    ax2.tick_params(axis="y", labelcolor=color_spec)
    ax2.set_ylim(0, max(valid_divergence) * 1.2 if valid_divergence else 50)

    # Combined legend
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", fontsize=11, framealpha=0.95)

    ax1.set_title(
        "Training Progress: Loss vs Modality Specialization\n"
        + "Does specialization emerge during training?",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_and_specialization.png"))
    plt.close()
    print("  ✅ Saved: loss_and_specialization.png")
