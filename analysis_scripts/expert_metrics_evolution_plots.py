#!/usr/bin/env python3
"""Across-epoch plots and the text report for ``plot_expert_metrics``.

The companion to ``expert_metrics_plots``, which draws the per-layer views of a
single epoch. Everything here reads the whole run: how specialisation, routing
confidence and loss moved from the first epoch to the last. Split out because
the two concerns together ran past the 800-line guideline.

Pure functions: take loaded metrics plus an output directory, write files.
"""

import json
import logging
import os

import matplotlib.pyplot as plt

from analysis_scripts._lib import set_publication_rcparams

logger = logging.getLogger(__name__)

set_publication_rcparams()


def plot_specialization_evolution(all_metrics, output_dir, selected_epochs=None):
    """
    Plot how expert specialisation evolves across epochs.
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
    logger.info("  Saved: specialization_evolution.png")


def plot_aggregate_summary(all_metrics, output_dir):
    """
    Plot aggregate summary statistics for the latest epoch.
    Shows overall expert utilisation patterns (simplified version).
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
    for i, (_expert, load) in enumerate(zip(experts, loads, strict=True)):
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
        for i, (_expert, load) in enumerate(zip(visual_experts, visual_loads, strict=True)):
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
        for i, (_expert, load) in enumerate(zip(text_experts, text_loads, strict=True)):
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
    # These axes come from add_gridspec, which tight_layout cannot reflow — it
    # warned on every call that the result "might be incorrect". The gridspec
    # already carries the spacing, so leave the room for the suptitle directly.
    fig.subplots_adjust(top=0.86)
    plt.savefig(os.path.join(output_dir, "aggregate_summary.png"))
    plt.close()
    logger.info("  Saved: aggregate_summary.png")


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

            # Compute specialisation score
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

    logger.info("  Saved: expert_metrics_report.txt")


def plot_modality_specialization_divergence(all_metrics, output_dir, selected_epochs=None):
    """
    Plot modality specialisation divergence over epochs.
    Shows |Visual_Expert0% - Text_Expert0%| to quantify how differently
    experts handle visual vs text tokens.

    This is THE KEY METRIC for understanding modality-specific specialisation.
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
    logger.info("  Saved: specialization_divergence.png")


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
    logger.info("  Saved: routing_confidence_evolution.png")


def _loss_and_divergence(all_metrics, training_metrics, epochs):
    """Line the loss history up with the per-epoch specialisation divergence.

    Returns the epochs that have both, plus the three parallel series. An epoch
    missing from either side drops out of all four together — they are plotted
    against each other, so a gap in one has to be a gap in all.
    """
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

        # Compute specialisation divergence
        agg = all_metrics[epoch]["aggregate"]
        if "visual_routing" in agg and "text_routing" in agg:
            visual_e0 = agg["visual_routing"].get("expert_0", 50)
            text_e0 = agg["text_routing"].get("expert_0", 50)
            divergence_values.append(abs(visual_e0 - text_e0))
        else:
            divergence_values.append(None)

    valid_epochs = [
        e
        for i, e in enumerate(epochs)
        if train_loss[i] is not None and divergence_values[i] is not None
    ]
    valid_train_loss = [value for value in train_loss if value is not None]
    valid_val_loss = [value for value in val_loss if value is not None]
    valid_divergence = [d for d in divergence_values if d is not None]
    return valid_epochs, valid_train_loss, valid_val_loss, valid_divergence


def plot_loss_and_specialization(all_metrics, output_dir, metrics_json_path, selected_epochs=None):
    """
    Dual-axis plot showing training/validation loss and specialisation divergence.

    THE GOLD PLOT for papers: Shows that specialisation emerges during training
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
        logger.warning(f"   Warning: Training metrics not found at {metrics_json_path}")
        logger.info("     Skipping loss_and_specialization plot")
        return

    with open(metrics_json_path) as f:
        training_metrics = json.load(f)

    # Extract loss values for selected epochs
    valid_epochs, valid_train_loss, valid_val_loss, valid_divergence = _loss_and_divergence(
        all_metrics, training_metrics, epochs
    )
    if not valid_epochs:
        logger.warning("   Warning: No valid data for loss_and_specialization plot")
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

    # Right Y-axis: Specialisation Divergence
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
    labels = [line.get_label() for line in lines]
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
    logger.info("  Saved: loss_and_specialization.png")
