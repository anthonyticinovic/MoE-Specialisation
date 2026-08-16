#!/usr/bin/env python3
"""
Expert Metrics Visualisation Script

Analyses and visualises MoE expert utilisation patterns from Stage 3 training.
Generates per-layer and aggregate metrics plots showing:
1. Expert load distribution across layers
2. Routing entropy across layers
3. High confidence fraction across layers
4. Visual vs Text routing patterns across layers
5. Expert specialisation evolution across epochs

Usage:
    python analysis_scripts/plot_expert_metrics.py --metrics_dir /path/to/expert_metrics --output_dir results/expert_metrics
"""

import argparse
import logging
import glob
import json
import os

from analysis_scripts import expert_metrics_plots as emp
from models.utils.common import setup_logging

logger = logging.getLogger(__name__)


def load_expert_metrics(metrics_path):
    """Load expert metrics JSON file."""
    with open(metrics_path) as f:
        return json.load(f)


def extract_epoch_number(filename):
    """Extract epoch number from filename like 'expert_metrics_epoch_3.json'."""
    import re

    match = re.search(r"epoch_(\d+)", filename)
    if match:
        return int(match.group(1))
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Plot expert utilisation metrics from Stage 3 training"
    )
    parser.add_argument(
        "--metrics_dir",
        type=str,
        required=True,
        help="Directory containing expert metrics JSON files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/expert_metrics",
        help="Directory to save output plots",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="0 7 15 23 31",
        help="Layer indices to plot (e.g. '0 7 15 23 31' or 'all_layers')",
    )
    parser.add_argument(
        "--epochs",
        type=str,
        default=None,
        help="Epochs to plot (e.g. '1,2,5' or '1-5,7'). Default: all epochs.",
    )
    parser.add_argument(
        "--training_metrics",
        type=str,
        default=None,
        help=(
            "training_metrics_stage3.json for the loss plot. "
            "Defaults to the parent of --metrics_dir, which is where a training run puts it."
        ),
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 80)
    logger.info("EXPERT METRICS VISUALISATION")
    logger.info("=" * 80)
    logger.info(f"Metrics directory: {args.metrics_dir}")
    logger.info(f"Output directory:  {args.output_dir}\n")

    # Find all expert metrics files
    metrics_files = glob.glob(os.path.join(args.metrics_dir, "expert_metrics_epoch_*.json"))
    if not metrics_files:
        logger.error(f"No expert metrics files found in {args.metrics_dir}")
        logger.info("   Expected files matching pattern: expert_metrics_epoch_*.json")
        return
    logger.info(f"Found {len(metrics_files)} epoch(s) of metrics:")
    # Load all metrics
    all_metrics = {}
    for metrics_file in sorted(metrics_files):
        epoch = extract_epoch_number(os.path.basename(metrics_file))
        if epoch is not None:
            metrics = load_expert_metrics(metrics_file)
            all_metrics[epoch] = metrics
            logger.info(f"   Epoch {epoch}: {os.path.basename(metrics_file)}")
    if not all_metrics:
        logger.error("Failed to load any metrics files")
        return

    # Parse epochs argument
    available_epochs = sorted(all_metrics.keys())
    if args.epochs:
        selected_epochs = set()
        for part in args.epochs.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-")
                selected_epochs.update(range(int(start), int(end) + 1))
            else:
                selected_epochs.add(int(part))
        selected_epochs = sorted(e for e in selected_epochs if e in available_epochs)
        if not selected_epochs:
            logger.error(f"No matching epochs found for --epochs {args.epochs}")
            return
    else:
        selected_epochs = available_epochs

    # Filter all_metrics to selected epochs
    all_metrics = {e: all_metrics[e] for e in selected_epochs}

    # Parse layers argument
    # If 'all_layers', use all available layers from the first epoch
    num_layers = len(next(iter(all_metrics.values()))["per_layer"])
    if args.layers.strip() == "all_layers":
        selected_layers = list(range(num_layers))
    else:
        selected_layers = [int(x) for x in args.layers.strip().split()]
        # The default sampling assumes the paper's 32-layer model. Say so
        # plainly rather than failing with an IndexError deep inside a plot.
        out_of_range = [layer for layer in selected_layers if layer >= num_layers]
        if out_of_range:
            logger.error(
                "Requested layer(s) %s but these metrics cover %d layer(s) (0-%d). "
                "Pass --layers 'all_layers' or a subset in range.",
                out_of_range,
                num_layers,
                num_layers - 1,
            )
            return

    logger.info(f"\n{'=' * 80}")
    logger.info("GENERATING PLOTS")
    logger.info("=" * 80)
    logger.info(f"Selected layers: {selected_layers}")
    logger.info(f"Selected epochs: {selected_epochs}\n")

    # Generate all plots
    logger.info("Generating per-layer plots...")
    emp.plot_expert_load_distribution(all_metrics, args.output_dir, selected_layers)
    emp.plot_routing_entropy(all_metrics, args.output_dir, selected_layers)
    emp.plot_high_confidence_fraction(all_metrics, args.output_dir, selected_layers)
    emp.plot_visual_vs_text_routing(all_metrics, args.output_dir, selected_layers)

    logger.info("\nGenerating specialisation evolution plot...")
    emp.plot_specialization_evolution(all_metrics, args.output_dir)

    logger.info("\nGenerating aggregate summary...")
    emp.plot_aggregate_summary(all_metrics, args.output_dir)

    logger.info("\nGenerating the headline plots...")
    logger.info("   (Modality specialisation, routing confidence, loss correlation)")

    emp.plot_modality_specialization_divergence(all_metrics, args.output_dir, selected_epochs)
    emp.plot_routing_confidence_evolution(all_metrics, args.output_dir, selected_epochs)

    # A training run writes the loss history beside the expert_metrics/ directory.
    training_metrics_path = args.training_metrics or os.path.join(
        os.path.dirname(args.metrics_dir), "training_metrics_stage3.json"
    )
    if not os.path.exists(training_metrics_path):
        logger.warning(
            "No training metrics at %s — skipping the loss plot. Pass --training_metrics.",
            training_metrics_path,
        )
    emp.plot_loss_and_specialization(
        all_metrics, args.output_dir, training_metrics_path, selected_epochs
    )

    logger.info("\nGenerating text report...")
    emp.generate_report(all_metrics, args.output_dir)

    logger.info(f"\n{'=' * 80}")
    logger.info("COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nAll plots saved to: {args.output_dir}/")
    logger.info("\nGenerated files:")
    logger.info("  Per-layer analysis:")
    logger.info("    • expert_load_distribution.png")
    logger.info("    • routing_entropy.png")
    logger.info("    • high_confidence_fraction.png")
    logger.info("    • visual_vs_text_routing.png")
    logger.info("\n  Epoch-wise evolution:")
    logger.info("    • specialization_evolution.png")
    logger.info("    • aggregate_summary.png")
    logger.info("\n  Headline plots:")
    logger.info("    • specialization_divergence.png        (Modality specialisation over time)")
    logger.info("    • routing_confidence_evolution.png     (Confidence & entropy trends)")
    logger.info("    • loss_and_specialization.png          (Loss vs specialisation dual-axis)")
    logger.info("\n  Text report:")
    logger.info("    • expert_metrics_report.txt")
    logger.info("")


if __name__ == "__main__":
    setup_logging()
    main()
