#!/usr/bin/env python3
"""
Expert Metrics Visualization Script

Analyzes and visualizes MoE expert utilization patterns from Stage 3 training.
Generates per-layer and aggregate metrics plots showing:
1. Expert load distribution across layers
2. Routing entropy across layers
3. High confidence fraction across layers
4. Visual vs Text routing patterns across layers
5. Expert specialization evolution across epochs

Usage:
    python analysis_scripts/plot_expert_metrics.py --metrics_dir /path/to/expert_metrics --output_dir results/expert_metrics
"""

import argparse
import glob
import json
import os

from analysis_scripts import expert_metrics_plots as emp


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
        description="Visualize expert utilization metrics from Stage 3 training"
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
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("EXPERT METRICS VISUALIZATION")
    print("=" * 80)
    print(f"📂 Metrics directory: {args.metrics_dir}")
    print(f"📊 Output directory:  {args.output_dir}\n")

    # Find all expert metrics files
    metrics_files = glob.glob(os.path.join(args.metrics_dir, "expert_metrics_epoch_*.json"))
    if not metrics_files:
        print(f"❌ No expert metrics files found in {args.metrics_dir}")
        print("   Expected files matching pattern: expert_metrics_epoch_*.json")
        return
    print(f"📋 Found {len(metrics_files)} epoch(s) of metrics:")
    # Load all metrics
    all_metrics = {}
    for metrics_file in sorted(metrics_files):
        epoch = extract_epoch_number(os.path.basename(metrics_file))
        if epoch is not None:
            metrics = load_expert_metrics(metrics_file)
            all_metrics[epoch] = metrics
            print(f"   ✓ Epoch {epoch}: {os.path.basename(metrics_file)}")
    if not all_metrics:
        print("❌ Failed to load any metrics files")
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
            print(f"❌ No matching epochs found for --epochs {args.epochs}")
            return
    else:
        selected_epochs = available_epochs

    # Filter all_metrics to selected epochs
    all_metrics = {e: all_metrics[e] for e in selected_epochs}

    # Parse layers argument
    # If 'all_layers', use all available layers from the first epoch
    if args.layers.strip() == "all_layers":
        first_epoch = next(iter(all_metrics.values()))
        num_layers = len(first_epoch["per_layer"])
        selected_layers = list(range(num_layers))
    else:
        selected_layers = [int(x) for x in args.layers.strip().split()]

    print(f"\n{'=' * 80}")
    print("GENERATING PLOTS")
    print("=" * 80)
    print(f"📍 Selected layers: {selected_layers}")
    print(f"📍 Selected epochs: {selected_epochs}\n")

    # Generate all plots
    print("📈 Generating per-layer plots...")
    emp.plot_expert_load_distribution(all_metrics, args.output_dir, selected_layers)
    emp.plot_routing_entropy(all_metrics, args.output_dir, selected_layers)
    emp.plot_high_confidence_fraction(all_metrics, args.output_dir, selected_layers)
    emp.plot_visual_vs_text_routing(all_metrics, args.output_dir, selected_layers)

    print("\n📈 Generating specialization evolution plot...")
    emp.plot_specialization_evolution(all_metrics, args.output_dir)

    print("\n📈 Generating aggregate summary...")
    emp.plot_aggregate_summary(all_metrics, args.output_dir)

    print("\n� Generating KEY RESEARCH PLOTS...")
    print("   (Modality specialization, routing confidence, loss correlation)")

    # NEW: Top 3 recommended plots for research analysis
    emp.plot_modality_specialization_divergence(all_metrics, args.output_dir, selected_epochs)
    emp.plot_routing_confidence_evolution(all_metrics, args.output_dir, selected_epochs)

    # Try to find training metrics JSON for the combined loss plot
    # Look in parent directory of metrics_dir (typically OUTPUT_DIR)
    metrics_parent = os.path.dirname(args.metrics_dir)
    training_metrics_path = os.path.join(metrics_parent, "training_metrics_stage3.json")
    emp.plot_loss_and_specialization(
        all_metrics, args.output_dir, training_metrics_path, selected_epochs
    )

    print("\n�📝 Generating text report...")
    emp.generate_report(all_metrics, args.output_dir)

    print(f"\n{'=' * 80}")
    print("✅ COMPLETE!")
    print("=" * 80)
    print(f"\n📁 All plots saved to: {args.output_dir}/")
    print("\nGenerated files:")
    print("  Per-layer analysis:")
    print("    • expert_load_distribution.png")
    print("    • routing_entropy.png")
    print("    • high_confidence_fraction.png")
    print("    • visual_vs_text_routing.png")
    print("\n  Epoch-wise evolution:")
    print("    • specialization_evolution.png")
    print("    • aggregate_summary.png")
    print("\n  🌟 KEY RESEARCH PLOTS:")
    print("    • specialization_divergence.png        (Modality specialization over time)")
    print("    • routing_confidence_evolution.png     (Confidence & entropy trends)")
    print("    • loss_and_specialization.png          (Loss vs specialization dual-axis)")
    print("\n  Text report:")
    print("    • expert_metrics_report.txt")
    print()


if __name__ == "__main__":
    main()
