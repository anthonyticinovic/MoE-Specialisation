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
    python analysis_scripts/plot_expert_metrics.py \
        --metrics_dir /path/to/expert_metrics \
        --output_dir results/expert_metrics
"""

import argparse
import glob
import json
import logging
import os

from analysis_scripts import expert_metrics_evolution_plots as eme
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


def _build_parser() -> argparse.ArgumentParser:
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
        help="Path to training_metrics_stage3.json (default: beside --metrics_dir)",
    )
    return parser


def _load_all_metrics(metrics_dir: str) -> dict[int, dict] | None:
    """Read every ``expert_metrics_epoch_*.json``, keyed by epoch number."""
    metrics_files = glob.glob(os.path.join(metrics_dir, "expert_metrics_epoch_*.json"))
    if not metrics_files:
        logger.error("No expert metrics files found in %s", metrics_dir)
        logger.error("   Expected files matching pattern: expert_metrics_epoch_*.json")
        return None

    logger.info("Found %d epoch(s) of metrics:", len(metrics_files))
    all_metrics = {}
    for metrics_file in sorted(metrics_files):
        epoch = extract_epoch_number(os.path.basename(metrics_file))
        if epoch is not None:
            all_metrics[epoch] = load_expert_metrics(metrics_file)
            logger.info("   Epoch %d: %s", epoch, os.path.basename(metrics_file))

    if not all_metrics:
        logger.error("Found %d file(s) but none carried a parseable epoch", len(metrics_files))
        return None
    return all_metrics


def _select_epochs(spec: str | None, available: list[int]) -> list[int] | None:
    """Parse an ``--epochs`` spec like ``1,2,5`` or ``1-5,7``. None means all."""
    if not spec:
        return available

    wanted: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            wanted.update(range(int(start), int(end) + 1))
        else:
            wanted.add(int(part))

    selected = sorted(epoch for epoch in wanted if epoch in available)
    if not selected:
        logger.error("No matching epochs found for --epochs %s (have %s)", spec, available)
        return None
    return selected


def _select_layers(spec: str, num_layers: int) -> list[int] | None:
    """Parse a ``--layers`` spec, rejecting indices the metrics do not cover."""
    if spec.strip() == "all_layers":
        return list(range(num_layers))

    selected = [int(x) for x in spec.strip().split()]
    # The default sampling assumes the paper's 32-layer model. Say so plainly
    # rather than failing with an IndexError deep inside a plot.
    out_of_range = [layer for layer in selected if layer >= num_layers]
    if out_of_range:
        logger.error(
            "Requested layer(s) %s but these metrics cover %d layer(s) (0-%d). "
            "Pass --layers 'all_layers' or a subset in range.",
            out_of_range,
            num_layers,
            num_layers - 1,
        )
        return None
    return selected


def _render(all_metrics: dict, args, layers: list[int], epochs: list[int]) -> None:
    """Draw every figure and write the text report."""
    logger.info("Generating per-layer plots...")
    emp.plot_expert_load_distribution(all_metrics, args.output_dir, layers)
    emp.plot_routing_entropy(all_metrics, args.output_dir, layers)
    emp.plot_high_confidence_fraction(all_metrics, args.output_dir, layers)
    emp.plot_visual_vs_text_routing(all_metrics, args.output_dir, layers)

    logger.info("\nGenerating specialisation evolution plot...")
    eme.plot_specialization_evolution(all_metrics, args.output_dir)

    logger.info("\nGenerating aggregate summary...")
    eme.plot_aggregate_summary(all_metrics, args.output_dir)

    logger.info("\nGenerating the headline plots...")
    logger.info("   (Modality specialisation, routing confidence, loss correlation)")
    eme.plot_modality_specialization_divergence(all_metrics, args.output_dir, epochs)
    eme.plot_routing_confidence_evolution(all_metrics, args.output_dir, epochs)

    # A training run writes the loss history beside the expert_metrics/ directory.
    training_metrics_path = args.training_metrics or os.path.join(
        os.path.dirname(args.metrics_dir), "training_metrics_stage3.json"
    )
    if not os.path.exists(training_metrics_path):
        logger.warning(
            "No training metrics at %s — skipping the loss plot. Pass --training_metrics.",
            training_metrics_path,
        )
    eme.plot_loss_and_specialization(all_metrics, args.output_dir, training_metrics_path, epochs)

    logger.info("\nGenerating text report...")
    eme.generate_report(all_metrics, args.output_dir)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, plot, and return an exit code.

    Takes ``argv`` so a test can call it instead of launching a subprocess, and
    returns a code rather than ``None`` so a refusal to plot is visible to
    whatever invoked it — every guard below used to ``return`` bare and let the
    script exit 0 having produced nothing.
    """
    args = _build_parser().parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 80)
    logger.info("EXPERT METRICS VISUALISATION")
    logger.info("=" * 80)
    logger.info("Metrics directory: %s", args.metrics_dir)
    logger.info("Output directory:  %s\n", args.output_dir)

    all_metrics = _load_all_metrics(args.metrics_dir)
    if all_metrics is None:
        return 1

    epochs = _select_epochs(args.epochs, sorted(all_metrics))
    if epochs is None:
        return 1
    all_metrics = {epoch: all_metrics[epoch] for epoch in epochs}

    layers = _select_layers(args.layers, len(next(iter(all_metrics.values()))["per_layer"]))
    if layers is None:
        return 1

    logger.info("\n%s", "=" * 80)
    logger.info("GENERATING PLOTS")
    logger.info("=" * 80)
    logger.info("Selected layers: %s", layers)
    logger.info("Selected epochs: %s\n", epochs)

    _render(all_metrics, args, layers, epochs)

    logger.info("\n%s", "=" * 80)
    logger.info("COMPLETE! All plots saved to: %s", args.output_dir)
    logger.info("=" * 80)
    return 0


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
