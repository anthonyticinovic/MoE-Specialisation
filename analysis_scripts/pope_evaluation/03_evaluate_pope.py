#!/usr/bin/env python3
"""
Step 3: Evaluate POPE answers and compute metrics.
Computes accuracy, precision, recall, F1 for object hallucination detection.
"""

import argparse
import json
import sys
from pathlib import Path

# pope_utils lives alongside this script
sys.path.insert(0, str(Path(__file__).parent))

from pope_utils import compute_metrics  # noqa: E402


def print_metrics(stage_name: str, difficulty: str, metrics: dict):
    """Print metrics in a formatted table."""
    print(f"\n{'=' * 80}")
    print(f"{stage_name.upper()} - {difficulty.upper()} DIFFICULTY")
    print(f"{'=' * 80}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"  Precision:   {metrics['precision']:.4f} ({metrics['precision'] * 100:.2f}%)")
    print(f"  Recall:      {metrics['recall']:.4f} ({metrics['recall'] * 100:.2f}%)")
    print(f"  F1 Score:    {metrics['f1']:.4f} ({metrics['f1'] * 100:.2f}%)")
    print(f"  Yes Ratio:   {metrics['yes_ratio']:.4f} ({metrics['yes_ratio'] * 100:.2f}%)")
    print(f"{'=' * 80}")
    print("  Confusion Matrix:")
    print(f"    True Positive:   {metrics['true_positive']:4d} (Correctly said yes)")
    print(f"    False Positive:  {metrics['false_positive']:4d} (Hallucination - said yes but no)")
    print(f"    True Negative:   {metrics['true_negative']:4d} (Correctly said no)")
    print(f"    False Negative:  {metrics['false_negative']:4d} (Missed - said no but yes)")
    print(f"    Unclear:         {metrics['num_unclear']:4d}")
    print(f"{'=' * 80}")


def create_comparison_table(
    stage2_metrics: dict, stage3_metrics: dict, difficulties: list[str]
) -> str:
    """Create a comparison table for Stage 2 vs Stage 3."""
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("POPE EVALUATION COMPARISON: STAGE 2 vs STAGE 3")
    lines.append("=" * 100)

    for difficulty in difficulties:
        if difficulty not in stage2_metrics or difficulty not in stage3_metrics:
            continue

        s2 = stage2_metrics[difficulty]
        s3 = stage3_metrics[difficulty]

        lines.append(f"\n{difficulty.upper()} Difficulty:")
        lines.append("-" * 100)
        lines.append(
            f"{'Metric':<20} {'Stage 2':>15} {'Stage 3':>15} {'Δ (S3-S2)':>15} {'Winner':>15}"
        )
        lines.append("-" * 100)

        metrics_to_compare = [
            ("Accuracy", "accuracy"),
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("F1 Score", "f1"),
            ("Yes Ratio", "yes_ratio"),
        ]

        for metric_name, metric_key in metrics_to_compare:
            s2_val = s2[metric_key]
            s3_val = s3[metric_key]
            delta = s3_val - s2_val

            # For Yes Ratio, lower is better (less hallucination)
            # For others, higher is better
            if metric_key == "yes_ratio":
                winner = "Stage 2" if s2_val < s3_val else "Stage 3" if s3_val < s2_val else "Tie"
            else:
                winner = "Stage 3" if s3_val > s2_val else "Stage 2" if s2_val > s3_val else "Tie"

            lines.append(
                f"{metric_name:<20} {s2_val:>14.4f} {s3_val:>14.4f} {delta:>+14.4f} {winner:>15}"
            )

        lines.append("-" * 100)
        lines.append(
            f"Hallucinations (FP): {s2['false_positive']:>7d}        {s3['false_positive']:>7d}        {s3['false_positive'] - s2['false_positive']:>+7d}"
        )

    lines.append("=" * 100)

    # Summary
    lines.append("\nKey Observations:")
    lines.append("-" * 100)

    for difficulty in difficulties:
        if difficulty not in stage2_metrics or difficulty not in stage3_metrics:
            continue

        s2 = stage2_metrics[difficulty]
        s3 = stage3_metrics[difficulty]

        acc_winner = "Stage 3" if s3["accuracy"] > s2["accuracy"] else "Stage 2"
        acc_margin = abs(s3["accuracy"] - s2["accuracy"]) * 100

        f1_winner = "Stage 3" if s3["f1"] > s2["f1"] else "Stage 2"
        f1_margin = abs(s3["f1"] - s2["f1"]) * 100

        halluc_change = s3["false_positive"] - s2["false_positive"]
        halluc_direction = "more" if halluc_change > 0 else "fewer"

        lines.append(
            f"• {difficulty.capitalize()} - Accuracy: {acc_winner} wins by {acc_margin:.2f}%"
        )
        lines.append(
            f"• {difficulty.capitalize()} - F1 Score: {f1_winner} wins by {f1_margin:.2f}%"
        )
        lines.append(
            f"• {difficulty.capitalize()} - Hallucinations: Stage 3 has {abs(halluc_change)} {halluc_direction} ({halluc_change:+d})"
        )

    lines.append("=" * 100)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate POPE answers")
    parser.add_argument(
        "--stage2_dir",
        type=str,
        default="results/pope_evaluation",
        help="Directory containing Stage 2 answers",
    )
    parser.add_argument(
        "--stage3_dir",
        type=str,
        default="results/pope_evaluation",
        help="Directory containing Stage 3 answers",
    )
    parser.add_argument(
        "--difficulties",
        nargs="+",
        default=["random", "popular", "adversarial"],
        help="Difficulty levels to evaluate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/pope_evaluation",
        help="Output directory for results",
    )

    args = parser.parse_args()

    print("=" * 100)
    print("POPE EVALUATION".center(100))
    print("=" * 100)

    stage2_metrics = {}
    stage3_metrics = {}

    # Evaluate each difficulty level for both stages
    for difficulty in args.difficulties:
        # Stage 2
        stage2_file = Path(args.stage2_dir) / f"stage2_{difficulty}_answers.json"
        if stage2_file.exists():
            print(f"\n📊 Evaluating Stage 2 - {difficulty.capitalize()}...")
            with open(stage2_file) as f:
                stage2_answers = json.load(f)

            stage2_metrics[difficulty] = compute_metrics(stage2_answers)
            print_metrics("stage2", difficulty, stage2_metrics[difficulty])
        else:
            print(f"\n⚠️  Stage 2 {difficulty} answers not found: {stage2_file}")

        # Stage 3
        stage3_file = Path(args.stage3_dir) / f"stage3_{difficulty}_answers.json"
        if stage3_file.exists():
            print(f"\n📊 Evaluating Stage 3 - {difficulty.capitalize()}...")
            with open(stage3_file) as f:
                stage3_answers = json.load(f)

            stage3_metrics[difficulty] = compute_metrics(stage3_answers)
            print_metrics("stage3", difficulty, stage3_metrics[difficulty])
        else:
            print(f"\n⚠️  Stage 3 {difficulty} answers not found: {stage3_file}")

    # Create comparison if we have both stages
    if stage2_metrics and stage3_metrics:
        comparison = create_comparison_table(stage2_metrics, stage3_metrics, args.difficulties)
        print(comparison)

        # Save comparison
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        comparison_file = output_dir / "pope_comparison.txt"
        with open(comparison_file, "w") as f:
            f.write(comparison)
        print(f"\n💾 Saved comparison: {comparison_file}")

    # Save metrics JSON
    all_metrics = {"stage2": stage2_metrics, "stage3": stage3_metrics}

    metrics_file = Path(args.output_dir) / "pope_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"💾 Saved metrics: {metrics_file}")

    print("\n" + "=" * 100)
    print("✅ POPE EVALUATION COMPLETE".center(100))
    print("=" * 100)


if __name__ == "__main__":
    main()
