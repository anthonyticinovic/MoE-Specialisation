"""
Compare POPE results across different priming strategies.

Shows whether priming Stage 3 with "fake previous answers" improves performance
by exploiting its learned multi-turn conversation behaviour.
"""

import argparse
import json
import logging
from pathlib import Path

from analysis_scripts.pope_evaluation.pope_utils import confusion_counts
from models.utils.common import setup_logging

logger = logging.getLogger(__name__)


def compute_priming_metrics(results):
    """POPE metrics as percentages, plus specificity and the answer breakdown.

    A different presentation from ``pope_utils.compute_metrics`` — percentages
    rather than fractions, and an extra specificity column — which is why both
    exist. The *counting* is shared, so the two can no longer disagree about
    what a correct answer is: this copy used to compare the raw strings without
    lower-casing them, scoring a model that answered "Yes" as always wrong.
    """
    counts = confusion_counts(results)

    def pct(numerator, denominator):
        return (numerator / denominator * 100) if denominator > 0 else 0

    precision = pct(counts.true_positive, counts.predicted_yes)
    recall = pct(counts.true_positive, counts.true_positive + counts.false_negative)

    return {
        "accuracy": pct(counts.correct, counts.answerable),
        "precision": precision,
        "recall": recall,
        "specificity": pct(counts.true_negative, counts.true_negative + counts.false_positive),
        "f1": (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0,
        "yes_pct": pct(counts.predicted_yes, counts.answerable),
        "no_pct": pct(counts.predicted_no, counts.answerable),
        "unclear_pct": pct(counts.unclear, counts.total),
        "answerable": counts.answerable,
        "total": counts.total,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare POPE priming strategies")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/pope_evaluation",
        help="POPE evaluation results directory (contains answers_primed/)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    answers_dir = results_dir / "answers_primed"

    if not answers_dir.exists():
        logger.error(f"Primed answers directory not found: {answers_dir}")
        return

    strategies = ["simple", "conversational", "none"]
    difficulties = ["random", "popular", "adversarial"]

    logger.info("=" * 80)
    logger.info("POPE EVALUATION - PRIMING STRATEGY COMPARISON")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Testing whether priming Stage 3 with 'fake previous answers' improves")
    logger.info("performance by exploiting learned multi-turn conversation behavior.")
    logger.info("")
    logger.info("Strategies tested:")
    logger.info("  - simple: Prime with 'This image has been analyzed.'")
    logger.info("  - conversational: Prime with full Q&A pair")
    logger.info("  - none: No priming (baseline)")
    logger.info("")
    logger.info("=" * 80)
    logger.info("")

    # Collect results for each strategy
    all_results = {}

    for strategy in strategies:
        all_results[strategy] = {}

        for difficulty in difficulties:
            answer_file = answers_dir / f"stage3_{difficulty}_{strategy}.json"

            if not answer_file.exists():
                logger.warning(f" Missing: {answer_file.name}")
                continue

            with open(answer_file) as f:
                results = json.load(f)

            metrics = compute_priming_metrics(results)
            all_results[strategy][difficulty] = metrics

    # Print comparison table
    logger.info("")
    logger.info("RESULTS BY PRIMING STRATEGY")
    logger.info("=" * 80)
    logger.info("")

    for difficulty in difficulties:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"DIFFICULTY: {difficulty.upper()}")
        logger.info(f"{'=' * 80}")
        logger.info("")
        logger.info(
            f"{'Strategy':<20} {'Accuracy':<10} {'Yes%':<10} {'No%':<10} {'Unclear%':<10} "
            f"{'F1':<10}"
        )
        logger.info("-" * 80)

        for strategy in strategies:
            if difficulty not in all_results[strategy]:
                logger.info(f"{strategy:<20} {'N/A':<10}")
                continue

            m = all_results[strategy][difficulty]
            logger.info(
                f"{strategy:<20} {m['accuracy']:>9.1f}% {m['yes_pct']:>9.1f}% "
                f"{m['no_pct']:>9.1f}% {m['unclear_pct']:>9.1f}% {m['f1']:>9.1f}"
            )

        logger.info("")

    # Compare best vs worst
    logger.info("")
    logger.info("=" * 80)
    logger.info("KEY FINDINGS")
    logger.info("=" * 80)
    logger.info("")

    # Average accuracy across difficulties for each strategy
    avg_accuracies = {}
    avg_unclear = {}
    avg_no_pct = {}

    for strategy in strategies:
        accuracies = [m["accuracy"] for m in all_results[strategy].values() if m]
        unclear_pcts = [m["unclear_pct"] for m in all_results[strategy].values() if m]
        no_pcts = [m["no_pct"] for m in all_results[strategy].values() if m]

        if accuracies:
            avg_accuracies[strategy] = sum(accuracies) / len(accuracies)
            avg_unclear[strategy] = sum(unclear_pcts) / len(unclear_pcts)
            avg_no_pct[strategy] = sum(no_pcts) / len(no_pcts)

    logger.info("Average Performance Across All Difficulties:")
    logger.info("")
    logger.info(f"{'Strategy':<20} {'Avg Accuracy':<15} {'Avg Unclear%':<15} {'Avg No%':<10}")
    logger.info("-" * 60)

    for strategy in strategies:
        if strategy in avg_accuracies:
            logger.info(
                f"{strategy:<20} {avg_accuracies[strategy]:>14.1f}% "
                f"{avg_unclear[strategy]:>14.1f}% {avg_no_pct[strategy]:>9.1f}%"
            )

    logger.info("")

    # Find best strategy
    if avg_accuracies:
        best_strategy = max(avg_accuracies, key=avg_accuracies.get)
        best_accuracy = avg_accuracies[best_strategy]

        # Compare to baseline (none)
        if "none" in avg_accuracies:
            baseline_accuracy = avg_accuracies["none"]
            improvement = best_accuracy - baseline_accuracy

            logger.info(f"Best Strategy: {best_strategy}")
            logger.info(f"   Average Accuracy: {best_accuracy:.1f}%")
            logger.info(f"   Baseline (none): {baseline_accuracy:.1f}%")
            logger.info(f"   Improvement: {improvement:+.1f}%")
            logger.info("")

            if improvement > 5:
                logger.info("PRIMING WORKS! Significant improvement detected.")
                logger.info("   Stage 3's learned multi-turn behaviour can be exploited.")
            elif improvement > 1:
                logger.warning(" MINOR IMPROVEMENT: Priming helps a bit, but not dramatically.")
            else:
                logger.error("NO IMPROVEMENT: Priming doesn't help (or makes it worse).")
                logger.info("   Stage 3's issues may be too fundamental for priming to fix.")

    logger.info("")
    logger.info("=" * 80)


if __name__ == "__main__":
    setup_logging()
    main()
