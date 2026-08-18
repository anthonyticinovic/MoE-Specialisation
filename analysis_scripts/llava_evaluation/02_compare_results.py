"""
Compare LLaVA-Wild evaluation results between Stage 2 and Stage 3.

This analysis tests the hypothesis:
- Stage 3 performs BETTER on in-distribution tasks (LLaVA-style conversational VQA)
- Stage 3 performs WORSE on out-of-distribution tasks (POPE, COCO captioning)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Running this file directly puts *its own* directory on sys.path, not the repo
# root, so the first-party imports below would fail. Add the root explicitly.
# This replaces sys.path edits that were spread across several modules and
# depended on import order to take effect before they were needed.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.utils.common import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def load_results(json_path):
    """Load evaluation results."""
    with open(json_path) as f:
        data = json.load(f)
    return data


def paired_scores(stage2_data, stage3_data):
    """Return the two score lists, refusing to pair runs of different lengths.

    Everything downstream pairs these by position — the win counts, the tie
    count, the side-by-side samples and the per-sample scatter. If the two runs
    evaluated different numbers of samples that pairing compares unrelated
    samples, and the failure surfaces much later as an opaque matplotlib error
    about mismatched x/y sizes. Fail here, with the reason.
    """
    s2_scores = [r["score"] for r in stage2_data["results"]]
    s3_scores = [r["score"] for r in stage3_data["results"]]

    if len(s2_scores) != len(s3_scores):
        raise ValueError(
            f"Stage 2 evaluated {len(s2_scores)} samples and Stage 3 evaluated "
            f"{len(s3_scores)}. This comparison pairs them by position, so the "
            "results would compare different samples. Re-run the missing side, "
            "or filter both to their common sample ids first."
        )
    return s2_scores, s3_scores


def analyze_results(stage2_data, stage3_data):
    """
    Compare Stage 2 vs Stage 3 performance.
    """
    logger.info("=" * 80)
    logger.info("LLAVA-WILD EVALUATION COMPARISON")
    logger.info("=" * 80)

    # Summary metrics
    s2_summary = stage2_data["summary"]
    s3_summary = stage3_data["summary"]

    logger.info("\nOVERALL SCORES:")
    logger.info(f"{'Metric':<30} {'Stage 2':<15} {'Stage 3':<15} {'Difference':<15}")
    logger.info("-" * 80)

    s2_score = s2_summary["average_score"]
    s3_score = s3_summary["average_score"]
    diff = s3_score - s2_score

    logger.info(f"{'Average Score (0-100)':<30} {s2_score:<15.1f} {s3_score:<15.1f} {diff:+.1f}")
    logger.info(
        f"{'Samples Evaluated':<30} {s2_summary['num_samples']:<15} {s3_summary['num_samples']:<15}"
    )

    # Distribution analysis
    s2_scores, s3_scores = paired_scores(stage2_data, stage3_data)

    if len(s2_scores) == 0 or len(s3_scores) == 0:
        logger.error("\nERROR: No samples were evaluated!")
        logger.info(f"   Stage 2: {len(s2_scores)} samples")
        logger.info(f"   Stage 3: {len(s3_scores)} samples")
        logger.info("\n   This likely means image files were not found.")
        logger.info("   Check the image directory path and image filenames.")
        return None

    logger.info("\nSCORE DISTRIBUTION:")
    logger.info(f"{'Statistic':<30} {'Stage 2':<15} {'Stage 3':<15}")
    logger.info("-" * 80)
    logger.info(f"{'Mean':<30} {np.mean(s2_scores):<15.1f} {np.mean(s3_scores):<15.1f}")
    logger.info(f"{'Median':<30} {np.median(s2_scores):<15.1f} {np.median(s3_scores):<15.1f}")
    logger.info(f"{'Std Dev':<30} {np.std(s2_scores):<15.1f} {np.std(s3_scores):<15.1f}")
    logger.info(f"{'Min':<30} {min(s2_scores):<15.1f} {min(s3_scores):<15.1f}")
    logger.info(f"{'Max':<30} {max(s2_scores):<15.1f} {max(s3_scores):<15.1f}")

    # Quality analysis (score ranges)
    def score_category_count(scores):
        excellent = sum(1 for s in scores if s >= 80)
        good = sum(1 for s in scores if 60 <= s < 80)
        fair = sum(1 for s in scores if 40 <= s < 60)
        poor = sum(1 for s in scores if s < 40)
        return excellent, good, fair, poor

    s2_exc, s2_good, s2_fair, s2_poor = score_category_count(s2_scores)
    s3_exc, s3_good, s3_fair, s3_poor = score_category_count(s3_scores)

    logger.info("\nQUALITY DISTRIBUTION:")
    logger.info(f"{'Category':<30} {'Stage 2':<15} {'Stage 3':<15}")
    logger.info("-" * 80)
    logger.info(f"{'Excellent (80-100)':<30} {s2_exc:<15} {s3_exc:<15}")
    logger.info(f"{'Good (60-79)':<30} {s2_good:<15} {s3_good:<15}")
    logger.info(f"{'Fair (40-59)':<30} {s2_fair:<15} {s3_fair:<15}")
    logger.info(f"{'Poor (<40)':<30} {s2_poor:<15} {s3_poor:<15}")

    # Sample comparisons
    logger.info("\nSAMPLE COMPARISONS (First 5):")
    logger.info("=" * 80)

    for i in range(min(5, len(stage2_data["results"]))):
        s2_result = stage2_data["results"][i]
        s3_result = stage3_data["results"][i]

        logger.info(f"\nSample {i + 1}: {s2_result['image']}")
        logger.info(f"Question: {s2_result['question'][:80]}...")
        logger.info(f"Reference: {s2_result['reference_answer'][:80]}...")
        logger.info(
            f"\nStage 2 ({s2_result['score']:.0f}): {s2_result['generated_answer'][:100]}..."
        )
        logger.info(f"Stage 3 ({s3_result['score']:.0f}): {s3_result['generated_answer'][:100]}...")
        logger.info("-" * 80)

    # Context: POPE and COCO results
    logger.info("\nCONTEXT - ALL BENCHMARKS:")
    logger.info("=" * 80)
    logger.info("Benchmark               Metric                Stage 2      Stage 3      Change")
    logger.info("-" * 80)
    logger.error(f"{'POPE (OOD)':<23} {'Accuracy':<25} {71.5:<12.1f} {30.0:<12.1f} {-41.5:+.1f} ")
    logger.error(
        f"{'COCO Captions (OOD)':<23} {'CIDEr':<25} {0.76:<12.2f} {0.08:<12.2f} {-0.68:+.2f} "
    )
    logger.info(
        f"{'LLaVA-Wild (ID)':<23} {'Quality Score':<25} {s2_score:<12.1f} {s3_score:<12.1f} {diff:+.1f} {'' if diff > 0 else ''}"
    )

    logger.info("\nINTERPRETATION:")
    if s3_score > s2_score:
        logger.info("   Stage 3 OUTPERFORMS Stage 2 on in-distribution task (LLaVA-Wild)")
        logger.info("   This confirms Stage 3 learned the training distribution well")
        logger.warning("    But Stage 3 FAILS catastrophically on out-of-distribution tasks")
        logger.info("   Thesis conclusion: Soft routing + instruction tuning = over-specialisation")
    else:
        logger.error("   Stage 3 does NOT outperform Stage 2 even on in-distribution task")
        logger.warning("    This suggests Stage 3's issues are more fundamental than expected")
        logger.info("   Need to investigate: Is LLaVA-Wild actually in-distribution?")

    logger.info("=" * 80)

    return {
        "stage2_mean": np.mean(s2_scores),
        "stage3_mean": np.mean(s3_scores),
        "stage2_scores": s2_scores,
        "stage3_scores": s3_scores,
    }


def create_plots(stage2_data, stage3_data, output_dir):
    """Create comparison visualisations."""
    s2_scores, s3_scores = paired_scores(stage2_data, stage3_data)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Score distributions (histogram)
    ax = axes[0, 0]
    bins = np.arange(0, 101, 10)
    ax.hist(s2_scores, bins=bins, alpha=0.6, label="Stage 2", color="blue")
    ax.hist(s3_scores, bins=bins, alpha=0.6, label="Stage 3", color="orange")
    ax.axvline(
        np.mean(s2_scores),
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Stage 2 Mean: {np.mean(s2_scores):.1f}",
    )
    ax.axvline(
        np.mean(s3_scores),
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Stage 3 Mean: {np.mean(s3_scores):.1f}",
    )
    ax.set_xlabel("Score (0-100)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("LLaVA-Wild Score Distribution", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Box plot comparison
    ax = axes[0, 1]
    ax.boxplot([s2_scores, s3_scores], labels=["Stage 2", "Stage 3"])
    ax.set_ylabel("Score (0-100)", fontsize=12)
    ax.set_title("Score Comparison (Box Plot)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # 3. All benchmarks comparison
    ax = axes[1, 0]
    benchmarks = ["POPE\n(OOD)", "COCO\n(OOD)", "LLaVA-Wild\n(In-Dist)"]
    # Normalize to 0-100 scale for comparison
    s2_values = [71.5, 76.0, np.mean(s2_scores)]  # POPE accuracy, COCO CIDEr*100, LLaVA score
    s3_values = [30.0, 8.0, np.mean(s3_scores)]  # POPE accuracy, COCO CIDEr*100, LLaVA score

    x = np.arange(len(benchmarks))
    width = 0.35

    ax.bar(x - width / 2, s2_values, width, label="Stage 2", color="blue", alpha=0.7)
    ax.bar(x + width / 2, s3_values, width, label="Stage 3", color="orange", alpha=0.7)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Performance Across Benchmarks", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add annotations
    for i, (s2, s3) in enumerate(zip(s2_values, s3_values, strict=True)):
        ax.text(i - width / 2, s2 + 2, f"{s2:.0f}", ha="center", va="bottom", fontsize=10)
        ax.text(i + width / 2, s3 + 2, f"{s3:.0f}", ha="center", va="bottom", fontsize=10)

    # 4. Per-sample comparison scatter
    ax = axes[1, 1]
    ax.scatter(s2_scores, s3_scores, alpha=0.5, s=30)
    max_score = max(max(s2_scores), max(s3_scores))
    ax.plot([0, max_score], [0, max_score], "r--", linewidth=2, label="Equal performance")
    ax.set_xlabel("Stage 2 Score", fontsize=12)
    ax.set_ylabel("Stage 3 Score", fontsize=12)
    ax.set_title("Per-Sample Score Comparison", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Count winners
    s2_wins = sum(1 for s2, s3 in zip(s2_scores, s3_scores, strict=True) if s2 > s3)
    s3_wins = sum(1 for s2, s3 in zip(s2_scores, s3_scores, strict=True) if s3 > s2)
    ties = len(s2_scores) - s2_wins - s3_wins
    ax.text(
        0.05,
        0.95,
        f"Stage 2 wins: {s2_wins}\nStage 3 wins: {s3_wins}\nTies: {ties}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    # Save
    output_path = Path(output_dir) / "llava_wild_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"\nSaved visualisation: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare LLaVA-Wild results")
    parser.add_argument("--stage2", type=str, required=True, help="Stage 2 results JSON")
    parser.add_argument("--stage3", type=str, required=True, help="Stage 3 results JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")

    args = parser.parse_args()

    # Load results
    logger.info("\nLoading results...")
    stage2_data = load_results(args.stage2)
    stage3_data = load_results(args.stage3)

    # Analyse
    stats = analyze_results(stage2_data, stage3_data)

    if stats is None:
        logger.error("\nAnalysis failed due to missing data. Exiting.")
        return

    # Create plots
    logger.info("\nCreating visualizations...")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    create_plots(stage2_data, stage3_data, args.output_dir)

    logger.info("\nAnalysis complete!\n")


if __name__ == "__main__":
    setup_logging()
    main()
