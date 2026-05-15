#!/usr/bin/env python3
"""
Step 6: Visualize evaluation results and generate comprehensive report.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from karpathy_utils import load_json, print_banner


def plot_retrieval_metrics(stage2_metrics: dict, stage3_metrics: dict, output_path: str):
    """Create bar chart comparing retrieval metrics."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # I2T metrics
    ax = axes[0]
    metrics = ["R@1", "R@5", "R@10"]
    s2_vals = [stage2_metrics["image_to_text"][m] for m in metrics]
    s3_vals = [stage3_metrics["image_to_text"][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(x - width / 2, s2_vals, width, label="Stage 2", alpha=0.8, color="#3498db")
    ax.bar(x + width / 2, s3_vals, width, label="Stage 3", alpha=0.8, color="#e74c3c")

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Recall (%)", fontsize=12)
    ax.set_title("Image-to-Text Retrieval", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # T2I metrics
    ax = axes[1]
    s2_vals = [stage2_metrics["text_to_image"][m] for m in metrics]
    s3_vals = [stage3_metrics["text_to_image"][m] for m in metrics]

    ax.bar(x - width / 2, s2_vals, width, label="Stage 2", alpha=0.8, color="#3498db")
    ax.bar(x + width / 2, s3_vals, width, label="Stage 3", alpha=0.8, color="#e74c3c")

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Recall (%)", fontsize=12)
    ax.set_title("Text-to-Image Retrieval", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"   Saved: {output_path}")
    plt.close()


def plot_captioning_metrics(stage2_metrics: dict, stage3_metrics: dict, output_path: str):
    """Create bar chart comparing captioning metrics."""

    fig, ax = plt.subplots(figsize=(12, 6))

    # Select key metrics
    metrics = ["Bleu_1", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "SPICE"]

    # Get values (handle missing metrics)
    s2_vals = [stage2_metrics.get(m, 0) for m in metrics]
    s3_vals = [stage3_metrics.get(m, 0) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(x - width / 2, s2_vals, width, label="Stage 2", alpha=0.8, color="#3498db")
    ax.bar(x + width / 2, s3_vals, width, label="Stage 3", alpha=0.8, color="#e74c3c")

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Captioning Metrics Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"   Saved: {output_path}")
    plt.close()


def plot_combined_comparison(
    retrieval_stage2: dict,
    retrieval_stage3: dict,
    captioning_stage2: dict,
    captioning_stage3: dict,
    output_path: str,
):
    """Create combined comparison plot."""

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # I2T
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ["R@1", "R@5", "R@10"]
    s2_vals = [retrieval_stage2["image_to_text"][m] for m in metrics]
    s3_vals = [retrieval_stage3["image_to_text"][m] for m in metrics]
    x = np.arange(len(metrics))
    width = 0.35
    ax1.bar(x - width / 2, s2_vals, width, label="Stage 2", alpha=0.8, color="#3498db")
    ax1.bar(x + width / 2, s3_vals, width, label="Stage 3", alpha=0.8, color="#e74c3c")
    ax1.set_title("Image-to-Text", fontweight="bold")
    ax1.set_ylabel("Recall (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # T2I
    ax2 = fig.add_subplot(gs[0, 1])
    s2_vals = [retrieval_stage2["text_to_image"][m] for m in metrics]
    s3_vals = [retrieval_stage3["text_to_image"][m] for m in metrics]
    ax2.bar(x - width / 2, s2_vals, width, label="Stage 2", alpha=0.8, color="#3498db")
    ax2.bar(x + width / 2, s3_vals, width, label="Stage 3", alpha=0.8, color="#e74c3c")
    ax2.set_title("Text-to-Image", fontweight="bold")
    ax2.set_ylabel("Recall (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # Mean recall
    ax3 = fig.add_subplot(gs[0, 2])
    s2_mean = np.mean(s2_vals + [retrieval_stage2["image_to_text"][m] for m in metrics])
    s3_mean = np.mean(s3_vals + [retrieval_stage3["image_to_text"][m] for m in metrics])
    ax3.bar([0], [s2_mean], 0.5, label="Stage 2", alpha=0.8, color="#3498db")
    ax3.bar([1], [s3_mean], 0.5, label="Stage 3", alpha=0.8, color="#e74c3c")
    ax3.set_title("Mean Recall", fontweight="bold")
    ax3.set_ylabel("Recall (%)")
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(["Stage 2", "Stage 3"])
    ax3.grid(axis="y", alpha=0.3)

    # Captioning metrics
    ax4 = fig.add_subplot(gs[1, :])
    cap_metrics = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "SPICE"]
    s2_vals = [captioning_stage2.get(m, 0) for m in cap_metrics]
    s3_vals = [captioning_stage3.get(m, 0) for m in cap_metrics]
    x = np.arange(len(cap_metrics))
    width = 0.35
    ax4.bar(x - width / 2, s2_vals, width, label="Stage 2", alpha=0.8, color="#3498db")
    ax4.bar(x + width / 2, s3_vals, width, label="Stage 3", alpha=0.8, color="#e74c3c")
    ax4.set_title("Captioning Metrics", fontweight="bold")
    ax4.set_ylabel("Score")
    ax4.set_xlabel("Metric")
    ax4.set_xticks(x)
    ax4.set_xticklabels(cap_metrics, rotation=45, ha="right")
    ax4.legend()
    ax4.grid(axis="y", alpha=0.3)

    plt.suptitle(
        "Karpathy COCO Evaluation: Stage 2 vs Stage 3", fontsize=16, fontweight="bold", y=0.995
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"   Saved: {output_path}")
    plt.close()


def generate_text_report(retrieval_metrics: dict, captioning_metrics: dict, output_path: str):
    """Generate comprehensive text report."""

    lines = []
    lines.append("=" * 80)
    lines.append("KARPATHY COCO EVALUATION: COMPREHENSIVE REPORT")
    lines.append("=" * 80)

    # Summary
    lines.append("\n1. SUMMARY")
    lines.append("-" * 80)
    lines.append("This report presents results from evaluating Stage 2 (hard routing) and")
    lines.append("Stage 3 (soft routing) models on the Karpathy COCO test split.")
    lines.append("")
    lines.append("Evaluation tasks:")
    lines.append("  • Image-Text Retrieval (R@1, R@5, R@10 for both I2T and T2I)")
    lines.append("  • Image Captioning (BLEU, CIDEr, METEOR, SPICE, ROUGE-L)")

    # Retrieval results
    lines.append("\n2. RETRIEVAL RESULTS")
    lines.append("-" * 80)

    lines.append("\n2.1 Image-to-Text (I2T)")
    for metric in ["R@1", "R@5", "R@10"]:
        s2 = retrieval_metrics["stage2"]["image_to_text"][metric]
        s3 = retrieval_metrics["stage3"]["image_to_text"][metric]
        delta = s3 - s2
        lines.append(f"  {metric:<6} Stage 2: {s2:6.2f}%   Stage 3: {s3:6.2f}%   Δ: {delta:+.2f}%")

    lines.append("\n2.2 Text-to-Image (T2I)")
    for metric in ["R@1", "R@5", "R@10"]:
        s2 = retrieval_metrics["stage2"]["text_to_image"][metric]
        s3 = retrieval_metrics["stage3"]["text_to_image"][metric]
        delta = s3 - s2
        lines.append(f"  {metric:<6} Stage 2: {s2:6.2f}%   Stage 3: {s3:6.2f}%   Δ: {delta:+.2f}%")

    # Captioning results
    lines.append("\n3. CAPTIONING RESULTS")
    lines.append("-" * 80)

    for metric in ["Bleu_1", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "SPICE"]:
        if metric in captioning_metrics["stage2"] and metric in captioning_metrics["stage3"]:
            s2 = captioning_metrics["stage2"][metric]
            s3 = captioning_metrics["stage3"][metric]
            delta = s3 - s2
            lines.append(f"  {metric:<10} Stage 2: {s2:.4f}   Stage 3: {s3:.4f}   Δ: {delta:+.4f}")

    # Analysis
    lines.append("\n4. ANALYSIS")
    lines.append("-" * 80)

    # Determine winner
    cider_s2 = captioning_metrics["stage2"].get("CIDEr", 0)
    cider_s3 = captioning_metrics["stage3"].get("CIDEr", 0)

    i2t_r1_s2 = retrieval_metrics["stage2"]["image_to_text"]["R@1"]
    i2t_r1_s3 = retrieval_metrics["stage3"]["image_to_text"]["R@1"]

    lines.append("\n4.1 Key Findings:")

    if cider_s3 > cider_s2:
        lines.append(f"  • Stage 3 achieves higher CIDEr ({cider_s3:.4f} vs {cider_s2:.4f})")
        lines.append("    CIDEr is the primary metric for captioning quality")
    else:
        lines.append(f"  • Stage 2 achieves higher CIDEr ({cider_s2:.4f} vs {cider_s3:.4f})")
        lines.append("    CIDEr is the primary metric for captioning quality")

    if i2t_r1_s3 > i2t_r1_s2:
        lines.append(f"  • Stage 3 achieves higher I2T R@1 ({i2t_r1_s3:.2f}% vs {i2t_r1_s2:.2f}%)")
    else:
        lines.append(f"  • Stage 2 achieves higher I2T R@1 ({i2t_r1_s2:.2f}% vs {i2t_r1_s3:.2f}%)")

    lines.append("\n4.2 Interpretation:")
    lines.append("  • Retrieval metrics measure cross-modal alignment quality")
    lines.append("  • Captioning metrics measure generative language quality")
    lines.append("  • Differences reflect trade-offs between hard and soft routing mechanisms")

    lines.append("\n5. CONCLUSION")
    lines.append("-" * 80)
    lines.append("Both models demonstrate competitive performance on the Karpathy COCO benchmark.")
    lines.append("Results enable comparison with published baselines (CLIP, BLIP, ALBEF, etc.).")
    lines.append("These standardized metrics complement the compositional analysis findings.")

    lines.append("\n" + "=" * 80)

    report = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(report)

    print(f"   Saved: {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Visualize evaluation results")
    parser.add_argument(
        "--retrieval_metrics",
        type=str,
        default="results/karpathy_evaluation/retrieval/retrieval_metrics.json",
        help="Path to retrieval metrics JSON",
    )
    parser.add_argument(
        "--captioning_metrics",
        type=str,
        default="results/karpathy_evaluation/captioning/captioning_metrics.json",
        help="Path to captioning metrics JSON",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/karpathy_evaluation",
        help="Output directory for visualizations",
    )

    args = parser.parse_args()

    print_banner("VISUALIZATION & REPORTING")

    # Load metrics
    print("\n📂 Loading metrics...")
    retrieval_metrics = load_json(args.retrieval_metrics)
    captioning_metrics = load_json(args.captioning_metrics)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("\n📊 Generating plots...")

    plot_retrieval_metrics(
        retrieval_metrics["stage2"],
        retrieval_metrics["stage3"],
        str(output_dir / "retrieval_comparison.png"),
    )

    plot_captioning_metrics(
        captioning_metrics["stage2"],
        captioning_metrics["stage3"],
        str(output_dir / "captioning_comparison.png"),
    )

    plot_combined_comparison(
        retrieval_metrics["stage2"],
        retrieval_metrics["stage3"],
        captioning_metrics["stage2"],
        captioning_metrics["stage3"],
        str(output_dir / "combined_comparison.png"),
    )

    # Generate text report
    print("\n📝 Generating text report...")
    report = generate_text_report(
        retrieval_metrics, captioning_metrics, str(output_dir / "evaluation_report.txt")
    )

    # Print report
    print("\n" + report)

    print_banner("✅ VISUALIZATION COMPLETE")


if __name__ == "__main__":
    main()
