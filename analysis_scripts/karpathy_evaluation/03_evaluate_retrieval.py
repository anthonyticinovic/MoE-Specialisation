#!/usr/bin/env python3
"""
Step 3: Evaluate retrieval metrics (Image-to-Text and Text-to-Image).
Computes R@1, R@5, R@10 for both directions.
"""

import argparse
from pathlib import Path

import numpy as np
from karpathy_utils import print_banner, save_json


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalize embeddings for cosine similarity."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, 1e-8, None)


def compute_similarity_matrix(
    image_embeddings: np.ndarray, text_embeddings: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity matrix between images and texts.

    Args:
        image_embeddings: (num_images, dim)
        text_embeddings: (num_captions, dim)

    Returns:
        similarity: (num_images, num_captions) matrix
    """
    print("\n📊 Computing similarity matrix...")
    print(f"   Image embeddings: {image_embeddings.shape}")
    print(f"   Text embeddings: {text_embeddings.shape}")

    # Normalize for cosine similarity
    image_emb_norm = normalize_embeddings(image_embeddings)
    text_emb_norm = normalize_embeddings(text_embeddings)

    # Compute similarity: (num_images, num_captions)
    similarity = image_emb_norm @ text_emb_norm.T

    print(f"   Similarity matrix: {similarity.shape}")
    print(f"   Range: [{similarity.min():.4f}, {similarity.max():.4f}]")
    print(f"   Mean: {similarity.mean():.4f}")

    return similarity


def evaluate_image_to_text(similarity: np.ndarray, num_captions_per_image: int = 5) -> dict:
    """
    Evaluate Image-to-Text retrieval.

    For each image, rank all captions and check if any of the 5 ground-truth
    captions appear in top-k.

    Args:
        similarity: (num_images, num_captions) similarity matrix
        num_captions_per_image: Number of captions per image (default: 5)

    Returns:
        metrics: Dict with R@1, R@5, R@10
    """
    num_images = similarity.shape[0]

    ranks = []

    print("\n📸→💬 Evaluating Image-to-Text retrieval...")

    for img_idx in range(num_images):
        # Get similarities for this image to all captions
        img_sims = similarity[img_idx]

        # Ground truth caption indices (5 captions per image)
        gt_caption_start = img_idx * num_captions_per_image
        gt_caption_indices = set(range(gt_caption_start, gt_caption_start + num_captions_per_image))

        # Rank captions by similarity (descending)
        ranked_indices = np.argsort(-img_sims)

        # Find rank of first ground-truth caption
        for rank, cap_idx in enumerate(ranked_indices):
            if cap_idx in gt_caption_indices:
                ranks.append(rank + 1)  # 1-indexed rank
                break

    ranks = np.array(ranks)

    # Compute recall at k
    r1 = 100.0 * np.mean(ranks <= 1)
    r5 = 100.0 * np.mean(ranks <= 5)
    r10 = 100.0 * np.mean(ranks <= 10)

    metrics = {
        "R@1": r1,
        "R@5": r5,
        "R@10": r10,
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
    }

    print(f"   R@1:  {r1:.2f}%")
    print(f"   R@5:  {r5:.2f}%")
    print(f"   R@10: {r10:.2f}%")
    print(f"   Median rank: {metrics['median_rank']:.1f}")

    return metrics


def evaluate_text_to_image(similarity: np.ndarray, num_captions_per_image: int = 5) -> dict:
    """
    Evaluate Text-to-Image retrieval.

    For each caption, rank all images and check if the ground-truth image
    appears in top-k.

    Args:
        similarity: (num_images, num_captions) similarity matrix
        num_captions_per_image: Number of captions per image (default: 5)

    Returns:
        metrics: Dict with R@1, R@5, R@10
    """
    num_images, num_captions = similarity.shape

    ranks = []

    print("\n💬→📸 Evaluating Text-to-Image retrieval...")

    for cap_idx in range(num_captions):
        # Get similarities for this caption to all images
        cap_sims = similarity[:, cap_idx]

        # Ground truth image index
        gt_image_idx = cap_idx // num_captions_per_image

        # Rank images by similarity (descending)
        ranked_indices = np.argsort(-cap_sims)

        # Find rank of ground-truth image
        rank = np.where(ranked_indices == gt_image_idx)[0][0] + 1  # 1-indexed
        ranks.append(rank)

    ranks = np.array(ranks)

    # Compute recall at k
    r1 = 100.0 * np.mean(ranks <= 1)
    r5 = 100.0 * np.mean(ranks <= 5)
    r10 = 100.0 * np.mean(ranks <= 10)

    metrics = {
        "R@1": r1,
        "R@5": r5,
        "R@10": r10,
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
    }

    print(f"   R@1:  {r1:.2f}%")
    print(f"   R@5:  {r5:.2f}%")
    print(f"   R@10: {r10:.2f}%")
    print(f"   Median rank: {metrics['median_rank']:.1f}")

    return metrics


def format_comparison_table(stage2_metrics: dict, stage3_metrics: dict) -> str:
    """Format a comparison table for Stage 2 vs Stage 3."""

    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("RETRIEVAL METRICS COMPARISON: STAGE 2 vs STAGE 3")
    lines.append("=" * 80)

    lines.append("\nImage-to-Text Retrieval (I2T):")
    lines.append("-" * 80)
    lines.append(f"{'Metric':<20} {'Stage 2':<15} {'Stage 3':<15} {'Δ (S3-S2)':<15}")
    lines.append("-" * 80)

    for metric in ["R@1", "R@5", "R@10"]:
        s2_val = stage2_metrics["image_to_text"][metric]
        s3_val = stage3_metrics["image_to_text"][metric]
        delta = s3_val - s2_val
        delta_str = f"{delta:+.2f}%"

        lines.append(f"{metric:<20} {s2_val:>6.2f}%        {s3_val:>6.2f}%        {delta_str:>10}")

    lines.append("\nText-to-Image Retrieval (T2I):")
    lines.append("-" * 80)
    lines.append(f"{'Metric':<20} {'Stage 2':<15} {'Stage 3':<15} {'Δ (S3-S2)':<15}")
    lines.append("-" * 80)

    for metric in ["R@1", "R@5", "R@10"]:
        s2_val = stage2_metrics["text_to_image"][metric]
        s3_val = stage3_metrics["text_to_image"][metric]
        delta = s3_val - s2_val
        delta_str = f"{delta:+.2f}%"

        lines.append(f"{metric:<20} {s2_val:>6.2f}%        {s3_val:>6.2f}%        {delta_str:>10}")

    # Mean recall
    lines.append("\nMean Recall (average of all 6 metrics):")
    lines.append("-" * 80)
    s2_mean = np.mean(
        [
            stage2_metrics["image_to_text"]["R@1"],
            stage2_metrics["image_to_text"]["R@5"],
            stage2_metrics["image_to_text"]["R@10"],
            stage2_metrics["text_to_image"]["R@1"],
            stage2_metrics["text_to_image"]["R@5"],
            stage2_metrics["text_to_image"]["R@10"],
        ]
    )
    s3_mean = np.mean(
        [
            stage3_metrics["image_to_text"]["R@1"],
            stage3_metrics["image_to_text"]["R@5"],
            stage3_metrics["image_to_text"]["R@10"],
            stage3_metrics["text_to_image"]["R@1"],
            stage3_metrics["text_to_image"]["R@5"],
            stage3_metrics["text_to_image"]["R@10"],
        ]
    )
    delta_mean = s3_mean - s2_mean

    lines.append(
        f"{'Mean Recall':<20} {s2_mean:>6.2f}%        {s3_mean:>6.2f}%        {delta_mean:+.2f}%"
    )
    lines.append("=" * 80)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval metrics")
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default="results/karpathy_evaluation/retrieval",
        help="Directory with embeddings",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/karpathy_evaluation/retrieval",
        help="Output directory for metrics",
    )

    args = parser.parse_args()

    print_banner("RETRIEVAL EVALUATION")

    embeddings_dir = Path(args.embeddings_dir)

    # Load embeddings
    print("\n📂 Loading embeddings...")

    stage2_img = np.load(embeddings_dir / "stage2_image_embeddings.npy")
    stage2_txt = np.load(embeddings_dir / "stage2_text_embeddings.npy")
    stage3_img = np.load(embeddings_dir / "stage3_image_embeddings.npy")
    stage3_txt = np.load(embeddings_dir / "stage3_text_embeddings.npy")

    print(f"   Stage 2 images: {stage2_img.shape}")
    print(f"   Stage 2 texts: {stage2_txt.shape}")
    print(f"   Stage 3 images: {stage3_img.shape}")
    print(f"   Stage 3 texts: {stage3_txt.shape}")

    # Evaluate Stage 2
    print("\n" + "=" * 80)
    print("STAGE 2 EVALUATION")
    print("=" * 80)

    sim_stage2 = compute_similarity_matrix(stage2_img, stage2_txt)
    i2t_stage2 = evaluate_image_to_text(sim_stage2)
    t2i_stage2 = evaluate_text_to_image(sim_stage2)

    stage2_metrics = {"image_to_text": i2t_stage2, "text_to_image": t2i_stage2}

    # Evaluate Stage 3
    print("\n" + "=" * 80)
    print("STAGE 3 EVALUATION")
    print("=" * 80)

    sim_stage3 = compute_similarity_matrix(stage3_img, stage3_txt)
    i2t_stage3 = evaluate_image_to_text(sim_stage3)
    t2i_stage3 = evaluate_text_to_image(sim_stage3)

    stage3_metrics = {"image_to_text": i2t_stage3, "text_to_image": t2i_stage3}

    # Save metrics
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {"stage2": stage2_metrics, "stage3": stage3_metrics}

    metrics_path = output_dir / "retrieval_metrics.json"
    save_json(all_metrics, str(metrics_path))

    # Print comparison table
    comparison = format_comparison_table(stage2_metrics, stage3_metrics)
    print(comparison)

    # Save comparison to text file
    comparison_path = output_dir / "retrieval_comparison.txt"
    with open(comparison_path, "w") as f:
        f.write(comparison)
    print(f"\n💾 Saved comparison: {comparison_path}")

    print_banner("✅ RETRIEVAL EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
