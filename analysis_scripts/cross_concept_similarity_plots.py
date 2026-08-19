"""Plots and result files for the cross-concept similarity matrix.

The 2N×2N heatmap, the cross-modal-only view, the colour-coherence score that
summarises them, and the JSON writer. Split from the analyser so neither file
runs past the 800-line guideline.

A mixin rather than free functions to keep the analyser's method set — and so
the resolution order seen by ``compositional_case_study`` — unchanged.
"""

from __future__ import annotations

import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)


class SimilarityMatrixPlotsMixin:
    def visualize_matrix(self, matrix: np.ndarray, labels: list[str], output_dir: str, layer: int):
        """
        Create and save heatmap visualisations of similarity matrix.

        Generates two plots:
        1. Standard: Single color scale for all values
        2. Dual-scale: Separate color scales for within-modality vs cross-modality

        Args:
            matrix: 2N×2N similarity matrix
            labels: List of row/column labels
            output_dir: Directory to save plot
            layer: Layer number (for title and filename)
        """
        os.makedirs(output_dir, exist_ok=True)

        # Determine figure size based on matrix size
        n = matrix.shape[0]
        figsize = max(10, n * 0.6)
        mode_str = (
            "Forced Routing"
            if self.mode == "stage2"
            else f"Learned Soft Routing (T={self.temperature})"
        )

        # ============================================================
        # PLOT 1: Standard single color scale (original)
        # ============================================================
        fig, ax = plt.subplots(figsize=(figsize, figsize))

        # Create mask for upper triangle (exclude diagonal)
        mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)

        # Create heatmap
        sns.heatmap(
            matrix,
            mask=mask,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=-1,
            vmax=1,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar_kws={"label": "Cosine Similarity"},
            square=True,
            linewidths=0.5,
            linecolor="lightgray",
        )

        # Customize labels
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(labels, rotation=0, fontsize=9)

        # Add title
        ax.set_title(
            f"Cross-Concept Similarity Matrix (Layer {layer})\n"
            f"{n // 2} Concepts | Stage 3 | Soft Routing",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(output_dir, f"similarity_matrix_layer{layer}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"   Saved standard heatmap to: {plot_path}")

        # ============================================================
        # PLOT 2: Cross-modal only (txt vs img)
        # ============================================================
        self._visualize_cross_modal_only(matrix, labels, output_dir, layer, mode_str)

    def _visualize_cross_modal_only(
        self, matrix: np.ndarray, labels: list[str], output_dir: str, layer: int, mode_str: str
    ):
        """
        Create focused heatmap showing only cross-modal (txt↔img) similarities.

        Extracts the bottom-left quadrant (txt rows × img columns) and displays
        it with an optimised color scale for maximum sensitivity.

        Args:
            matrix: 2N×2N similarity matrix
            labels: List of row/column labels
            output_dir: Directory to save plot
            layer: Layer number
            mode_str: Mode description string for title
        """
        n = matrix.shape[0]
        half_n = n // 2  # Split point between img and txt

        # Extract cross-modal submatrix (txt rows × img columns)
        cross_modal_matrix = matrix[half_n:, :half_n]

        # Extract corresponding labels
        img_labels = [label.replace("img:", "") for label in labels[:half_n]]
        txt_labels = [label.replace("txt:", "") for label in labels[half_n:]]

        # Compute statistics
        logger.info("\n   Cross-modal only statistics:")
        logger.info(f"      Shape: {cross_modal_matrix.shape} (txt × img)")
        logger.info(
            f"      Range: [{cross_modal_matrix.min():.3f}, {cross_modal_matrix.max():.3f}]"
        )
        logger.info(f"      Mean: {cross_modal_matrix.mean():.3f}")
        logger.info(f"      Std: {cross_modal_matrix.std():.3f}")

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))

        # Create heatmap with optimised color scale
        sns.heatmap(
            cross_modal_matrix,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=cross_modal_matrix.min() - 0.005,  # Add small margin
            vmax=cross_modal_matrix.max() + 0.005,
            xticklabels=img_labels,
            yticklabels=txt_labels,
            ax=ax,
            cbar_kws={"label": "Cosine Similarity (Cross-Modal)"},
            square=True,
            linewidths=0.5,
            linecolor="lightgray",
        )

        # Customize labels
        ax.set_xlabel("Image Concepts", fontsize=12, fontweight="bold")
        ax.set_ylabel("Text Concepts", fontsize=12, fontweight="bold")
        ax.set_xticklabels(img_labels, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(txt_labels, rotation=0, fontsize=10)

        # Add title
        ax.set_title(
            f"Cross-Modal Similarity Matrix (Layer {layer})\n"
            f"Text ↔ Image Alignment | {half_n} Concepts | {mode_str}",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(output_dir, f"similarity_matrix_layer{layer}_cross_modal.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"   Saved cross-modal heatmap to: {plot_path}")

    def save_results(self, matrix: np.ndarray, labels: list[str], output_dir: str, layer: int):
        """
        Save similarity matrix and labels to JSON files.

        Args:
            matrix: 2N×2N similarity matrix
            labels: List of row/column labels
            output_dir: Directory to save results
            layer: Layer number (for filename)
        """
        os.makedirs(output_dir, exist_ok=True)

        # Compute color coherence score
        logger.info("\n   Computing Color Coherence Score (CCS)...")
        ccs_results = self.compute_color_coherence_score(matrix, labels)

        if ccs_results["color_coherence_score"] is not None:
            logger.info(f"   Color Coherence Score: {ccs_results['color_coherence_score']:.3f}")
            logger.info(
                f"      • Same color, diff object: "
                f"{ccs_results['same_color_diff_object_mean']:.3f} "
                f"(n={ccs_results['same_color_diff_object_count']} pairs)"
            )
            logger.info(
                f"      • Diff color, same object: "
                f"{ccs_results['diff_color_same_object_mean']:.3f} "
                f"(n={ccs_results['diff_color_same_object_count']} pairs)"
            )

            # Interpret the score
            ccs = ccs_results["color_coherence_score"]
            if ccs > 1.05:
                interpretation = "Strong color binding (color disentangled from object)"
            elif ccs > 0.95:
                interpretation = "Weak/no color binding (color-object entangled)"
            else:
                interpretation = "Category dominance (object identity dominates over color)"
            logger.info(f"      • Interpretation: {interpretation}")

        # Save matrix as JSON (convert to list for JSON serialization)
        matrix_path = os.path.join(output_dir, f"similarity_matrix_layer{layer}.json")
        with open(matrix_path, "w") as f:
            json.dump(
                {
                    "matrix": matrix.tolist(),
                    "layer": layer,
                    "shape": list(matrix.shape),
                    "color_coherence_score": ccs_results,
                },
                f,
                indent=2,
            )
        logger.info(f"   Saved matrix to: {matrix_path}")

        # Save labels as JSON
        labels_path = os.path.join(output_dir, f"labels_layer{layer}.json")
        with open(labels_path, "w") as f:
            json.dump(
                {"labels": labels, "layer": layer, "num_pairs": len(labels) // 2}, f, indent=2
            )
        logger.info(f"   Saved labels to: {labels_path}")

    def compute_color_coherence_score(self, matrix: np.ndarray, labels: list[str]) -> dict:
        """
        Compute Color Coherence Score (CCS) to measure color-object binding.

        CCS = mean(same_color_diff_object) / mean(diff_color_same_object)

        CCS > 1.0: Strong color binding (color is disentangled from object)
        CCS ≈ 1.0: No color binding (color and object are entangled)
        CCS < 1.0: Anti-binding (objects dominate over color)

        Args:
            matrix: 2N×2N similarity matrix
            labels: List of row/column labels (format: "modality:concept")

        Returns:
            Dictionary containing CCS scores and component similarities
        """
        # Parse labels to extract concepts
        # Format: "img:red_apple", "txt:blue_car", etc.
        concepts = []
        for label in labels:
            if ":" in label:
                modality, concept = label.split(":", 1)
                concepts.append({"modality": modality, "concept": concept})
            else:
                concepts.append({"modality": "unknown", "concept": label})

        # Extract color and object from concept names (assumes format: color_object or just object)
        def parse_concept(concept_str):
            """Parse 'red_apple' -> ('red', 'apple'), 'banana' -> (None, 'banana')"""
            parts = concept_str.split("_", 1)
            if len(parts) == 2:
                color, obj = parts
                # Check if first part is actually a color (heuristic)
                color_words = [
                    "red",
                    "green",
                    "blue",
                    "yellow",
                    "white",
                    "black",
                    "pink",
                    "orange",
                    "purple",
                ]
                if color.lower() in color_words:
                    return (color, obj)
            return (None, concept_str)

        parsed_concepts = [parse_concept(c["concept"]) for c in concepts]

        # Check if we have colored concepts
        has_colors = any(color is not None for color, _ in parsed_concepts)

        if not has_colors:
            logger.warning("\n    No colored concepts detected (format should be 'color_object')")
            return {
                "color_coherence_score": None,
                "same_color_diff_object_mean": None,
                "diff_color_same_object_mean": None,
                "note": "No colored concepts found",
            }

        # Extract cross-modal similarities (txt rows × img columns)
        n = len(labels)
        n_concepts = n // 2

        # Indices: first half are images, second half are text
        img_indices = list(range(n_concepts))
        txt_indices = list(range(n_concepts, n))

        same_color_diff_object = []
        diff_color_same_object = []

        # Compare all txt-img pairs
        for txt_idx in txt_indices:
            txt_color, txt_obj = parsed_concepts[txt_idx]
            if txt_color is None:
                continue

            for img_idx in img_indices:
                img_color, img_obj = parsed_concepts[img_idx]
                if img_color is None:
                    continue

                similarity = matrix[txt_idx, img_idx]

                # Same color, different object (e.g., red_apple text vs red_car image)
                if txt_color == img_color and txt_obj != img_obj:
                    same_color_diff_object.append(similarity)

                # Different color, same object (e.g., red_apple text vs green_apple image)
                elif txt_color != img_color and txt_obj == img_obj:
                    diff_color_same_object.append(similarity)

        # Compute metrics
        if len(same_color_diff_object) == 0 or len(diff_color_same_object) == 0:
            logger.warning("\n    Insufficient color-object pairs for CCS computation")
            return {
                "color_coherence_score": None,
                "same_color_diff_object_mean": None,
                "diff_color_same_object_mean": None,
                "note": "Insufficient pairs",
            }

        same_color_mean = np.mean(same_color_diff_object)
        diff_color_mean = np.mean(diff_color_same_object)
        ccs = same_color_mean / diff_color_mean if diff_color_mean != 0 else None

        return {
            "color_coherence_score": ccs,
            "same_color_diff_object_mean": same_color_mean,
            "diff_color_same_object_mean": diff_color_mean,
            "same_color_diff_object_count": len(same_color_diff_object),
            "diff_color_same_object_count": len(diff_color_same_object),
        }
