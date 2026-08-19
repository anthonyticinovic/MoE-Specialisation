"""
Compositional Case Study: Stage 2 vs Stage 3 Representation Analysis

This script compares how Stage 2 (hard routing) vs Stage 3 (soft routing) represent
objects with different attributes (e.g., color, shape, size) using similarity matrices.

For N stimuli, computes 2N×2N similarity matrices (vision + text representations):
- Stage 2: Forced routing (vision→expert0, text→expert1)
- Stage 3: Learned soft routing (natural model behaviour)

Output: One matrix per stage per layer (e.g., stage2_layer31.png, stage3_layer31.png)

Usage:
    python analysis_scripts/compositional_case_study.py \\
        --config-file configs/compositional_case_study.json

Config file format (JSON):
    {
      "manifest_file": "data/compositional_stimuli.json",
      "layers": [0, 16, 31],
      "pooling": "mean",
      "temperature": 0.01,
      "output_dir": "results/compositional_case_study/",
      "stage2_checkpoint": "/path/to/stage2_checkpoint.pth",
      "stage3_checkpoint": "/path/to/stage3_checkpoint.pth"
    }

Manifest file format (JSON):
    {
      "stimuli": [
        {"id": "red_apple", "image_path": "data/images/red_apple.jpg", "caption": "A red apple"},
        {"id": "green_apple", "image_path": "data/images/green_apple.jpg",
         "caption": "A green apple"},
        ...
      ]
    }
"""

import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from analysis_scripts._lib import compute_cosine_similarity_matrix, load_analysis_config

# REUSE: Import existing analyzer class for model loading and representation extraction
from analysis_scripts.cross_concept_similarity_matrix import CrossConceptSimilarityAnalyzer
from models.utils.common import get_device, setup_logging

logger = logging.getLogger(__name__)


class CompositionalCaseStudyAnalyzer:
    """
    Analyzer for compositional representation case study comparing Stage 2 vs Stage 3.

    Heavily reuses CrossConceptSimilarityAnalyzer for:
    - Model loading (Stage 2 and Stage 3)
    - Representation extraction (vision and text, mean pooling)
    - Similarity matrix computation
    - Visualisation

    Key differences from CrossConceptSimilarityAnalyzer:
    - Uses pre-specified stimuli from JSON manifest (not COCO sampling)
    - Runs both Stage 2 AND Stage 3 in single script
    - Compares representations across stages
    """

    def __init__(
        self,
        config_path: str | None = None,
        device: str | None = None,
        temperature: float = 0.01,
    ):
        """
        Initialize analyzer.

        Args:
            config_path: Path to training configuration YAML
            device: Device to run on (cuda/cpu)
            temperature: Routing temperature for Stage 3
        """
        logger.info("Initialising Compositional Case Study Analyzer")
        logger.info(f"   Device: {device}")
        logger.info(f"   Temperature: {temperature}")

        self.config_path = config_path
        self.device = device or get_device()
        self.temperature = temperature

        # Will hold analyzers for each stage
        self.stage2_analyzer = None
        self.stage3_analyzer = None

    def load_manifest(self, manifest_file: str) -> list[dict]:
        """
        Load stimulus manifest from JSON file.

        Args:
            manifest_file: Path to JSON manifest with stimuli

        Returns:
            List of stimulus dicts with keys: id, image_path, caption
        """
        logger.info(f"\nLoading stimulus manifest from: {manifest_file}")

        with open(manifest_file) as f:
            manifest_data = json.load(f)

        stimuli = manifest_data.get("stimuli", [])

        # Validate manifest format
        required_fields = ["id", "image_path", "caption"]
        for idx, stimulus in enumerate(stimuli):
            for field in required_fields:
                if field not in stimulus:
                    raise ValueError(
                        f"Stimulus {idx} missing required field: {field}\n"
                        f"Required fields: {required_fields}"
                    )

        logger.info(f"   Loaded {len(stimuli)} stimuli:")
        for stimulus in stimuli:
            logger.info(f"      - {stimulus['id']}: {stimulus['caption']}")

        return stimuli

    def initialize_stage_analyzers(self, stage2_checkpoint: str, stage3_checkpoint: str):
        """
        Initialize separate analyzers for Stage 2 and Stage 3.

        REUSES: CrossConceptSimilarityAnalyzer class for both stages

        Args:
            stage2_checkpoint: Path to Stage 2 checkpoint
            stage3_checkpoint: Path to Stage 3 checkpoint
        """
        logger.info("\nInitializing Stage 2 analyzer...")
        self.stage2_analyzer = CrossConceptSimilarityAnalyzer(
            config_path=self.config_path,
            device=self.device,
            mode="stage2",
            stage2_checkpoint=stage2_checkpoint,
            temperature=self.temperature,
        )
        self.stage2_analyzer.load_models()
        logger.info("   Stage 2 analyzer ready")

        logger.info("\nInitializing Stage 3 analyzer...")
        self.stage3_analyzer = CrossConceptSimilarityAnalyzer(
            config_path=self.config_path,
            device=self.device,
            mode="stage3",
            stage3_checkpoint=stage3_checkpoint,
            temperature=self.temperature,
        )
        self.stage3_analyzer.load_models()
        logger.info("   Stage 3 analyzer ready")

    def extract_stimulus_representations(
        self,
        stimuli: list[dict],
        layer: int,
        pooling: str,
        analyzer: CrossConceptSimilarityAnalyzer,
        stage_name: str,
    ) -> tuple[list[np.ndarray], list[str]]:
        """
        Extract representations for all stimuli (vision + text).

        REUSES: analyzer._extract_representation() for both modalities

        Args:
            stimuli: List of stimulus dicts (id, image_path, caption)
            layer: Layer to extract from
            pooling: Pooling strategy (mean)
            analyzer: Stage-specific analyzer (Stage 2 or Stage 3)
            stage_name: Stage name for logging (e.g., "Stage 2")

        Returns:
            Tuple of (representations, labels) where:
                - representations: List of 2N numpy arrays [hidden_dim]
                - labels: List of 2N strings (img:id, txt:id)
        """
        N = len(stimuli)
        logger.info(f"\nExtracting {stage_name} representations at layer {layer}")
        logger.info(f"   Stimuli: {N}, Total entries: {2 * N} (vision + text)")

        representations = []
        labels = []

        # Extract vision representations (first N entries)
        logger.info("\nExtracting vision representations...")
        for stimulus in stimuli:
            logger.info(f"   Processing: {stimulus['id']}")

            # REUSE: analyzer._extract_representation() with vision modality
            img_rep = analyzer._extract_representation(
                concept=stimulus["image_path"],
                expert="vision",
                layer=layer,
                modality="vision",
                pooling=pooling,
            )

            representations.append(img_rep)
            labels.append(f"img:{stimulus['id']}")
            logger.info(f"      Vision: norm={np.linalg.norm(img_rep):.2f}")

        # Extract text representations (next N entries)
        logger.info("\nExtracting text representations...")
        for stimulus in stimuli:
            logger.info(f"   Processing: {stimulus['id']}")

            # REUSE: analyzer._extract_representation() with text modality
            txt_rep = analyzer._extract_representation(
                concept=stimulus["caption"],
                expert="text",
                layer=layer,
                modality="text",
                pooling=pooling,
            )

            representations.append(txt_rep)
            labels.append(f"txt:{stimulus['id']}")
            logger.info(f"      Text: norm={np.linalg.norm(txt_rep):.2f}")

        logger.info(f"\n   Extracted {2 * N} representations ({N} vision + {N} text)")
        return representations, labels

    def compute_similarity_matrix(
        self, representations: list[np.ndarray], stage_name: str
    ) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix.

        REUSES: Same cosine similarity formula as CrossConceptSimilarityAnalyzer

        Args:
            representations: List of N representations [hidden_dim]
            stage_name: Stage name for logging

        Returns:
            N×N similarity matrix
        """
        n = len(representations)
        logger.info(f"\nComputing {stage_name} similarity matrix ({n}×{n})...")

        matrix = compute_cosine_similarity_matrix(representations)

        logger.info(f"   Matrix computed: shape={matrix.shape}")
        logger.info(f"   Similarity range: [{matrix.min():.3f}, {matrix.max():.3f}]")
        logger.info(f"   Mean similarity: {matrix.mean():.3f}")

        return matrix

    def save_results(
        self, matrix: np.ndarray, labels: list[str], output_dir: str, stage_name: str, layer: int
    ):
        """
        Save similarity matrix and labels to JSON.

        Args:
            matrix: Similarity matrix
            labels: Row/column labels
            output_dir: Output directory
            stage_name: Stage name (stage2/stage3)
            layer: Layer number
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save matrix
        matrix_path = os.path.join(output_dir, f"{stage_name}_layer{layer}_matrix.json")
        with open(matrix_path, "w") as f:
            json.dump(
                {
                    "stage": stage_name,
                    "layer": layer,
                    "matrix": matrix.tolist(),
                    "shape": list(matrix.shape),
                },
                f,
                indent=2,
            )
        logger.info(f"   Saved matrix to: {matrix_path}")

        # Save labels
        labels_path = os.path.join(output_dir, f"{stage_name}_layer{layer}_labels.json")
        with open(labels_path, "w") as f:
            json.dump(
                {
                    "stage": stage_name,
                    "layer": layer,
                    "labels": labels,
                    "num_stimuli": len(labels) // 2,
                },
                f,
                indent=2,
            )
        logger.info(f"   Saved labels to: {labels_path}")

    def visualize_matrix(
        self,
        matrix: np.ndarray,
        labels: list[str],
        output_dir: str,
        stage_name: str,
        layer: int,
        temperature: float,
    ):
        """
        Create heatmap visualisation of similarity matrix.

        REUSES: Similar visualisation style as CrossConceptSimilarityAnalyzer

        Args:
            matrix: Similarity matrix
            labels: Row/column labels
            output_dir: Output directory
            stage_name: Stage name (stage2/stage3)
            layer: Layer number
            temperature: Temperature (for Stage 3 title)
        """
        os.makedirs(output_dir, exist_ok=True)

        n = matrix.shape[0]
        num_stimuli = n // 2
        figsize = max(10, n * 0.8)

        # Determine mode string for title
        if stage_name == "stage2":
            mode_str = "Hard Routing (Forced)"
        else:
            mode_str = f"Soft Routing (T={temperature})"

        # Create figure
        fig, ax = plt.subplots(figsize=(figsize, figsize))

        # Create mask for upper triangle (exclude diagonal)
        mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)

        # REUSE: Same heatmap style as existing scripts
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
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(labels, rotation=0, fontsize=10)

        # Add title
        ax.set_title(
            f"Compositional Case Study: {stage_name.upper()} (Layer {layer})\n"
            f"{num_stimuli} Stimuli | {mode_str}",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(output_dir, f"{stage_name}_layer{layer}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"   Saved heatmap to: {plot_path}")

    def visualize_cross_modal_comparison(
        self,
        stage2_matrix: np.ndarray,
        stage3_matrix: np.ndarray,
        labels: list[str],
        output_dir: str,
        layer: int,
    ):
        """
        Create side-by-side comparison of Stage 2 vs Stage 3 cross-modal similarities.

        Shows only the txt×img quadrants for focused comparison.

        Args:
            stage2_matrix: Stage 2 similarity matrix
            stage3_matrix: Stage 3 similarity matrix
            labels: Row/column labels
            output_dir: Output directory
            layer: Layer number
        """
        n = stage2_matrix.shape[0]
        half_n = n // 2

        # Extract cross-modal submatrices (txt rows × img columns)
        stage2_cross = stage2_matrix[half_n:, :half_n]
        stage3_cross = stage3_matrix[half_n:, :half_n]

        # Extract labels
        img_labels = [label.replace("img:", "") for label in labels[:half_n]]
        txt_labels = [label.replace("txt:", "") for label in labels[half_n:]]

        logger.info("\nCreating cross-modal comparison visualization...")
        logger.info(
            f"   Stage 2 cross-modal range: [{stage2_cross.min():.3f}, {stage2_cross.max():.3f}]"
        )
        logger.info(
            f"   Stage 3 cross-modal range: [{stage3_cross.min():.3f}, {stage3_cross.max():.3f}]"
        )

        # Create side-by-side figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

        # Compute INDEPENDENT color scales for maximum sensitivity per stage
        stage2_vmin = stage2_cross.min() - 0.005
        stage2_vmax = stage2_cross.max() + 0.005
        stage3_vmin = stage3_cross.min() - 0.005
        stage3_vmax = stage3_cross.max() + 0.005

        logger.info(f"   Stage 2 color scale: [{stage2_vmin:.3f}, {stage2_vmax:.3f}]")
        logger.info(f"   Stage 3 color scale: [{stage3_vmin:.3f}, {stage3_vmax:.3f}]")

        # Plot Stage 2 with its own scale
        sns.heatmap(
            stage2_cross,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=stage2_vmin,
            vmax=stage2_vmax,
            xticklabels=img_labels,
            yticklabels=txt_labels,
            ax=ax1,
            cbar_kws={"label": "Cosine Similarity (Stage 2)"},
            square=True,
            linewidths=0.5,
            linecolor="lightgray",
        )
        ax1.set_title(f"Stage 2: Hard Routing\nLayer {layer}", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Image Stimuli", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Text Stimuli", fontsize=11, fontweight="bold")

        # Plot Stage 3 with its own scale
        sns.heatmap(
            stage3_cross,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=stage3_vmin,
            vmax=stage3_vmax,
            xticklabels=img_labels,
            yticklabels=txt_labels,
            ax=ax2,
            cbar_kws={"label": "Cosine Similarity (Stage 3)"},
            square=True,
            linewidths=0.5,
            linecolor="lightgray",
        )
        ax2.set_title(
            f"Stage 3: Soft Routing (T={self.temperature})\nLayer {layer}",
            fontsize=12,
            fontweight="bold",
        )
        ax2.set_xlabel("Image Stimuli", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Text Stimuli", fontsize=11, fontweight="bold")

        # Overall title
        fig.suptitle(
            f"Cross-Modal Similarity Comparison: Stage 2 vs Stage 3 (Layer {layer})",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save
        plot_path = os.path.join(output_dir, f"comparison_cross_modal_layer{layer}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"   Saved comparison plot to: {plot_path}")

    def run_analysis(
        self,
        manifest_file: str,
        layers: list[int],
        pooling: str,
        output_dir: str,
        stage2_checkpoint: str,
        stage3_checkpoint: str,
    ):
        """
        Run complete compositional case study analysis.

        For each layer:
        1. Extract Stage 2 representations → compute matrix → save/visualise
        2. Extract Stage 3 representations → compute matrix → save/visualise
        3. Create comparison visualisation

        Args:
            manifest_file: Path to stimulus manifest JSON
            layers: List of layer indices to analyse
            pooling: Pooling strategy (mean)
            output_dir: Output directory
            stage2_checkpoint: Path to Stage 2 checkpoint
            stage3_checkpoint: Path to Stage 3 checkpoint
        """
        logger.info("=" * 80)
        logger.info("COMPOSITIONAL CASE STUDY: Stage 2 vs Stage 3 Representation Analysis")
        logger.info("=" * 80)

        # Load stimuli
        stimuli = self.load_manifest(manifest_file)
        num_stimuli = len(stimuli)

        logger.info("\nAnalysis Configuration:")
        logger.info(f"   Number of stimuli: {num_stimuli}")
        logger.info(f"   Matrix size: {2 * num_stimuli}×{2 * num_stimuli} (vision + text)")
        logger.info(f"   Layers: {layers}")
        logger.info(f"   Pooling: {pooling}")
        logger.info(f"   Output directory: {output_dir}")
        logger.info("=" * 80)

        # Initialize stage analyzers
        self.initialize_stage_analyzers(stage2_checkpoint, stage3_checkpoint)

        # Process each layer
        for layer_idx, layer in enumerate(layers):
            logger.info(f"\n{'=' * 80}")
            logger.info(f"LAYER {layer} ({layer_idx + 1}/{len(layers)})")
            logger.info(f"{'=' * 80}")

            # ============================================================
            # STAGE 2 ANALYSIS
            # ============================================================
            logger.info("\nSTAGE 2 ANALYSIS")
            logger.info(f"{'─' * 80}")

            # Extract representations
            stage2_reps, stage2_labels = self.extract_stimulus_representations(
                stimuli=stimuli,
                layer=layer,
                pooling=pooling,
                analyzer=self.stage2_analyzer,
                stage_name="Stage 2",
            )

            # Compute similarity matrix
            stage2_matrix = self.compute_similarity_matrix(stage2_reps, "Stage 2")

            # Save results
            logger.info("\nSaving Stage 2 results...")
            self.save_results(stage2_matrix, stage2_labels, output_dir, "stage2", layer)

            # Visualise
            logger.info("\nCreating Stage 2 visualization...")
            self.visualize_matrix(
                stage2_matrix, stage2_labels, output_dir, "stage2", layer, self.temperature
            )

            # ============================================================
            # STAGE 3 ANALYSIS
            # ============================================================
            logger.info("\nSTAGE 3 ANALYSIS")
            logger.info(f"{'─' * 80}")

            # Extract representations
            stage3_reps, stage3_labels = self.extract_stimulus_representations(
                stimuli=stimuli,
                layer=layer,
                pooling=pooling,
                analyzer=self.stage3_analyzer,
                stage_name="Stage 3",
            )

            # Compute similarity matrix
            stage3_matrix = self.compute_similarity_matrix(stage3_reps, "Stage 3")

            # Save results
            logger.info("\nSaving Stage 3 results...")
            self.save_results(stage3_matrix, stage3_labels, output_dir, "stage3", layer)

            # Visualise
            logger.info("\nCreating Stage 3 visualization...")
            self.visualize_matrix(
                stage3_matrix, stage3_labels, output_dir, "stage3", layer, self.temperature
            )

            # ============================================================
            # COMPARISON VISUALISATION
            # ============================================================
            logger.info("\nCOMPARISON ANALYSIS")
            logger.info(f"{'─' * 80}")

            self.visualize_cross_modal_comparison(
                stage2_matrix, stage3_matrix, stage2_labels, output_dir, layer
            )

            logger.info(f"\nLayer {layer} analysis complete!")

        logger.info(f"\n{'=' * 80}")
        logger.info("ANALYSIS COMPLETE!")
        logger.info(f"   Processed {len(layers)} layers")
        logger.info(f"   Generated {len(layers) * 5} output files per layer:")
        logger.info("      - 2 matrices (stage2, stage3)")
        logger.info("      - 2 labels (stage2, stage3)")
        logger.info("      - 2 heatmaps (stage2, stage3)")
        logger.info("      - 1 comparison plot")
        logger.info(f"   Results saved to: {output_dir}")
        logger.info("=" * 80)


def load_config(config_file: str) -> dict:
    """Load the JSON analysis config for the compositional case study."""
    return load_analysis_config(
        config_file,
        required_fields=["manifest_file", "stage2_checkpoint", "stage3_checkpoint"],
        defaults={
            "layers": [0, 16, 31],
            "pooling": "mean",
            "temperature": 0.01,
            "output_dir": "results/compositional_case_study/",
        },
    )


def main():
    """Main entry point for compositional case study analysis."""
    parser = argparse.ArgumentParser(
        description="Compositional Case Study: Stage 2 vs Stage 3 Representation Analysis"
    )
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Path to JSON config file with analysis parameters",
    )
    parser.add_argument(
        "--training-config",
        type=str,
        default=None,
        help="Path to training config (default: $MOE_CONFIG, else configs/training_config.yaml)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (cuda/cpu)",
    )

    args = parser.parse_args()

    # Load config
    logger.info(f"Loading configuration from: {args.config_file}")
    config = load_config(args.config_file)

    # Print configuration
    logger.info("\nConfiguration:")
    logger.info(f"   Manifest file: {config['manifest_file']}")
    logger.info(f"   Layers: {config['layers']}")
    logger.info(f"   Pooling: {config['pooling']}")
    logger.info(f"   Temperature: {config['temperature']}")
    logger.info(f"   Output directory: {config['output_dir']}")
    logger.info(f"   Stage 2 checkpoint: {config['stage2_checkpoint']}")
    logger.info(f"   Stage 3 checkpoint: {config['stage3_checkpoint']}")

    # Initialize analyzer
    analyzer = CompositionalCaseStudyAnalyzer(
        config_path=args.training_config, device=args.device, temperature=config["temperature"]
    )

    # Run analysis
    analyzer.run_analysis(
        manifest_file=config["manifest_file"],
        layers=config["layers"],
        pooling=config["pooling"],
        output_dir=config["output_dir"],
        stage2_checkpoint=config["stage2_checkpoint"],
        stage3_checkpoint=config["stage3_checkpoint"],
    )


if __name__ == "__main__":
    setup_logging()
    main()
