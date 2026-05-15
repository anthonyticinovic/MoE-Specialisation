"""
Compositional Case Study: Stage 2 vs Stage 3 Representation Analysis

This script compares how Stage 2 (hard routing) vs Stage 3 (soft routing) represent
objects with different attributes (e.g., color, shape, size) using similarity matrices.

For N stimuli, computes 2N×2N similarity matrices (vision + text representations):
- Stage 2: Forced routing (vision→expert0, text→expert1)
- Stage 3: Learned soft routing (natural model behavior)

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
        {"id": "green_apple", "image_path": "data/images/green_apple.jpg", "caption": "A green apple"},
        ...
      ]
    }
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

# REUSE: Import existing analyzer class for model loading and representation extraction
from analysis_scripts.cross_concept_similarity_matrix import CrossConceptSimilarityAnalyzer


class CompositionalCaseStudyAnalyzer:
    """
    Analyzer for compositional representation case study comparing Stage 2 vs Stage 3.

    Heavily reuses CrossConceptSimilarityAnalyzer for:
    - Model loading (Stage 2 and Stage 3)
    - Representation extraction (vision and text, mean pooling)
    - Similarity matrix computation
    - Visualization

    Key differences from CrossConceptSimilarityAnalyzer:
    - Uses pre-specified stimuli from JSON manifest (not COCO sampling)
    - Runs both Stage 2 AND Stage 3 in single script
    - Compares representations across stages
    """

    def __init__(
        self,
        config_path: str = "configs/training_config.yaml",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        temperature: float = 0.01,
    ):
        """
        Initialize analyzer.

        Args:
            config_path: Path to training configuration YAML
            device: Device to run on (cuda/cpu)
            temperature: Routing temperature for Stage 3
        """
        print("🔧 Initializing Compositional Case Study Analyzer")
        print(f"   Device: {device}")
        print(f"   Temperature: {temperature}")

        self.config_path = config_path
        self.device = device
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
        print(f"\n📄 Loading stimulus manifest from: {manifest_file}")

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

        print(f"   ✓ Loaded {len(stimuli)} stimuli:")
        for stimulus in stimuli:
            print(f"      - {stimulus['id']}: {stimulus['caption']}")

        return stimuli

    def initialize_stage_analyzers(self, stage2_checkpoint: str, stage3_checkpoint: str):
        """
        Initialize separate analyzers for Stage 2 and Stage 3.

        REUSES: CrossConceptSimilarityAnalyzer class for both stages

        Args:
            stage2_checkpoint: Path to Stage 2 checkpoint
            stage3_checkpoint: Path to Stage 3 checkpoint
        """
        print("\n🔧 Initializing Stage 2 analyzer...")
        self.stage2_analyzer = CrossConceptSimilarityAnalyzer(
            config_path=self.config_path,
            device=self.device,
            mode="stage2",
            stage2_checkpoint=stage2_checkpoint,
            temperature=self.temperature,
        )
        self.stage2_analyzer.load_models()
        print("   ✓ Stage 2 analyzer ready")

        print("\n🔧 Initializing Stage 3 analyzer...")
        self.stage3_analyzer = CrossConceptSimilarityAnalyzer(
            config_path=self.config_path,
            device=self.device,
            mode="stage3",
            stage3_checkpoint=stage3_checkpoint,
            temperature=self.temperature,
        )
        self.stage3_analyzer.load_models()
        print("   ✓ Stage 3 analyzer ready")

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
        print(f"\n📊 Extracting {stage_name} representations at layer {layer}")
        print(f"   Stimuli: {N}, Total entries: {2 * N} (vision + text)")

        representations = []
        labels = []

        # Extract vision representations (first N entries)
        print("\n📸 Extracting vision representations...")
        for stimulus in stimuli:
            print(f"   Processing: {stimulus['id']}")

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
            print(f"      ✓ Vision: norm={np.linalg.norm(img_rep):.2f}")

        # Extract text representations (next N entries)
        print("\n💬 Extracting text representations...")
        for stimulus in stimuli:
            print(f"   Processing: {stimulus['id']}")

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
            print(f"      ✓ Text: norm={np.linalg.norm(txt_rep):.2f}")

        print(f"\n   ✓ Extracted {2 * N} representations ({N} vision + {N} text)")
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
        print(f"\n📊 Computing {stage_name} similarity matrix ({n}×{n})...")

        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    # REUSE: Same cosine similarity formula
                    cos_sim = np.dot(representations[i], representations[j]) / (
                        np.linalg.norm(representations[i]) * np.linalg.norm(representations[j])
                        + 1e-8
                    )
                    matrix[i, j] = float(cos_sim)

        print(f"   ✓ Matrix computed: shape={matrix.shape}")
        print(f"   ✓ Similarity range: [{matrix.min():.3f}, {matrix.max():.3f}]")
        print(f"   ✓ Mean similarity: {matrix.mean():.3f}")

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
        print(f"   ✓ Saved matrix to: {matrix_path}")

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
        print(f"   ✓ Saved labels to: {labels_path}")

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
        Create heatmap visualization of similarity matrix.

        REUSES: Similar visualization style as CrossConceptSimilarityAnalyzer

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

        print(f"   ✓ Saved heatmap to: {plot_path}")

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
        img_labels = [l.replace("img:", "") for l in labels[:half_n]]
        txt_labels = [l.replace("txt:", "") for l in labels[half_n:]]

        print("\n📊 Creating cross-modal comparison visualization...")
        print(f"   Stage 2 cross-modal range: [{stage2_cross.min():.3f}, {stage2_cross.max():.3f}]")
        print(f"   Stage 3 cross-modal range: [{stage3_cross.min():.3f}, {stage3_cross.max():.3f}]")

        # Create side-by-side figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

        # Compute INDEPENDENT color scales for maximum sensitivity per stage
        stage2_vmin = stage2_cross.min() - 0.005
        stage2_vmax = stage2_cross.max() + 0.005
        stage3_vmin = stage3_cross.min() - 0.005
        stage3_vmax = stage3_cross.max() + 0.005

        print(f"   Stage 2 color scale: [{stage2_vmin:.3f}, {stage2_vmax:.3f}]")
        print(f"   Stage 3 color scale: [{stage3_vmin:.3f}, {stage3_vmax:.3f}]")

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

        print(f"   ✓ Saved comparison plot to: {plot_path}")

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
        1. Extract Stage 2 representations → compute matrix → save/visualize
        2. Extract Stage 3 representations → compute matrix → save/visualize
        3. Create comparison visualization

        Args:
            manifest_file: Path to stimulus manifest JSON
            layers: List of layer indices to analyze
            pooling: Pooling strategy (mean)
            output_dir: Output directory
            stage2_checkpoint: Path to Stage 2 checkpoint
            stage3_checkpoint: Path to Stage 3 checkpoint
        """
        print("=" * 80)
        print("COMPOSITIONAL CASE STUDY: Stage 2 vs Stage 3 Representation Analysis")
        print("=" * 80)

        # Load stimuli
        stimuli = self.load_manifest(manifest_file)
        num_stimuli = len(stimuli)

        print("\n📋 Analysis Configuration:")
        print(f"   Number of stimuli: {num_stimuli}")
        print(f"   Matrix size: {2 * num_stimuli}×{2 * num_stimuli} (vision + text)")
        print(f"   Layers: {layers}")
        print(f"   Pooling: {pooling}")
        print(f"   Output directory: {output_dir}")
        print("=" * 80)

        # Initialize stage analyzers
        self.initialize_stage_analyzers(stage2_checkpoint, stage3_checkpoint)

        # Process each layer
        for layer_idx, layer in enumerate(layers):
            print(f"\n{'=' * 80}")
            print(f"LAYER {layer} ({layer_idx + 1}/{len(layers)})")
            print(f"{'=' * 80}")

            # ============================================================
            # STAGE 2 ANALYSIS
            # ============================================================
            print("\n🔵 STAGE 2 ANALYSIS")
            print(f"{'─' * 80}")

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
            print("\n💾 Saving Stage 2 results...")
            self.save_results(stage2_matrix, stage2_labels, output_dir, "stage2", layer)

            # Visualize
            print("\n📈 Creating Stage 2 visualization...")
            self.visualize_matrix(
                stage2_matrix, stage2_labels, output_dir, "stage2", layer, self.temperature
            )

            # ============================================================
            # STAGE 3 ANALYSIS
            # ============================================================
            print("\n🟢 STAGE 3 ANALYSIS")
            print(f"{'─' * 80}")

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
            print("\n💾 Saving Stage 3 results...")
            self.save_results(stage3_matrix, stage3_labels, output_dir, "stage3", layer)

            # Visualize
            print("\n📈 Creating Stage 3 visualization...")
            self.visualize_matrix(
                stage3_matrix, stage3_labels, output_dir, "stage3", layer, self.temperature
            )

            # ============================================================
            # COMPARISON VISUALIZATION
            # ============================================================
            print("\n🔀 COMPARISON ANALYSIS")
            print(f"{'─' * 80}")

            self.visualize_cross_modal_comparison(
                stage2_matrix, stage3_matrix, stage2_labels, output_dir, layer
            )

            print(f"\n✅ Layer {layer} analysis complete!")

        print(f"\n{'=' * 80}")
        print("✅ ANALYSIS COMPLETE!")
        print(f"   Processed {len(layers)} layers")
        print(f"   Generated {len(layers) * 5} output files per layer:")
        print("      - 2 matrices (stage2, stage3)")
        print("      - 2 labels (stage2, stage3)")
        print("      - 2 heatmaps (stage2, stage3)")
        print("      - 1 comparison plot")
        print(f"   📁 Results saved to: {output_dir}")
        print("=" * 80)


def load_config(config_file: str) -> dict:
    """
    Load configuration from JSON file.

    Args:
        config_file: Path to JSON config

    Returns:
        Dictionary with configuration parameters
    """
    with open(config_file) as f:
        config = json.load(f)

    # Validate required fields
    required_fields = ["manifest_file", "stage2_checkpoint", "stage3_checkpoint"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Config file missing required field: {field}")

    # Set defaults for optional fields
    config.setdefault("layers", [0, 16, 31])
    config.setdefault("pooling", "mean")
    config.setdefault("temperature", 0.01)
    config.setdefault("output_dir", "results/compositional_case_study/")

    return config


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
        default="configs/training_config.yaml",
        help="Path to training config file for model paths",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda/cpu)",
    )

    args = parser.parse_args()

    # Load config
    print(f"📄 Loading configuration from: {args.config_file}")
    config = load_config(args.config_file)

    # Print configuration
    print("\n📋 Configuration:")
    print(f"   Manifest file: {config['manifest_file']}")
    print(f"   Layers: {config['layers']}")
    print(f"   Pooling: {config['pooling']}")
    print(f"   Temperature: {config['temperature']}")
    print(f"   Output directory: {config['output_dir']}")
    print(f"   Stage 2 checkpoint: {config['stage2_checkpoint']}")
    print(f"   Stage 3 checkpoint: {config['stage3_checkpoint']}")

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
    main()
