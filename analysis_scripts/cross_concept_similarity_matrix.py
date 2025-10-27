"""
Cross-Concept Similarity Matrix Analysis for MoE Vision-Language Models

This script computes 2N×2N similarity matrices comparing N image-text pairs at specified layers.
Uses expert routing to force images through vision expert and text through text expert.

Usage:
    python analysis_scripts/cross_concept_similarity_matrix.py \\
        --config-file experiments/similarity_config.json

Config file format (JSON):
    {
      "image_text_pairs": [
        {"image": "path/to/image1.jpg", "text": "description one"},
        {"image": "path/to/image2.jpg", "text": "description two"}
      ],
      "layers": [31],
      "pooling": "mean",
      "output_dir": "results/similarity_matrix/"
    }
"""

import torch
import numpy as np
import json
import os
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns

from cross_modality_purity import CrossModalityPurityAnalyzer


class CrossConceptSimilarityAnalyzer:
    """
    Analyzer for computing cross-concept similarity matrices.
    
    Reuses core functionality from CrossModalityPurityAnalyzer for model loading,
    expert routing, and representation extraction.
    """
    
    def __init__(
        self,
        config_path: str = "configs/training_config.yaml",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize analyzer by creating base analyzer for model access.
        
        Args:
            config_path: Path to training configuration
            device: Device to run on (cuda/cpu)
        """
        print(f"🔧 Initializing Cross-Concept Similarity Analyzer on {device}")
        
        # Initialize base analyzer to reuse model loading and extraction methods
        self.base_analyzer = CrossModalityPurityAnalyzer(
            config_path=config_path,
            device=device
        )
        
        # Quick access to base analyzer's attributes
        self.device = device
    
    def load_models(self):
        """Load all required models via base analyzer."""
        self.base_analyzer.load_models()
    
    def _extract_representation(
        self,
        concept: str,
        expert: str,
        layer: int,
        modality: str,
        pooling: str = "mean"
    ) -> np.ndarray:
        """
        Extract hidden state representation for a concept at a specific layer.
        
        This is a wrapper around the base analyzer's representation extraction logic.
        
        Args:
            concept: Image path (for vision) or text string (for text)
            expert: "vision" or "text" - which expert to route through
            layer: Layer index to extract from (0-31)
            modality: "vision" or "text" - type of input
            pooling: "mean" for mean-pooling (default)
        
        Returns:
            Numpy array of shape [hidden_dim] representing the pooled hidden state
        """
        # Delegate to base analyzer's analyze_representation method
        return self.base_analyzer.analyze_representation(
            concept=concept,
            expert=expert,
            layer=layer,
            modality=modality,
            pooling=pooling
        )
    
    def _compute_cosine_similarity_matrix(
        self,
        representations: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix for a list of representations.
        
        Args:
            representations: List of N representations, each of shape [hidden_dim]
        
        Returns:
            N×N matrix where matrix[i,j] = cosine_similarity(rep_i, rep_j)
        """
        n = len(representations)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    # Cosine similarity: dot(a,b) / (norm(a) * norm(b))
                    cos_sim = np.dot(representations[i], representations[j]) / (
                        np.linalg.norm(representations[i]) * 
                        np.linalg.norm(representations[j]) + 1e-8
                    )
                    matrix[i, j] = float(cos_sim)
        
        return matrix
    
    def compute_cross_concept_matrix(
        self,
        image_text_pairs: List[Dict[str, str]],
        layer: int = 31,
        pooling: str = "mean"
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute 2N×2N cross-concept similarity matrix.
        
        Args:
            image_text_pairs: List of dicts with "image" (path) and "text" (string) keys
            layer: Layer to extract representations from (default: 31)
            pooling: Pooling strategy (default: "mean")
        
        Returns:
            Tuple of:
                - matrix: 2N×2N numpy array of cosine similarities
                - labels: List of 2N labels for rows/columns
        
        Matrix structure:
            [img1, img2, ..., imgN, txt1, txt2, ..., txtN]
        """
        N = len(image_text_pairs)
        print(f"\n🔬 Computing {2*N}×{2*N} similarity matrix at layer {layer}")
        print(f"   Pooling strategy: {pooling}")
        print(f"   Number of image-text pairs: {N}")
        
        representations = []
        labels = []
        
        # Extract image representations (first N entries)
        print(f"\n📸 Extracting image representations through vision expert...")
        for idx, pair in enumerate(image_text_pairs):
            image_path = pair["image"]
            image_name = Path(image_path).stem
            
            print(f"   [{idx+1}/{N}] Processing image: {image_name}")
            
            img_rep = self._extract_representation(
                concept=image_path,
                expert="vision",
                layer=layer,
                modality="vision",
                pooling=pooling
            )
            
            representations.append(img_rep)
            labels.append(f"img:{image_name}")
            
            print(f"       ✓ Extracted representation: shape={img_rep.shape}, norm={np.linalg.norm(img_rep):.2f}")
        
        # Extract text representations (next N entries)
        print(f"\n💬 Extracting text representations through text expert...")
        for idx, pair in enumerate(image_text_pairs):
            text = pair["text"]
            
            print(f"   [{idx+1}/{N}] Processing text: '{text}'")
            
            txt_rep = self._extract_representation(
                concept=text,
                expert="text",
                layer=layer,
                modality="text",
                pooling=pooling
            )
            
            representations.append(txt_rep)
            labels.append(f"txt:{text}")
            
            print(f"       ✓ Extracted representation: shape={txt_rep.shape}, norm={np.linalg.norm(txt_rep):.2f}")
        
        # Compute pairwise similarity matrix
        print(f"\n📊 Computing {2*N}×{2*N} cosine similarity matrix...")
        matrix = self._compute_cosine_similarity_matrix(representations)
        
        print(f"   ✓ Matrix computed: shape={matrix.shape}")
        print(f"   ✓ Similarity range: [{matrix.min():.3f}, {matrix.max():.3f}]")
        
        return matrix, labels
    
    def save_results(
        self,
        matrix: np.ndarray,
        labels: List[str],
        output_dir: str,
        layer: int
    ):
        """
        Save similarity matrix and labels to JSON files.
        
        Args:
            matrix: 2N×2N similarity matrix
            labels: List of row/column labels
            output_dir: Directory to save results
            layer: Layer number (for filename)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save matrix as JSON (convert to list for JSON serialization)
        matrix_path = os.path.join(output_dir, f"similarity_matrix_layer{layer}.json")
        with open(matrix_path, "w") as f:
            json.dump({
                "matrix": matrix.tolist(),
                "layer": layer,
                "shape": list(matrix.shape)
            }, f, indent=2)
        print(f"   ✓ Saved matrix to: {matrix_path}")
        
        # Save labels as JSON
        labels_path = os.path.join(output_dir, f"labels_layer{layer}.json")
        with open(labels_path, "w") as f:
            json.dump({
                "labels": labels,
                "layer": layer,
                "num_pairs": len(labels) // 2
            }, f, indent=2)
        print(f"   ✓ Saved labels to: {labels_path}")
    
    def visualize_matrix(
        self,
        matrix: np.ndarray,
        labels: List[str],
        output_dir: str,
        layer: int
    ):
        """
        Create and save heatmap visualization of similarity matrix.
        
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
            cbar_kws={'label': 'Cosine Similarity'},
            square=True,
            linewidths=0.5,
            linecolor='lightgray'
        )
        
        # Customize labels
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(labels, rotation=0, fontsize=9)
        
        # Add title
        ax.set_title(
            f'Cross-Concept Similarity Matrix (Layer {layer})\n'
            f'{n//2} Image-Text Pairs | Mean-Pooled Representations',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f"similarity_matrix_layer{layer}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved heatmap to: {plot_path}")
    
    def run_analysis(
        self,
        image_text_pairs: List[Dict[str, str]],
        layers: List[int] = [31],
        pooling: str = "mean",
        output_dir: str = "results/similarity_matrix/"
    ) -> Dict:
        """
        Run complete similarity matrix analysis for specified layers.
        
        Args:
            image_text_pairs: List of image-text pair dicts
            layers: List of layer indices to analyze
            pooling: Pooling strategy
            output_dir: Output directory for results
        
        Returns:
            Dictionary containing results for each layer
        """
        print("=" * 80)
        print("Cross-Concept Similarity Matrix Analysis")
        print("=" * 80)
        
        results = {}
        
        for layer in layers:
            print(f"\n{'='*80}")
            print(f"LAYER {layer}")
            print(f"{'='*80}")
            
            # Compute similarity matrix
            matrix, labels = self.compute_cross_concept_matrix(
                image_text_pairs=image_text_pairs,
                layer=layer,
                pooling=pooling
            )
            
            # Save results
            print(f"\n💾 Saving results...")
            self.save_results(matrix, labels, output_dir, layer)
            
            # Visualize
            print(f"\n📈 Generating visualization...")
            self.visualize_matrix(matrix, labels, output_dir, layer)
            
            results[f"layer_{layer}"] = {
                "matrix": matrix,
                "labels": labels
            }
        
        print(f"\n{'='*80}")
        print("✅ Analysis complete!")
        print(f"📁 Results saved to: {output_dir}")
        print("=" * 80)
        
        return results


def load_config(config_file: str) -> Dict:
    """
    Load configuration from JSON file.
    
    Args:
        config_file: Path to JSON config file
    
    Returns:
        Dictionary with configuration parameters
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Validate required fields
    required_fields = ["image_text_pairs"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Config file missing required field: {field}")
    
    # Set defaults for optional fields
    config.setdefault("layers", [31])
    config.setdefault("pooling", "mean")
    config.setdefault("output_dir", "results/similarity_matrix/")
    
    return config


def main():
    """Main entry point for cross-concept similarity matrix analysis."""
    parser = argparse.ArgumentParser(
        description="Cross-Concept Similarity Matrix Analysis for MoE VLM"
    )
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Path to JSON config file with image-text pairs and parameters"
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        help="Layer indices to analyze (overrides config file)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (overrides config file)"
    )
    parser.add_argument(
        "--training-config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training config file for model paths"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # Load config file
    print(f"📄 Loading configuration from: {args.config_file}")
    config = load_config(args.config_file)
    
    # Override with command-line arguments if provided
    if args.layers:
        config["layers"] = args.layers
        print(f"   ⚠️  Overriding layers from command line: {args.layers}")
    
    if args.output_dir:
        config["output_dir"] = args.output_dir
        print(f"   ⚠️  Overriding output directory from command line: {args.output_dir}")
    
    # Print configuration
    print(f"\n📋 Configuration:")
    print(f"   Image-text pairs: {len(config['image_text_pairs'])}")
    print(f"   Layers: {config['layers']}")
    print(f"   Pooling: {config['pooling']}")
    print(f"   Output directory: {config['output_dir']}")
    
    # Initialize analyzer
    analyzer = CrossConceptSimilarityAnalyzer(
        config_path=args.training_config,
        device=args.device
    )
    
    # Load models
    analyzer.load_models()
    
    # Run analysis
    results = analyzer.run_analysis(
        image_text_pairs=config["image_text_pairs"],
        layers=config["layers"],
        pooling=config["pooling"],
        output_dir=config["output_dir"]
    )


if __name__ == "__main__":
    main()
