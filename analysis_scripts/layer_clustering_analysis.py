"""
MoE Layer Clustering Analysis

Analyzes representation clustering at specific MoE layers to understand
whether experts specialize by modality and/or concept.

This script visualizes the representation space created by MoE layers to see
if the router and experts are separating inputs by modality or concept.

Usage:
    # Analyze using config file
    python analysis_scripts/layer_clustering_analysis.py \
        --config configs/clustering_analysis.json
    
    # Override config parameters
    python analysis_scripts/layer_clu        ax.set_xlabel("Dimension 1", fontsize=12)
        ax.set_ylabel("Dimension 2", fontsize=12)
        ax.set_title(f"Layer {layer_idx} - Expert Selection with Routing Confidence", fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', framealpha=0.9, title="Expert Choice")
        ax.grid(True, alpha=0.3)
        
        # Add colorbar for confidence
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        # Colorbar is just a reference for the confidence gradient
        
        # Create a colormap that goes from white to gray (for visualization reference)
        # This represents the confidence gradient
        # Use the actual confidence threshold from the config
        norm = Normalize(vmin=self.expert_confidence_threshold, vmax=1.0)
        sm = ScalarMappable(cmap=plt.cm.Greys, norm=norm)
        sm.set_array([])
        
        cbar = plt.colorbar(sm, ax=ax, pad=0.02, aspect=30)
        cbar.set_label('Routing Confidence', rotation=270, labelpad=20, fontsize=11) \
        --config configs/clustering_analysis.json \
        --layers 31 \
        --reduction-method tsne
"""

import argparse
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score

# Import base analyzer
from analysis_scripts.cross_modality_purity import CrossModalityPurityAnalyzer


class MoEClusteringAnalyzer(CrossModalityPurityAnalyzer):
    """
    Extends CrossModalityPurityAnalyzer to add clustering analysis capabilities.

    Analyzes representation clustering to understand modality and concept specialization
    in MoE layers.
    """

    def __init__(self, config_path: str = "configs/training_config.yaml", device: str = "cuda"):
        super().__init__(config_path, device)
        self.expert_confidence_threshold = 0.6  # Default threshold

    def extract_concept_samples(
        self, annotations_file: str, concepts: list[str], samples_per_concept: int, seed: int = 42
    ) -> dict[str, list[dict]]:
        """
        Extract balanced samples from COCO annotations for specified concepts.

        Args:
            annotations_file: Path to COCO annotations JSON
            concepts: List of concept keywords (e.g., ["cat", "dog", "car"])
            samples_per_concept: Target number of samples per concept
            seed: Random seed for reproducibility

        Returns:
            Dict mapping concept -> list of sample dicts with keys:
                - 'image_id': COCO image ID
                - 'caption': Image caption
                - 'image_path': Full path to image file
        """
        print("📚 Extracting concept samples from COCO annotations...")
        print(f"   Concepts: {concepts}")
        print(f"   Target: {samples_per_concept} samples per concept")

        # Load COCO annotations
        with open(annotations_file) as f:
            coco_data = json.load(f)

        # Build image_id -> image_path mapping
        image_id_to_path = {}
        for img in coco_data["images"]:
            image_id_to_path[img["id"]] = img["file_name"]

        # Set random seed
        np.random.seed(seed)

        # Extract samples for each concept
        concept_samples = {concept: [] for concept in concepts}

        for annotation in coco_data["annotations"]:
            caption = annotation["caption"].lower()
            image_id = annotation["image_id"]

            # Check which concepts appear in this caption
            matching_concepts = [c for c in concepts if c.lower() in caption.split()]

            # Skip if multiple specified concepts appear (ambiguous)
            if len(matching_concepts) > 1:
                continue

            # Skip if no concepts match
            if len(matching_concepts) == 0:
                continue

            # Add to the matching concept's sample list
            concept = matching_concepts[0]
            if len(concept_samples[concept]) < samples_per_concept:
                concept_samples[concept].append(
                    {
                        "image_id": image_id,
                        "caption": annotation["caption"],
                        "image_path": image_id_to_path[image_id],
                        "concept": concept,
                    }
                )

        # Print statistics
        print("\n   📊 Extracted samples:")
        for concept, samples in concept_samples.items():
            print(f"      {concept}: {len(samples)} samples")

        # Warn if any concept is under-sampled
        for concept, samples in concept_samples.items():
            if len(samples) < samples_per_concept:
                print(
                    f"   ⚠️  Warning: Only found {len(samples)} samples for '{concept}' (target: {samples_per_concept})"
                )

        return concept_samples

    def _extract_expert_choice(
        self, routing_probs: torch.Tensor, confidence_threshold: float = 0.6
    ) -> str:
        """
        Extract expert choice from routing probabilities using majority voting.

        Args:
            routing_probs: Routing probabilities [num_tokens, num_experts]
            confidence_threshold: Minimum fraction of votes needed for decisive label

        Returns:
            Expert label: "Expert 0", "Expert 1", or "mixed"
        """
        # Handle edge case: no tokens
        if routing_probs.shape[0] == 0:
            return "unknown"

        # Get argmax expert for each token
        expert_choices = routing_probs.argmax(dim=1).cpu().numpy()  # [num_tokens]

        # Count votes
        votes = Counter(expert_choices)
        total_votes = len(expert_choices)

        # Handle edge case: no votes (shouldn't happen but safety check)
        if total_votes == 0 or len(votes) == 0:
            return "unknown"

        # Find winner
        winner_expert = votes.most_common(1)[0][0]
        winner_count = votes[winner_expert]
        winner_fraction = winner_count / total_votes

        # Check if winner meets confidence threshold
        if winner_fraction >= confidence_threshold:
            return f"Expert {winner_expert}"
        else:
            return "mixed"

    def collect_representations(
        self,
        concept_samples: dict[str, list[dict]],
        image_dir: str,
        target_layers: list[int],
        pooling: str = "mean",
        confidence_threshold: float = 0.6,
    ) -> dict[int, pd.DataFrame]:
        """
        Collect hidden state representations from concept samples.

        Args:
            concept_samples: Dict mapping concept -> list of samples
            image_dir: Base directory for COCO images
            target_layers: List of layer indices to analyze
            pooling: Pooling strategy ("mean", "max", "cls", "last")
            confidence_threshold: Threshold for expert choice confidence

        Returns:
            Dict mapping layer_idx -> DataFrame with columns:
                - 'representation': np.ndarray of hidden state vector
                - 'concept': str (e.g., "cat", "dog", "car")
                - 'modality': str ("vision" or "text")
                - 'expert_choice': str ("Expert 0", "Expert 1", or "mixed")
                - 'routing_confidence': float (winner's vote fraction)
                - 'image_id': int (COCO image ID)
                - 'caption': str (COCO caption)
        """
        print(
            f"\n🔬 Collecting representations from {sum(len(s) for s in concept_samples.values())} samples..."
        )
        print(f"   Target layers: {target_layers}")
        print(f"   Pooling: {pooling}")
        print(f"   Expert confidence threshold: {confidence_threshold}")

        self.expert_confidence_threshold = confidence_threshold

        # Initialize dataframes for each layer
        layer_dataframes = {layer_idx: [] for layer_idx in target_layers}

        # Process each concept's samples
        total_samples = sum(len(samples) for samples in concept_samples.values())
        sample_count = 0

        for concept, samples in concept_samples.items():
            print(f"\n   Processing concept: {concept} ({len(samples)} samples)")

            for sample in samples:
                sample_count += 1
                image_path = os.path.join(image_dir, sample["image_path"])
                caption = sample["caption"]
                image_id = sample["image_id"]

                if sample_count % 10 == 0:
                    print(f"      Progress: {sample_count}/{total_samples} samples")

                # Prepare vision and text inputs
                try:
                    visual_tokens = self._prepare_vision_input(image_path)
                    text_embeddings = self._prepare_text_input(caption)
                except Exception as e:
                    print(f"      ⚠️  Skipping sample {image_id}: {e}")
                    continue

                # CRITICAL: Combine vision and text tokens for a single forward pass
                # This ensures _last_router_logits contains routing info for BOTH modalities
                num_vision_tokens = visual_tokens.shape[1]
                combined_embeddings = torch.cat([visual_tokens, text_embeddings], dim=1)

                # Extract all layer states with combined input (like training)
                combined_layer_states = self._extract_all_layer_states(
                    combined_embeddings,
                    routing_mask=None,  # Stage 3 uses learned soft routing
                )

                # For each target layer, collect representations
                for layer_idx in target_layers:
                    # Layer states: [embedding_output, layer_0, ..., layer_31]
                    # So layer_idx 0 is at index 1, layer_idx 31 is at index 32
                    combined_hidden = combined_layer_states[
                        layer_idx + 1
                    ]  # [1, 257+text_len, hidden_dim]

                    # Split combined hidden states back into vision and text
                    vision_hidden = combined_hidden[
                        :, :num_vision_tokens, :
                    ]  # [1, 257, hidden_dim]
                    text_hidden = combined_hidden[
                        :, num_vision_tokens:, :
                    ]  # [1, text_len, hidden_dim]

                    # Get routing probabilities from the MoE layer
                    # Access the specific layer's MoE module
                    model_to_inspect = self.llm.module if hasattr(self.llm, "module") else self.llm
                    moe_layer = model_to_inspect.model.layers[layer_idx].mlp

                    # Get router logits (stored during forward pass)
                    if hasattr(moe_layer, "_last_router_logits"):
                        # Split router logits for vision and text portions
                        all_router_logits = (
                            moe_layer._last_router_logits
                        )  # [batch, total_seq_len, num_experts]

                        # Debug: Print shape and sample values for first sample only
                        if sample_count == 1 and layer_idx == target_layers[0]:
                            print(
                                f"      DEBUG: all_router_logits shape: {all_router_logits.shape}"
                            )
                            print(f"      DEBUG: num_vision_tokens: {num_vision_tokens}")
                            print(
                                f"      DEBUG: Vision router logits sample (first 3 tokens): {all_router_logits[0, :3, :]}"
                            )
                            print(
                                f"      DEBUG: Text router logits sample (first 3 tokens): {all_router_logits[0, num_vision_tokens : num_vision_tokens + 3, :]}"
                            )

                        vision_router_logits = all_router_logits[
                            :, :num_vision_tokens, :
                        ]  # [1, 257, 2]
                        text_router_logits = all_router_logits[
                            :, num_vision_tokens:, :
                        ]  # [1, text_len, 2]

                        # Convert to probabilities
                        vision_router_probs = torch.softmax(
                            vision_router_logits[0], dim=-1
                        )  # [257, 2]
                        text_router_probs = torch.softmax(
                            text_router_logits[0], dim=-1
                        )  # [text_len, 2]

                        # Debug: Print probability distributions
                        if sample_count == 1 and layer_idx == target_layers[0]:
                            print(
                                f"      DEBUG: Vision probs sample (first 3 tokens): {vision_router_probs[:3, :]}"
                            )
                            print(
                                f"      DEBUG: Text probs sample (first 3 tokens): {text_router_probs[:3, :]}"
                            )
                            vision_expert0_count = (
                                (vision_router_probs.argmax(dim=1) == 0).sum().item()
                            )
                            text_expert0_count = (text_router_probs.argmax(dim=1) == 0).sum().item()
                            print(
                                f"      DEBUG: Vision tokens → Expert 0: {vision_expert0_count}/{len(vision_router_probs)}"
                            )
                            print(
                                f"      DEBUG: Text tokens → Expert 0: {text_expert0_count}/{len(text_router_probs)}"
                            )

                        # Extract expert choices with majority voting
                        vision_expert_choice = self._extract_expert_choice(
                            vision_router_probs, confidence_threshold
                        )
                        text_expert_choice = self._extract_expert_choice(
                            text_router_probs, confidence_threshold
                        )

                        # Compute routing confidence (winner's vote fraction)
                        vision_votes = vision_router_probs.argmax(dim=1).cpu().numpy()
                        if len(vision_votes) > 0:
                            vision_vote_counts = Counter(vision_votes)
                            if len(vision_vote_counts) > 0:
                                vision_winner_count = vision_vote_counts.most_common(1)[0][1]
                                vision_confidence = vision_winner_count / len(vision_votes)
                            else:
                                vision_confidence = 0.0
                        else:
                            vision_confidence = 0.0

                        text_votes = text_router_probs.argmax(dim=1).cpu().numpy()
                        if len(text_votes) > 0:
                            text_vote_counts = Counter(text_votes)
                            if len(text_vote_counts) > 0:
                                text_winner_count = text_vote_counts.most_common(1)[0][1]
                                text_confidence = text_winner_count / len(text_votes)
                            else:
                                text_confidence = 0.0
                        else:
                            text_confidence = 0.0
                    else:
                        # Fallback if routing logits not available
                        vision_expert_choice = "unknown"
                        text_expert_choice = "unknown"
                        vision_confidence = 0.0
                        text_confidence = 0.0

                    # Pool representations
                    vision_repr = self._pool_representation(vision_hidden, pooling, "vision")
                    text_repr = self._pool_representation(text_hidden, pooling, "text")

                    # Add to dataframe
                    layer_dataframes[layer_idx].append(
                        {
                            "representation": vision_repr,
                            "concept": concept,
                            "modality": "vision",
                            "expert_choice": vision_expert_choice,
                            "routing_confidence": vision_confidence,
                            "image_id": image_id,
                            "caption": caption,
                        }
                    )

                    layer_dataframes[layer_idx].append(
                        {
                            "representation": text_repr,
                            "concept": concept,
                            "modality": "text",
                            "expert_choice": text_expert_choice,
                            "routing_confidence": text_confidence,
                            "image_id": image_id,
                            "caption": caption,
                        }
                    )

        # Convert to DataFrames
        for layer_idx in target_layers:
            layer_dataframes[layer_idx] = pd.DataFrame(layer_dataframes[layer_idx])

        print(f"\n✅ Collected {sample_count} samples across {len(target_layers)} layers")
        return layer_dataframes

    def run_dimensionality_reduction(
        self,
        representations: np.ndarray,
        method: str = "pacmap",
        n_neighbors: int = 15,
        perplexity: int = 30,
    ) -> np.ndarray:
        """
        Reduce high-dimensional representations to 2D for visualization.

        Args:
            representations: Array of shape [n_samples, hidden_dim]
            method: "tsne" or "pacmap"
            n_neighbors: Number of neighbors (for PaCMAP)
            perplexity: Perplexity (for t-SNE)

        Returns:
            2D coordinates array of shape [n_samples, 2]
        """
        print(f"   🔄 Running {method.upper()} dimensionality reduction...")
        print(f"      Input shape: {representations.shape}")

        if method.lower() == "tsne":
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1)
            coords_2d = reducer.fit_transform(representations)

        elif method.lower() == "pacmap":
            try:
                import pacmap

                reducer = pacmap.PaCMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
                coords_2d = reducer.fit_transform(representations)
            except ImportError:
                print("      ⚠️  PaCMAP not installed. Falling back to t-SNE...")
                reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                coords_2d = reducer.fit_transform(representations)

        else:
            raise ValueError(f"Unknown reduction method: {method}. Use 'tsne' or 'pacmap'")

        print(f"      Output shape: {coords_2d.shape}")
        return coords_2d

    def plot_clustering_analysis(
        self, layer_idx: int, df: pd.DataFrame, coords_2d: np.ndarray, output_dir: str
    ):
        """
        Generate 3 scatter plots with identical layout but different coloring.

        Args:
            layer_idx: Layer index being visualized
            df: DataFrame with metadata (concept, modality, expert_choice)
            coords_2d: 2D coordinates from dimensionality reduction [n_samples, 2]
            output_dir: Directory to save plots
        """
        print(f"   📊 Generating plots for Layer {layer_idx}...")

        os.makedirs(output_dir, exist_ok=True)

        # Common plot settings
        figsize = (10, 8)
        alpha = 0.6
        s = 50  # Point size

        # Plot 1: Color by Concept
        fig, ax = plt.subplots(figsize=figsize)
        concepts = df["concept"].unique()
        concept_colors = sns.color_palette("tab10", n_colors=len(concepts))

        for i, concept in enumerate(concepts):
            mask = df["concept"] == concept
            ax.scatter(
                coords_2d[mask, 0],
                coords_2d[mask, 1],
                c=[concept_colors[i]],
                label=concept,
                alpha=alpha,
                s=s,
                edgecolors="black",
                linewidth=0.5,
            )

        ax.set_xlabel("Dimension 1", fontsize=12)
        ax.set_ylabel("Dimension 2", fontsize=12)
        ax.set_title(f"Layer {layer_idx} - Colored by Concept", fontsize=14, fontweight="bold")
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        concept_path = os.path.join(output_dir, f"layer_{layer_idx}_concept.png")
        plt.savefig(concept_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"      ✅ Saved: {concept_path}")

        # Plot 2: Color by Modality
        fig, ax = plt.subplots(figsize=figsize)
        modalities = df["modality"].unique()
        modality_colors = {
            "vision": "#1f77b4",
            "text": "#ff7f0e",
        }  # Blue for vision, orange for text

        for modality in modalities:
            mask = df["modality"] == modality
            ax.scatter(
                coords_2d[mask, 0],
                coords_2d[mask, 1],
                c=modality_colors[modality],
                label=modality,
                alpha=alpha,
                s=s,
                edgecolors="black",
                linewidth=0.5,
            )

        ax.set_xlabel("Dimension 1", fontsize=12)
        ax.set_ylabel("Dimension 2", fontsize=12)
        ax.set_title(f"Layer {layer_idx} - Colored by Modality", fontsize=14, fontweight="bold")
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        modality_path = os.path.join(output_dir, f"layer_{layer_idx}_modality.png")
        plt.savefig(modality_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"      ✅ Saved: {modality_path}")

        # Plot 3: Color by Expert Choice with Confidence
        fig, ax = plt.subplots(figsize=figsize)

        # Base colors for each expert (will be modulated by confidence)
        base_colors = {
            "Expert 0": np.array([44, 160, 44]) / 255.0,  # Green
            "Expert 1": np.array([214, 39, 40]) / 255.0,  # Red
            "mixed": np.array([148, 103, 189]) / 255.0,  # Purple
            "unknown": np.array([127, 127, 127]) / 255.0,  # Gray
        }

        # Create scatter plot with confidence-modulated colors
        experts = df["expert_choice"].unique()

        for expert in experts:
            mask = df["expert_choice"] == expert
            base_color = base_colors.get(expert, base_colors["unknown"])

            # Get confidence values for this expert's points
            confidences = df.loc[mask, "routing_confidence"].values

            # Modulate color intensity by confidence
            # Low confidence (threshold) -> lighter, High confidence (1.0) -> full color
            colors = []
            threshold = self.expert_confidence_threshold
            conf_range = 1.0 - threshold

            for conf in confidences:
                # Blend between white (low confidence) and base color (high confidence)
                # Scale confidence from [threshold, 1.0] to [0.0, 1.0] for better visual range
                scaled_conf = (conf - threshold) / conf_range if conf >= threshold else 0.0
                scaled_conf = np.clip(scaled_conf, 0.0, 1.0)

                # Interpolate: (1-scaled_conf)*white + scaled_conf*base_color
                color = (1 - scaled_conf) * np.array([1.0, 1.0, 1.0]) + scaled_conf * base_color
                colors.append(color)

            scatter = ax.scatter(
                coords_2d[mask, 0],
                coords_2d[mask, 1],
                c=colors,
                label=expert,
                s=s,
                edgecolors="black",
                linewidth=0.5,
            )

        ax.set_xlabel("Dimension 1", fontsize=12)
        ax.set_ylabel("Dimension 2", fontsize=12)
        ax.set_title(
            f"Layer {layer_idx} - Expert Selection with Routing Confidence",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(loc="upper left", framealpha=0.9, title="Expert Choice")
        ax.grid(True, alpha=0.3)

        # Add colorbar for confidence
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        # Create a colormap that goes from white to gray (for visualization reference)
        # This represents the confidence gradient
        # Use the actual confidence threshold from the config
        norm = Normalize(vmin=self.expert_confidence_threshold, vmax=1.0)
        sm = ScalarMappable(cmap=plt.cm.Greys, norm=norm)
        sm.set_array([])

        cbar = plt.colorbar(sm, ax=ax, pad=0.02, aspect=30)
        cbar.set_label("Routing Confidence", rotation=270, labelpad=20, fontsize=11)

        plt.tight_layout()

        expert_path = os.path.join(output_dir, f"layer_{layer_idx}_expert.png")
        plt.savefig(expert_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"      ✅ Saved: {expert_path}")

    def compute_clustering_metrics(
        self, coords_2d: np.ndarray, labels: np.ndarray, label_type: str
    ) -> dict[str, float]:
        """
        Compute quantitative clustering metrics.

        Args:
            coords_2d: 2D coordinates [n_samples, 2]
            labels: Labels for each sample (concept, modality, or expert)
            label_type: Type of labels ("concept", "modality", or "expert")

        Returns:
            Dict with metrics:
                - silhouette_score: Cluster quality (-1 to 1, higher is better)
                - davies_bouldin_index: Cluster separation (lower is better)
        """
        # Filter out 'unknown' and 'mixed' labels for cleaner metrics
        if label_type == "expert":
            valid_mask = ~np.isin(labels, ["unknown", "mixed"])
            coords_2d = coords_2d[valid_mask]
            labels = labels[valid_mask]

        # Need at least 2 clusters and 2 samples per cluster
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return {
                "silhouette_score": 0.0,
                "davies_bouldin_index": float("inf"),
                "n_clusters": len(unique_labels),
                "n_samples": len(labels),
            }

        # Compute metrics
        try:
            silhouette = silhouette_score(coords_2d, labels)
        except:
            silhouette = 0.0

        try:
            davies_bouldin = davies_bouldin_score(coords_2d, labels)
        except:
            davies_bouldin = float("inf")

        return {
            "silhouette_score": float(silhouette),
            "davies_bouldin_index": float(davies_bouldin),
            "n_clusters": len(unique_labels),
            "n_samples": len(labels),
        }

    def generate_clustering_report(
        self,
        layer_idx: int,
        df: pd.DataFrame,
        metrics: dict[str, dict[str, float]],
        output_dir: str,
    ):
        """
        Generate markdown report with statistics and metrics.

        Args:
            layer_idx: Layer index
            df: DataFrame with all sample data
            metrics: Dict mapping label_type -> metrics dict
            output_dir: Directory to save report
        """
        print(f"   📝 Generating report for Layer {layer_idx}...")

        report_path = os.path.join(output_dir, f"layer_{layer_idx}_report.md")

        with open(report_path, "w") as f:
            f.write(f"# Layer {layer_idx} - Clustering Analysis Report\n\n")

            # Summary statistics
            f.write("## Summary Statistics\n\n")
            f.write(f"- **Total samples**: {len(df)}\n")
            f.write(f"- **Modalities**: {df['modality'].value_counts().to_dict()}\n")
            f.write(f"- **Concepts**: {df['concept'].value_counts().to_dict()}\n")
            f.write(f"- **Expert choices**: {df['expert_choice'].value_counts().to_dict()}\n\n")

            # Average routing confidence
            avg_confidence = df["routing_confidence"].mean()
            f.write(f"- **Average routing confidence**: {avg_confidence:.3f}\n\n")

            # Clustering metrics
            f.write("## Clustering Quality Metrics\n\n")
            f.write(
                "| Label Type | Silhouette Score | Davies-Bouldin Index | # Clusters | # Samples |\n"
            )
            f.write(
                "|------------|------------------|----------------------|------------|-----------|\n"
            )

            for label_type in ["concept", "modality", "expert"]:
                m = metrics[label_type]
                f.write(
                    f"| {label_type.capitalize()} | {m['silhouette_score']:.4f} | {m['davies_bouldin_index']:.4f} | {m['n_clusters']} | {m['n_samples']} |\n"
                )

            f.write("\n")

            # Metric interpretation
            f.write("### Metric Interpretation\n\n")
            f.write(
                "- **Silhouette Score**: Measures how well samples are clustered (-1 to 1, higher is better)\n"
            )
            f.write("  - Score > 0.5: Strong clustering\n")
            f.write("  - Score 0.2-0.5: Moderate clustering\n")
            f.write("  - Score < 0.2: Weak clustering\n\n")
            f.write("- **Davies-Bouldin Index**: Measures cluster separation (lower is better)\n")
            f.write("  - Score < 1.0: Well-separated clusters\n")
            f.write("  - Score 1.0-2.0: Moderate separation\n")
            f.write("  - Score > 2.0: Poorly separated clusters\n\n")

            # Expert-Modality alignment analysis
            f.write("## Expert-Modality Alignment Analysis\n\n")

            # Create contingency table
            contingency = pd.crosstab(df["modality"], df["expert_choice"], normalize="index") * 100
            f.write("### Routing Pattern by Modality (% of samples)\n\n")

            # Try to use markdown format, fallback to string representation
            try:
                f.write(contingency.to_markdown())
            except ImportError:
                # Fallback if tabulate is not available
                f.write("```\n")
                f.write(str(contingency))
                f.write("\n```")
            f.write("\n\n")

            # Interpretation
            f.write("### Interpretation\n\n")
            if "Expert 0" in contingency.columns and "Expert 1" in contingency.columns:
                vision_to_0 = (
                    contingency.loc["vision", "Expert 0"] if "vision" in contingency.index else 0
                )
                vision_to_1 = (
                    contingency.loc["vision", "Expert 1"] if "vision" in contingency.index else 0
                )
                text_to_0 = (
                    contingency.loc["text", "Expert 0"] if "text" in contingency.index else 0
                )
                text_to_1 = (
                    contingency.loc["text", "Expert 1"] if "text" in contingency.index else 0
                )

                f.write(
                    f"- Vision tokens: {vision_to_0:.1f}% → Expert 0, {vision_to_1:.1f}% → Expert 1\n"
                )
                f.write(
                    f"- Text tokens: {text_to_0:.1f}% → Expert 0, {text_to_1:.1f}% → Expert 1\n\n"
                )

                # Determine specialization pattern
                if vision_to_0 > 70 and text_to_1 > 70:
                    f.write(
                        "**Strong modality specialization detected**: Vision → Expert 0, Text → Expert 1\n"
                    )
                elif vision_to_1 > 70 and text_to_0 > 70:
                    f.write(
                        "**Strong modality specialization detected**: Vision → Expert 1, Text → Expert 0\n"
                    )
                elif max(vision_to_0, vision_to_1) < 60 or max(text_to_0, text_to_1) < 60:
                    f.write(
                        "**Weak modality specialization**: Routing is relatively balanced across experts\n"
                    )
                else:
                    f.write(
                        "**Moderate modality specialization**: Some preference but not strongly separated\n"
                    )

        print(f"      ✅ Saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="MoE Layer Clustering Analysis - Visualize representation space specialization"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to clustering analysis config JSON"
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="Override config: Layer indices to analyze (e.g., 0 15 31)",
    )
    parser.add_argument(
        "--reduction-method",
        type=str,
        default=None,
        choices=["tsne", "pacmap"],
        help="Override config: Dimensionality reduction method",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda or cpu)",
    )

    args = parser.parse_args()

    # Load config
    print(f"📋 Loading config from {args.config}")
    with open(args.config) as f:
        config = json.load(f)

    # Override config with CLI arguments
    if args.layers is not None:
        config["analysis"]["target_layers"] = args.layers
    if args.reduction_method is not None:
        config["analysis"]["reduction_method"] = args.reduction_method

    # Print configuration
    print(f"\n{'=' * 70}")
    print("🔬 MoE Layer Clustering Analysis")
    print(f"{'=' * 70}")
    print(f"Checkpoint: {config['checkpoint_path']}")
    print(f"Concepts: {config['data']['concepts']}")
    print(f"Samples per concept: {config['data']['samples_per_concept']}")
    print(f"Target layers: {config['analysis']['target_layers']}")
    print(f"Reduction method: {config['analysis']['reduction_method']}")
    print(f"Expert confidence threshold: {config['analysis']['expert_confidence_threshold']}")
    print(f"Output directory: {config['output']['save_dir']}")
    print(f"{'=' * 70}\n")

    # Initialize analyzer
    analyzer = MoEClusteringAnalyzer(device=args.device)

    # Load Stage 3 models
    analyzer.load_stage3_models(config["checkpoint_path"])

    # Extract concept samples from COCO
    concept_samples = analyzer.extract_concept_samples(
        annotations_file=config["data"]["annotations_file"],
        concepts=config["data"]["concepts"],
        samples_per_concept=config["data"]["samples_per_concept"],
        seed=config["data"]["seed"],
    )

    # Collect representations
    layer_dataframes = analyzer.collect_representations(
        concept_samples=concept_samples,
        image_dir=config["data"]["image_dir"],
        target_layers=config["analysis"]["target_layers"],
        pooling=config["analysis"]["pooling"],
        confidence_threshold=config["analysis"]["expert_confidence_threshold"],
    )

    # Analyze each layer
    for layer_idx in config["analysis"]["target_layers"]:
        print(f"\n{'=' * 70}")
        print(f"📊 Analyzing Layer {layer_idx}")
        print(f"{'=' * 70}")

        df = layer_dataframes[layer_idx]
        print(f"   Samples: {len(df)} ({df['modality'].value_counts().to_dict()})")

        # Stack representations
        representations = np.stack(df["representation"].values)

        # Dimensionality reduction
        coords_2d = analyzer.run_dimensionality_reduction(
            representations,
            method=config["analysis"]["reduction_method"],
            n_neighbors=config["analysis"]["reduction_params"]["n_neighbors"],
            perplexity=config["analysis"]["reduction_params"]["perplexity"],
        )

        # Generate plots
        analyzer.plot_clustering_analysis(
            layer_idx=layer_idx, df=df, coords_2d=coords_2d, output_dir=config["output"]["save_dir"]
        )

        # Compute metrics
        metrics = {}
        for label_type in ["concept", "modality", "expert"]:
            # Map 'expert' to 'expert_choice' column name
            column_name = "expert_choice" if label_type == "expert" else label_type
            metrics[label_type] = analyzer.compute_clustering_metrics(
                coords_2d=coords_2d, labels=df[column_name].values, label_type=label_type
            )
            print(
                f"   {label_type.capitalize()} clustering - Silhouette: {metrics[label_type]['silhouette_score']:.4f}, Davies-Bouldin: {metrics[label_type]['davies_bouldin_index']:.4f}"
            )

        # Generate report
        analyzer.generate_clustering_report(
            layer_idx=layer_idx, df=df, metrics=metrics, output_dir=config["output"]["save_dir"]
        )

        # Save representations if requested
        if config["output"].get("save_representations", False):
            repr_path = os.path.join(
                config["output"]["save_dir"], f"layer_{layer_idx}_representations.npz"
            )
            np.savez(
                repr_path,
                representations=representations,
                coords_2d=coords_2d,
                concepts=df["concept"].values,
                modalities=df["modality"].values,
                expert_choices=df["expert_choice"].values,
                routing_confidences=df["routing_confidence"].values,
            )
            print(f"   💾 Saved representations: {repr_path}")

    print(f"\n{'=' * 70}")
    print(f"✅ Analysis complete! Results saved to {config['output']['save_dir']}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
