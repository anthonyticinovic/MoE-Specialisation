"""
MoE Attention-Routing Analysis

Analyzes how attention patterns and expert routing evolve across all layers
of the MoE model to understand when and how the model transitions from
modality-specific to multimodal processing.

Usage:
    python analysis_scripts/attention_routing_analysis.py \
        --config configs/attention_routing_analysis.json \
        --device cuda
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

# Import base analyzer
from analysis_scripts.cross_modality_purity import CrossModalityPurityAnalyzer


class AttentionRoutingAnalyzer(CrossModalityPurityAnalyzer):
    """
    Extends CrossModalityPurityAnalyzer to analyze attention and routing patterns.

    Examines how attention patterns evolve across layers and how they relate to
    expert routing decisions.
    """

    def __init__(self, config_path: str = "configs/training_config.yaml", device: str = "cuda"):
        super().__init__(config_path, device)

    def extract_random_samples(
        self, annotations_file: str, num_samples: int, min_caption_length: int = 5, seed: int = 42
    ) -> list[dict]:
        """
        Extract random samples from COCO annotations.

        Args:
            annotations_file: Path to COCO annotations JSON
            num_samples: Number of samples to extract
            min_caption_length: Minimum number of tokens in caption
            seed: Random seed for reproducibility

        Returns:
            List of sample dicts with keys:
                - 'image_id': COCO image ID
                - 'caption': Image caption
                - 'image_path': Image filename
        """
        print(f"📚 Extracting {num_samples} random samples from COCO...")
        print(f"   Minimum caption length: {min_caption_length} tokens")

        # Load COCO annotations
        with open(annotations_file) as f:
            coco_data = json.load(f)

        # Build image_id -> image_path mapping
        image_id_to_path = {}
        for img in coco_data["images"]:
            image_id_to_path[img["id"]] = img["file_name"]

        # Filter captions by length
        valid_samples = []
        for annotation in coco_data["annotations"]:
            caption = annotation["caption"]
            # Rough token count (split by whitespace)
            if len(caption.split()) >= min_caption_length:
                valid_samples.append(
                    {
                        "image_id": annotation["image_id"],
                        "caption": caption,
                        "image_path": image_id_to_path[annotation["image_id"]],
                    }
                )

        print(f"   Found {len(valid_samples)} valid samples")

        # Random sample
        np.random.seed(seed)
        selected_indices = np.random.choice(
            len(valid_samples), size=min(num_samples, len(valid_samples)), replace=False
        )
        selected_samples = [valid_samples[i] for i in selected_indices]

        print(f"   ✅ Selected {len(selected_samples)} random samples")
        return selected_samples

    def _extract_attention_with_routing(
        self, image_path: str, caption: str, num_vision_tokens: int = 257
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Extract attention weights and routing decisions from all layers.

        Args:
            image_path: Path to image file
            caption: Text caption
            num_vision_tokens: Number of vision tokens (default 257)

        Returns:
            Tuple of (attention_weights_list, routing_logits_list)
            - attention_weights_list: List of [num_heads, seq_len, seq_len] per layer
            - routing_logits_list: List of [seq_len, num_experts] per layer
        """
        # Prepare inputs
        visual_tokens = self._prepare_vision_input(image_path)
        text_embeddings = self._prepare_text_input(caption)

        # Combine for single forward pass
        combined_embeddings = torch.cat([visual_tokens, text_embeddings], dim=1)

        # Get model reference
        model_to_inspect = self.llm.module if hasattr(self.llm, "module") else self.llm

        # Forward pass with attention output
        with torch.no_grad():
            outputs = model_to_inspect(
                inputs_embeds=combined_embeddings,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )

        # Extract attention weights (tuple of tensors, one per layer)
        # Each is [batch, num_heads, seq_len, seq_len]
        attention_weights = outputs.attentions  # Tuple of 32 layers

        # Extract routing logits from each MoE layer
        routing_logits_list = []
        for layer_idx in range(len(model_to_inspect.model.layers)):
            moe_layer = model_to_inspect.model.layers[layer_idx].mlp
            if hasattr(moe_layer, "_last_router_logits"):
                routing_logits = moe_layer._last_router_logits[0]  # [seq_len, num_experts]
                routing_logits_list.append(routing_logits.cpu())
            else:
                # Fallback: create dummy routing (shouldn't happen)
                seq_len = combined_embeddings.shape[1]
                routing_logits_list.append(torch.zeros(seq_len, 2))

        # Convert attention to list and move to CPU
        attention_weights_list = [attn[0].cpu() for attn in attention_weights]  # Remove batch dim

        return attention_weights_list, routing_logits_list

    def _compute_token_level_attention_by_expert(
        self, attention_weights: torch.Tensor, routing_logits: torch.Tensor, num_vision_tokens: int
    ) -> dict[str, list[float]]:
        """
        Compute per-token attention statistics grouped by expert routing.

        For each TEXT token, determine which expert it routes to and compute
        its text→vision attention mass.

        Args:
            attention_weights: [num_heads, seq_len, seq_len]
            routing_logits: [seq_len, num_experts]
            num_vision_tokens: Number of vision tokens

        Returns:
            Dict with keys 'expert0_attentions' and 'expert1_attentions',
            each containing a list of text→vision attention values for tokens
            routed to that expert.
        """
        # Average attention across heads
        attn = attention_weights.mean(dim=0)  # [seq_len, seq_len]

        # Get expert assignments for all tokens
        routing_probs = torch.softmax(routing_logits, dim=-1)  # [seq_len, num_experts]
        expert_assignments = routing_probs.argmax(dim=1).cpu().numpy()  # [seq_len]

        # Identify text token indices
        seq_len = attn.shape[0]
        text_indices = list(range(num_vision_tokens, seq_len))

        # Compute text→vision attention for each text token, grouped by expert (0 or 1)
        expert0_attentions = []
        expert1_attentions = []

        for text_idx in text_indices:
            # Compute attention mass from this text token to all vision tokens
            text_to_vision = attn[text_idx, :num_vision_tokens].sum().item()

            # Group by expert assignment (hardcoded to Expert 0 and Expert 1)
            expert = expert_assignments[text_idx]
            if expert == 0:
                expert0_attentions.append(text_to_vision)
            elif expert == 1:
                expert1_attentions.append(text_to_vision)

        return {"expert0_attentions": expert0_attentions, "expert1_attentions": expert1_attentions}

    def _extract_expert_choice_for_tokens(
        self,
        routing_logits: torch.Tensor,
        token_indices: torch.Tensor,
        confidence_threshold: float = 0.6,
    ) -> tuple[str, float]:
        """
        Extract dominant expert for a set of tokens using majority voting.

        Args:
            routing_logits: [seq_len, num_experts] routing logits
            token_indices: Indices of tokens to analyze (e.g., text tokens only)
            confidence_threshold: Minimum vote fraction for decisive assignment

        Returns:
            Tuple of (expert_label, confidence)
            - expert_label: "Expert 0", "Expert 1", or "mixed"
            - confidence: Fraction of tokens voting for winner
        """
        if len(token_indices) == 0:
            return "unknown", 0.0

        # Convert logits to probabilities and get argmax expert for each token
        routing_probs = torch.softmax(routing_logits, dim=-1)  # [seq_len, num_experts]
        expert_choices = routing_probs[token_indices].argmax(dim=1).cpu().numpy()  # [num_tokens]

        # Count votes
        from collections import Counter

        votes = Counter(expert_choices)
        total_votes = len(expert_choices)

        if total_votes == 0 or len(votes) == 0:
            return "unknown", 0.0

        # Find winner
        winner_expert = votes.most_common(1)[0][0]
        winner_count = votes[winner_expert]
        winner_fraction = winner_count / total_votes

        # Check if winner meets confidence threshold
        if winner_fraction >= confidence_threshold:
            return f"Expert {winner_expert}", winner_fraction
        else:
            return "mixed", winner_fraction

    def _compute_attention_statistics(
        self,
        attention_weights: torch.Tensor,
        routing_logits: torch.Tensor,
        num_vision_tokens: int,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
        exclude_self_attention: bool = True,
    ) -> dict[str, float]:
        """
        Compute attention statistics for a single sample at a single layer.

        Args:
            attention_weights: [num_heads, seq_len, seq_len]
            routing_logits: [seq_len, num_experts]
            num_vision_tokens: Number of vision tokens
            pad_token_id: Padding token ID to exclude
            eos_token_id: EOS token ID to exclude
            exclude_self_attention: Whether to exclude self-attention

        Returns:
            Dict with metrics:
                - text_to_vision_attention: Mean attention mass from text to vision
                - vision_to_vision_attention: Mean attention mass within vision
                - text_to_text_attention: Mean attention mass within text
                - text_attention_entropy: Attention focus for text tokens
                - vision_attention_entropy: Attention focus for vision tokens
                - vision_routing_entropy: Normalized routing entropy for vision
                - text_routing_entropy: Normalized routing entropy for text
        """
        # Average across heads: [seq_len, seq_len]
        attn = attention_weights.mean(dim=0)

        # Identify vision and text regions
        vision_mask = torch.arange(attn.shape[0]) < num_vision_tokens
        text_mask = torch.arange(attn.shape[0]) >= num_vision_tokens

        # Extract attention submatrices
        vision_indices = torch.where(vision_mask)[0]
        text_indices = torch.where(text_mask)[0]

        # Compute cross-modal attention (attention mass)
        if len(text_indices) > 0 and len(vision_indices) > 0:
            # Text → Vision: sum of attention from each text token to all vision tokens
            # attn[text_indices][:, vision_indices] is [num_text, num_vision]
            # Sum over vision dimension, then average across text tokens
            text_to_vision = attn[text_indices][:, vision_indices].sum(dim=1).mean().item()
        else:
            text_to_vision = 0.0

        # Compute intra-modal attention mass (excluding self-attention)
        if len(vision_indices) > 1:
            vision_to_vision_matrix = attn[vision_indices][:, vision_indices]
            if exclude_self_attention:
                # Zero out diagonal, then sum over target dimension, average over source
                vision_to_vision_matrix_masked = vision_to_vision_matrix.clone()
                vision_to_vision_matrix_masked.fill_diagonal_(0)
                vision_to_vision = vision_to_vision_matrix_masked.sum(dim=1).mean().item()
            else:
                # Sum over target dimension, average over source
                vision_to_vision = vision_to_vision_matrix.sum(dim=1).mean().item()
        else:
            vision_to_vision = 0.0

        if len(text_indices) > 1:
            text_to_text_matrix = attn[text_indices][:, text_indices]
            if exclude_self_attention:
                # Zero out diagonal, then sum over target dimension, average over source
                text_to_text_matrix_masked = text_to_text_matrix.clone()
                text_to_text_matrix_masked.fill_diagonal_(0)
                text_to_text = text_to_text_matrix_masked.sum(dim=1).mean().item()
            else:
                # Sum over target dimension, average over source
                text_to_text = text_to_text_matrix.sum(dim=1).mean().item()
        else:
            text_to_text = 0.0

        # Compute attention entropy (focus)
        def attention_entropy(attn_dist):
            """Compute Shannon entropy of attention distribution."""
            # Add small epsilon to avoid log(0)
            eps = 1e-10
            attn_dist = attn_dist + eps
            entropy = -(attn_dist * torch.log(attn_dist)).sum(dim=-1)
            return entropy.mean().item()

        if len(text_indices) > 0:
            text_attn_dist = attn[text_indices]  # [num_text, seq_len]
            text_entropy = attention_entropy(text_attn_dist)
        else:
            text_entropy = 0.0

        if len(vision_indices) > 0:
            vision_attn_dist = attn[vision_indices]  # [num_vision, seq_len]
            vision_entropy = attention_entropy(vision_attn_dist)
        else:
            vision_entropy = 0.0

        # Compute routing entropy (normalized)
        routing_probs = torch.softmax(routing_logits, dim=-1)  # [seq_len, num_experts]
        num_experts = routing_probs.shape[1]

        def routing_entropy_normalized(probs, indices):
            """Compute normalized Shannon entropy of routing distribution."""
            if len(indices) == 0:
                return 0.0

            # Get fraction routing to each expert
            expert_votes = probs[indices].argmax(dim=-1)  # [num_tokens]
            counts = torch.bincount(expert_votes, minlength=num_experts).float()
            p = counts / counts.sum()

            # Shannon entropy
            eps = 1e-10
            entropy = -(p * torch.log(p + eps)).sum().item()

            # Normalize by max entropy (log(num_experts))
            max_entropy = np.log(num_experts)
            return entropy / max_entropy if max_entropy > 0 else 0.0

        vision_routing_entropy = routing_entropy_normalized(routing_probs, vision_indices)
        text_routing_entropy = routing_entropy_normalized(routing_probs, text_indices)

        return {
            "text_to_vision_attention": text_to_vision,
            "vision_to_vision_attention": vision_to_vision,
            "text_to_text_attention": text_to_text,
            "text_attention_entropy": text_entropy,
            "vision_attention_entropy": vision_entropy,
            "vision_routing_entropy": vision_routing_entropy,
            "text_routing_entropy": text_routing_entropy,
        }

    def analyze_attention_routing_across_layers(
        self,
        samples: list[dict],
        image_dir: str,
        num_vision_tokens: int = 257,
        exclude_self_attention: bool = True,
        expert_confidence_threshold: float = 0.6,
        analyze_expert_correlation: bool = False,
        output_dir: str = "results/attention_routing_analysis",
    ) -> dict[int, dict[str, list[float]]]:
        """
        Analyze attention and routing patterns across all layers for multiple samples.

        Args:
            samples: List of sample dicts with 'image_path' and 'caption'
            image_dir: Base directory for images
            num_vision_tokens: Number of vision tokens
            exclude_self_attention: Whether to exclude self-attention
            expert_confidence_threshold: Threshold for expert assignment confidence
            analyze_expert_correlation: Whether to analyze expert-attention correlation
            output_dir: Directory to save plots

        Returns:
            Dict mapping layer_idx -> metric_name -> list of values across samples
        """
        print(f"\n{'=' * 70}")
        print("🔬 Analyzing Attention-Routing Patterns Across All Layers")
        print(f"{'=' * 70}")
        print(f"Samples: {len(samples)}")
        print(f"Vision tokens: {num_vision_tokens}")
        print(f"Exclude self-attention: {exclude_self_attention}")
        if analyze_expert_correlation:
            print(f"Expert correlation: ENABLED (threshold={expert_confidence_threshold})")
        print(f"{'=' * 70}\n")

        # Initialize storage for metrics per layer
        num_layers = 32  # Mistral has 32 layers
        layer_metrics = {
            layer_idx: {
                "text_to_vision_attention": [],
                "vision_to_vision_attention": [],
                "text_to_text_attention": [],
                "text_attention_entropy": [],
                "vision_attention_entropy": [],
                "vision_routing_entropy": [],
                "text_routing_entropy": [],
            }
            for layer_idx in range(num_layers)
        }

        # Initialize storage for expert-attention correlation (if enabled)
        # Token-level: store lists of attention values per expert per layer
        layer_token_data = (
            {
                layer_idx: {"expert0_attentions": [], "expert1_attentions": []}
                for layer_idx in range(num_layers)
            }
            if analyze_expert_correlation
            else None
        )

        # Process each sample
        for sample_idx, sample in enumerate(samples):
            if (sample_idx + 1) % 10 == 0:
                print(f"   Processing sample {sample_idx + 1}/{len(samples)}...")

            image_path = os.path.join(image_dir, sample["image_path"])
            caption = sample["caption"]

            try:
                # Extract attention and routing for all layers
                attention_weights_list, routing_logits_list = self._extract_attention_with_routing(
                    image_path, caption, num_vision_tokens
                )

                # Compute statistics for each layer
                for layer_idx in range(num_layers):
                    stats = self._compute_attention_statistics(
                        attention_weights_list[layer_idx],
                        routing_logits_list[layer_idx],
                        num_vision_tokens,
                        exclude_self_attention=exclude_self_attention,
                    )

                    # Store metrics
                    for metric_name, value in stats.items():
                        layer_metrics[layer_idx][metric_name].append(value)

                    # If expert correlation analysis is enabled, collect token-level data
                    if analyze_expert_correlation:
                        # Compute token-level attention by expert
                        token_attentions = self._compute_token_level_attention_by_expert(
                            attention_weights_list[layer_idx],
                            routing_logits_list[layer_idx],
                            num_vision_tokens,
                        )

                        # Extend the attention lists for each expert
                        layer_token_data[layer_idx]["expert0_attentions"].extend(
                            token_attentions["expert0_attentions"]
                        )
                        layer_token_data[layer_idx]["expert1_attentions"].extend(
                            token_attentions["expert1_attentions"]
                        )

            except Exception as e:
                print(f"      ⚠️  Error processing sample {sample_idx}: {e}")
                continue

        print(f"\n✅ Processed {len(samples)} samples across {num_layers} layers")

        # Generate plots
        self._plot_attention_routing_evolution(layer_metrics, output_dir)

        # Generate expert-attention correlation plots if enabled
        if analyze_expert_correlation and layer_token_data is not None:
            self._plot_expert_attention_correlation(layer_token_data, output_dir)

        return layer_metrics

    def _plot_attention_routing_evolution(
        self, layer_metrics: dict[int, dict[str, list[float]]], output_dir: str
    ):
        """
        Generate line plots showing how attention metrics evolve across layers.

        Args:
            layer_metrics: Dict mapping layer_idx -> metric_name -> values
            output_dir: Directory to save plots
        """
        print("\n📊 Generating attention-routing evolution plots...")
        os.makedirs(output_dir, exist_ok=True)

        num_layers = len(layer_metrics)
        layers = np.arange(num_layers)

        # Compute mean and std for each metric at each layer
        def get_mean_std(metric_name):
            means = []
            stds = []
            for layer_idx in range(num_layers):
                values = layer_metrics[layer_idx][metric_name]
                if len(values) > 0:
                    means.append(np.mean(values))
                    stds.append(np.std(values))
                else:
                    means.append(0.0)
                    stds.append(0.0)
            return np.array(means), np.array(stds)

        # Plot 1: Attention Patterns (Cross-Modal + Intra-Modal)
        fig, ax = plt.subplots(figsize=(12, 6))

        # Cross-modal attention
        means_t2v, stds_t2v = get_mean_std("text_to_vision_attention")
        ax.plot(
            layers,
            means_t2v,
            linewidth=2.5,
            label="Text → Vision (cross-modal)",
            color="#1f77b4",
            linestyle="-",
        )
        ax.fill_between(
            layers, means_t2v - stds_t2v, means_t2v + stds_t2v, alpha=0.3, color="#1f77b4"
        )

        # Intra-modal attention (text only)
        means_t2t, stds_t2t = get_mean_std("text_to_text_attention")
        ax.plot(
            layers,
            means_t2t,
            linewidth=2,
            label="Text → Text (intra-modal)",
            color="#ff7f0e",
            linestyle="--",
        )
        ax.fill_between(
            layers, means_t2t - stds_t2t, means_t2t + stds_t2t, alpha=0.3, color="#ff7f0e"
        )

        ax.set_xlabel("Layer Index", fontsize=12)
        ax.set_ylabel("Mean Attention Mass", fontsize=12)
        ax.set_title("Attention Patterns Across Layers", fontsize=14, fontweight="bold")
        ax.legend(loc="best", framealpha=0.9, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, num_layers - 1)
        # Auto-scale y-axis with margin, minimum at 0
        y_max = max(max(means_t2v + stds_t2v), max(means_t2t + stds_t2t))
        ax.set_ylim(0, min(1.0, y_max * 1.1))  # Cap at 1.0

        # Add note about self-attention
        ax.text(
            0.98,
            0.02,
            "Note: Intra-modal excludes self-attention",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            horizontalalignment="right",
            alpha=0.7,
            style="italic",
        )

        plt.tight_layout()

        plot_path = os.path.join(output_dir, "attention_patterns.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"   ✅ Saved: {plot_path}")

        # Plot 3: Attention Focus (Entropy)
        fig, ax = plt.subplots(figsize=(12, 6))

        means_text_ent, stds_text_ent = get_mean_std("text_attention_entropy")
        means_vision_ent, stds_vision_ent = get_mean_std("vision_attention_entropy")

        ax.plot(
            layers, means_text_ent, linewidth=2, label="Text Attention Entropy", color="#d62728"
        )
        ax.fill_between(
            layers,
            means_text_ent - stds_text_ent,
            means_text_ent + stds_text_ent,
            alpha=0.3,
            color="#d62728",
        )

        ax.plot(
            layers, means_vision_ent, linewidth=2, label="Vision Attention Entropy", color="#9467bd"
        )
        ax.fill_between(
            layers,
            means_vision_ent - stds_vision_ent,
            means_vision_ent + stds_vision_ent,
            alpha=0.3,
            color="#9467bd",
        )

        ax.set_xlabel("Layer Index", fontsize=12)
        ax.set_ylabel("Attention Entropy (nats)", fontsize=12)
        ax.set_title("Attention Focus Across Layers", fontsize=14, fontweight="bold")
        ax.legend(loc="best", framealpha=0.9, title="Lower = more focused")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, num_layers - 1)
        plt.tight_layout()

        plot_path = os.path.join(output_dir, "attention_focus.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"   ✅ Saved: {plot_path}")

        print(f"\n✅ All plots saved to {output_dir}")

    def _plot_expert_attention_correlation(
        self, layer_token_data: dict[int, dict[str, list[float]]], output_dir: str
    ):
        """
        Generate plot showing token-level attention patterns grouped by expert routing.

        Plots mean text→vision attention for Expert 0 vs Expert 1 across all layers.

        Args:
            layer_token_data: Dict mapping layer_idx -> {'expert0_attentions': [...], 'expert1_attentions': [...]}
            output_dir: Directory to save plot
        """
        print("\n📊 Generating expert-attention correlation plot (token-level)...")

        num_layers = len(layer_token_data)
        layers = np.arange(num_layers)

        # For each layer, compute mean and std of token-level attention by expert
        expert0_means = []
        expert0_stds = []
        expert0_counts = []

        expert1_means = []
        expert1_stds = []
        expert1_counts = []

        for layer_idx in range(num_layers):
            # Expert 0 statistics
            expert0_attention = layer_token_data[layer_idx]["expert0_attentions"]
            if len(expert0_attention) > 0:
                expert0_means.append(np.mean(expert0_attention))
                expert0_stds.append(np.std(expert0_attention))
                expert0_counts.append(len(expert0_attention))
            else:
                expert0_means.append(np.nan)
                expert0_stds.append(np.nan)
                expert0_counts.append(0)

            # Expert 1 statistics
            expert1_attention = layer_token_data[layer_idx]["expert1_attentions"]
            if len(expert1_attention) > 0:
                expert1_means.append(np.mean(expert1_attention))
                expert1_stds.append(np.std(expert1_attention))
                expert1_counts.append(len(expert1_attention))
            else:
                expert1_means.append(np.nan)
                expert1_stds.append(np.nan)
                expert1_counts.append(0)

        # Convert to arrays
        expert0_means = np.array(expert0_means)
        expert0_stds = np.array(expert0_stds)
        expert0_counts = np.array(expert0_counts)

        expert1_means = np.array(expert1_means)
        expert1_stds = np.array(expert1_stds)
        expert1_counts = np.array(expert1_counts)

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Helper function to plot continuous segments
        def plot_continuous_segments(layers, means, stds, color, linestyle, label):
            """Plot line in continuous segments, skipping NaN values."""
            valid = ~np.isnan(means)

            # Find continuous segments
            segments = []
            start = None
            for i in range(len(valid)):
                if valid[i]:
                    if start is None:
                        start = i
                else:
                    if start is not None:
                        segments.append((start, i))
                        start = None
            if start is not None:
                segments.append((start, len(valid)))

            # Plot each continuous segment
            for seg_start, seg_end in segments:
                seg_layers = layers[seg_start:seg_end]
                seg_means = means[seg_start:seg_end]
                seg_stds = stds[seg_start:seg_end]

                # Plot line with uniform style
                ax.plot(
                    seg_layers,
                    seg_means,
                    color=color,
                    linestyle=linestyle,
                    linewidth=2.5,
                    solid_capstyle="round",
                )

                # Plot error band for this segment
                ax.fill_between(
                    seg_layers, seg_means - seg_stds, seg_means + seg_stds, color=color, alpha=0.2
                )

        # Plot Expert 0 (blue, solid)
        plot_continuous_segments(layers, expert0_means, expert0_stds, "#1f77b4", "-", "Expert 0")

        # Plot Expert 1 (red, dashed)
        plot_continuous_segments(layers, expert1_means, expert1_stds, "#d62728", "--", "Expert 1")

        # Add legend with dummy lines (full opacity for legend)
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color="#1f77b4", linewidth=2.5, linestyle="-", label="Expert 0"),
            Line2D([0], [0], color="#d62728", linewidth=2.5, linestyle="--", label="Expert 1"),
        ]
        ax.legend(handles=legend_elements, loc="best", framealpha=0.9, fontsize=11)

        # Labels and formatting
        ax.set_xlabel("Layer Index", fontsize=12)
        ax.set_ylabel("Mean Text → Vision Attention", fontsize=12)
        ax.set_title(
            "Expert-Specific Attention Patterns Across Layers", fontsize=14, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, num_layers - 1)
        ax.set_ylim(0, 1.0)

        # Add note about token-level analysis
        ax.text(
            0.98,
            0.02,
            "Note: Token-level analysis. Each point = one text token's attention to vision.",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            horizontalalignment="right",
            alpha=0.7,
            style="italic",
        )

        plt.tight_layout()

        plot_path = os.path.join(output_dir, "expert_attention_correlation.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"   ✅ Saved: {plot_path}")

        # Print summary statistics
        print("\n📈 Expert-Attention Correlation Summary (Token-Level):")
        print(
            f"   Expert 0 tokens per layer: {expert0_counts.mean():.0f} ± {expert0_counts.std():.0f}"
        )
        print(
            f"   Expert 1 tokens per layer: {expert1_counts.mean():.0f} ± {expert1_counts.std():.0f}"
        )
        print(f"   Total tokens analyzed: {expert0_counts.sum() + expert1_counts.sum():.0f}")

        # Report layers with zero tokens for either expert
        zero_e0_layers = np.where(expert0_counts == 0)[0]
        zero_e1_layers = np.where(expert1_counts == 0)[0]
        if len(zero_e0_layers) > 0:
            print(f"   ⚠️  Layers with ZERO Expert 0 tokens: {list(zero_e0_layers)}")
        if len(zero_e1_layers) > 0:
            print(f"   ⚠️  Layers with ZERO Expert 1 tokens: {list(zero_e1_layers)}")

        # Report expert balance
        total_e0 = expert0_counts.sum()
        total_e1 = expert1_counts.sum()
        if total_e0 + total_e1 > 0:
            e0_fraction = total_e0 / (total_e0 + total_e1)
            e1_fraction = total_e1 / (total_e0 + total_e1)
            print(f"   Expert routing balance: E0={e0_fraction:.1%}, E1={e1_fraction:.1%}")

        # Find layers with largest difference
        valid_both = valid_e0 & valid_e1
        if valid_both.any():
            differences = np.abs(expert0_means[valid_both] - expert1_means[valid_both])
            max_diff_idx = layers[valid_both][np.argmax(differences)]
            max_diff = differences.max()
            print(f"   Largest attention difference: Layer {max_diff_idx} ({max_diff:.3f})")
            print(f"      Expert 0: {expert0_means[max_diff_idx]:.3f}")
            print(f"      Expert 1: {expert1_means[max_diff_idx]:.3f}")


def main():
    """Main function for attention-routing analysis across layers."""
    parser = argparse.ArgumentParser(
        description="MoE Attention-Routing Analysis - Analyze attention patterns across all layers"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to attention routing analysis config JSON"
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

    # Print configuration
    print(f"\n{'=' * 70}")
    print("🔬 MoE Attention-Routing Analysis")
    print(f"{'=' * 70}")
    print(f"Checkpoint: {config['checkpoint_path']}")
    print(f"Number of samples: {config['data']['num_samples']}")
    print(f"Min caption length: {config['data']['min_caption_length']}")
    print(f"Exclude self-attention: {config['analysis']['exclude_self_attention']}")
    print(f"Output directory: {config['output']['save_dir']}")
    print(f"{'=' * 70}\n")

    # Initialize analyzer
    analyzer = AttentionRoutingAnalyzer(device=args.device)

    # Load Stage 3 models
    analyzer.load_stage3_models(config["checkpoint_path"])

    # Extract random samples from COCO
    samples = analyzer.extract_random_samples(
        annotations_file=config["data"]["annotations_file"],
        num_samples=config["data"]["num_samples"],
        min_caption_length=config["data"]["min_caption_length"],
        seed=config["data"]["seed"],
    )

    # Analyze attention and routing across all layers
    layer_metrics = analyzer.analyze_attention_routing_across_layers(
        samples=samples,
        image_dir=config["data"]["image_dir"],
        num_vision_tokens=config["analysis"]["num_vision_tokens"],
        exclude_self_attention=config["analysis"]["exclude_self_attention"],
        expert_confidence_threshold=config["analysis"].get("expert_confidence_threshold", 0.6),
        analyze_expert_correlation=config["analysis"].get("analyze_expert_correlation", False),
        output_dir=config["output"]["save_dir"],
    )

    print(f"\n{'=' * 70}")
    print("✅ Attention-routing analysis complete!")
    print(f"   Results saved to {config['output']['save_dir']}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
