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

import numpy as np
import torch

from analysis_scripts import attention_routing_plots as arp
from analysis_scripts._lib import majority_vote_expert

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

        return majority_vote_expert(expert_choices, confidence_threshold)

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
        arp.plot_attention_routing_evolution(layer_metrics, output_dir)

        # Generate expert-attention correlation plots if enabled
        if analyze_expert_correlation and layer_token_data is not None:
            arp.plot_expert_attention_correlation(layer_token_data, output_dir)

        return layer_metrics


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
