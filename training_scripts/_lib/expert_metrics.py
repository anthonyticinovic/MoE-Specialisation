"""Routing metrics collected during Stage 3 validation.

``ExpertUsageTracker`` accumulates four quantities per layer — expert load,
routing entropy, routing confidence, and the visual/text split — from the
router probabilities each :class:`~models.moe_layer.MoELayer` caches on its
forward pass. ``save_expert_metrics`` writes them to JSON and logs a summary.

These are the numbers the paper's routing analysis is built on, which is why
they live in their own module rather than inside the training script.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ExpertUsageTracker:
    """
    Lightweight tracker for MoE expert utilization and routing patterns.
    Collects metrics during validation for research analysis.

    Tracks 4 key metrics:
    1. Expert Load Distribution: How evenly work is distributed across experts
    2. Routing Entropy: Uncertainty in routing decisions
    3. Routing Confidence: Fraction of high-confidence routing decisions
    4. Visual vs Text Routing: Routing pattern differences by modality
    """

    def __init__(self, num_layers=32, num_experts=2, visual_token_end=255):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.visual_token_end = visual_token_end  # Tokens 0 to visual_token_end are visual

        # Per-layer accumulators (memory efficient - just sums and counts)
        self.layer_expert_loads = [np.zeros(num_experts) for _ in range(num_layers)]
        self.layer_entropies = [[] for _ in range(num_layers)]
        self.layer_high_conf_counts = [0 for _ in range(num_layers)]
        self.layer_total_tokens = [0 for _ in range(num_layers)]

        # Visual vs Text routing (per-layer)
        self.layer_visual_expert_loads = [np.zeros(num_experts) for _ in range(num_layers)]
        self.layer_text_expert_loads = [np.zeros(num_experts) for _ in range(num_layers)]
        self.layer_visual_tokens = [0 for _ in range(num_layers)]
        self.layer_text_tokens = [0 for _ in range(num_layers)]

    def update(self, layer_idx, router_probs, token_positions):
        """
        Update metrics for a single layer.

        Args:
            layer_idx: Layer index (0-31)
            router_probs: [batch_size, seq_len, num_experts] routing probabilities
            token_positions: [batch_size, seq_len] absolute token positions in sequence
        """
        # Flatten to [total_tokens, num_experts]
        probs = router_probs.reshape(-1, self.num_experts)
        positions = token_positions.reshape(-1)

        # 1. Expert Load Distribution (how much work each expert gets)
        expert_loads = probs.sum(dim=0).cpu().numpy()  # [num_experts]
        self.layer_expert_loads[layer_idx] += expert_loads

        # 2. Routing Entropy (uncertainty in routing decisions)
        # H = -sum(p * log(p)) for each token, then average
        eps = 1e-10
        token_entropies = -(probs * torch.log(probs + eps)).sum(dim=1)  # [total_tokens]
        self.layer_entropies[layer_idx].extend(token_entropies.cpu().numpy().tolist())

        # 3. Routing Confidence (fraction with prob > 0.7 for any expert)
        max_probs = probs.max(dim=1)[0]  # [total_tokens]
        high_conf = (max_probs > 0.7).sum().item()
        self.layer_high_conf_counts[layer_idx] += high_conf
        self.layer_total_tokens[layer_idx] += probs.shape[0]

        # 4. Visual vs Text Routing
        visual_mask = positions <= self.visual_token_end
        text_mask = positions > self.visual_token_end

        if visual_mask.any():
            visual_loads = probs[visual_mask].sum(dim=0).cpu().numpy()
            self.layer_visual_expert_loads[layer_idx] += visual_loads
            self.layer_visual_tokens[layer_idx] += visual_mask.sum().item()

        if text_mask.any():
            text_loads = probs[text_mask].sum(dim=0).cpu().numpy()
            self.layer_text_expert_loads[layer_idx] += text_loads
            self.layer_text_tokens[layer_idx] += text_mask.sum().item()

    def compute_metrics(self):
        """
        Compute final metrics from accumulated data.
        Returns dict with per-layer and aggregate metrics.
        """
        metrics = {"per_layer": [], "aggregate": {}}

        # Compute per-layer metrics
        for layer_idx in range(self.num_layers):
            layer_metrics = {
                "layer": layer_idx,
                "expert_load_distribution": {},
                "avg_routing_entropy": 0.0,
                "high_confidence_fraction": 0.0,
                "visual_vs_text_routing": {},
            }

            # 1. Expert Load Distribution (normalize to percentages)
            total_load = self.layer_expert_loads[layer_idx].sum()
            if total_load > 0:
                load_pcts = (self.layer_expert_loads[layer_idx] / total_load * 100).tolist()
                layer_metrics["expert_load_distribution"] = {
                    f"expert_{i}": round(pct, 2) for i, pct in enumerate(load_pcts)
                }

            # 2. Average Routing Entropy
            if self.layer_entropies[layer_idx]:
                layer_metrics["avg_routing_entropy"] = round(
                    np.mean(self.layer_entropies[layer_idx]), 4
                )

            # 3. High Confidence Fraction
            if self.layer_total_tokens[layer_idx] > 0:
                layer_metrics["high_confidence_fraction"] = round(
                    self.layer_high_conf_counts[layer_idx] / self.layer_total_tokens[layer_idx], 4
                )

            # 4. Visual vs Text Routing
            visual_total = self.layer_visual_expert_loads[layer_idx].sum()
            text_total = self.layer_text_expert_loads[layer_idx].sum()

            if visual_total > 0:
                visual_pcts = (
                    self.layer_visual_expert_loads[layer_idx] / visual_total * 100
                ).tolist()
                layer_metrics["visual_vs_text_routing"]["visual"] = {
                    f"expert_{i}": round(pct, 2) for i, pct in enumerate(visual_pcts)
                }

            if text_total > 0:
                text_pcts = (self.layer_text_expert_loads[layer_idx] / text_total * 100).tolist()
                layer_metrics["visual_vs_text_routing"]["text"] = {
                    f"expert_{i}": round(pct, 2) for i, pct in enumerate(text_pcts)
                }

            metrics["per_layer"].append(layer_metrics)

        # Compute aggregate metrics (average across all layers)
        all_expert_loads = np.sum(self.layer_expert_loads, axis=0)
        total_load = all_expert_loads.sum()
        if total_load > 0:
            metrics["aggregate"]["expert_load_distribution"] = {
                f"expert_{i}": round(pct, 2)
                for i, pct in enumerate((all_expert_loads / total_load * 100).tolist())
            }

        all_entropies = [e for layer in self.layer_entropies for e in layer]
        if all_entropies:
            metrics["aggregate"]["avg_routing_entropy"] = round(np.mean(all_entropies), 4)

        total_high_conf = sum(self.layer_high_conf_counts)
        total_tokens = sum(self.layer_total_tokens)
        if total_tokens > 0:
            metrics["aggregate"]["high_confidence_fraction"] = round(
                total_high_conf / total_tokens, 4
            )

        # Aggregate visual vs text
        all_visual_loads = np.sum(self.layer_visual_expert_loads, axis=0)
        all_text_loads = np.sum(self.layer_text_expert_loads, axis=0)

        visual_total = all_visual_loads.sum()
        text_total = all_text_loads.sum()

        if visual_total > 0:
            metrics["aggregate"]["visual_routing"] = {
                f"expert_{i}": round(pct, 2)
                for i, pct in enumerate((all_visual_loads / visual_total * 100).tolist())
            }

        if text_total > 0:
            metrics["aggregate"]["text_routing"] = {
                f"expert_{i}": round(pct, 2)
                for i, pct in enumerate((all_text_loads / text_total * 100).tolist())
            }

        return metrics


def save_expert_metrics(
    expert_metrics: dict, output_dir: str, epoch: int, num_visual_tokens: int
) -> str:
    """Write the epoch's routing metrics to JSON and log a readable summary."""
    metrics_dir = os.path.join(output_dir, "expert_metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, f"expert_metrics_epoch_{epoch + 1}.json")
    with open(metrics_path, "w") as f:
        json.dump(expert_metrics, f, indent=2)
    logger.info("✅ Expert metrics saved to %s", metrics_path)

    aggregate = expert_metrics["aggregate"]
    logger.info("\n%s", "=" * 70)
    logger.info("📈 EXPERT UTILISATION METRICS - Epoch %d", epoch + 1)
    logger.info("%s", "=" * 70)

    if "expert_load_distribution" in aggregate:
        logger.info("\n1️⃣  Expert load distribution (aggregate across all layers):")
        for expert, pct in aggregate["expert_load_distribution"].items():
            logger.info("   %s: %s%%", expert, pct)
    if "avg_routing_entropy" in aggregate:
        logger.info("\n2️⃣  Average routing entropy (aggregate):")
        logger.info("   %.4f (lower = more decisive routing)", aggregate["avg_routing_entropy"])
    if "high_confidence_fraction" in aggregate:
        logger.info("\n3️⃣  High-confidence routing fraction (aggregate):")
        logger.info(
            "   %.2f%% of tokens routed with >70%% confidence",
            aggregate["high_confidence_fraction"] * 100,
        )

    logger.info("\n4️⃣  Visual vs text token routing (aggregate):")
    if "visual_routing" in aggregate:
        logger.info("   Visual tokens (positions 0-%d):", num_visual_tokens - 1)
        for expert, pct in aggregate["visual_routing"].items():
            logger.info("      %s: %s%%", expert, pct)
    if "text_routing" in aggregate:
        logger.info("   Text tokens (positions %d+):", num_visual_tokens)
        for expert, pct in aggregate["text_routing"].items():
            logger.info("      %s: %s%%", expert, pct)

    # First, middle and last layer of whatever depth the model actually has.
    layer_count = len(expert_metrics["per_layer"])
    sample_layers = sorted({0, layer_count // 2, layer_count - 1})
    logger.debug("\n📋 Sample per-layer metrics (layers %s):", sample_layers)
    for layer_idx in sample_layers:
        layer_metrics = expert_metrics["per_layer"][layer_idx]
        logger.info("\n   Layer %d:", layer_idx)
        logger.info("      Expert load: %s", layer_metrics["expert_load_distribution"])
        logger.info("      Entropy: %.4f", layer_metrics["avg_routing_entropy"])
        logger.info("      High conf: %.2f%%", layer_metrics["high_confidence_fraction"] * 100)
        routing = layer_metrics["visual_vs_text_routing"]
        if "visual" in routing:
            logger.info("      Visual: %s", routing["visual"])
        if "text" in routing:
            logger.info("      Text: %s", routing["text"])

    logger.info("\n💡 Full per-layer metrics available in %s", metrics_path)
    logger.info("%s\n", "=" * 70)
    return metrics_path
