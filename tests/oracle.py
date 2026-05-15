"""Behavioural oracle for the conservative-refactor safety net.

Builds a tiny synthetic MoE Mistral model (never the real 7B weights) and runs
deterministic hard-routing (Stage 2 style) and soft-routing (Stage 2.5 style)
forward+backward passes, returning loss and total gradient norm.

These numbers must stay bit-identical across the formatting / helper-extraction /
logging phases. Any drift means a refactor changed numerics — stop and investigate.
"""

from __future__ import annotations

import torch

from models import MistralMoEConfig, MistralMoEForCausalLM

SEED = 1234


def _tiny_config() -> MistralMoEConfig:
    return MistralMoEConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )


def _build_model() -> MistralMoEForCausalLM:
    torch.manual_seed(SEED)
    model = MistralMoEForCausalLM(_tiny_config())
    model.eval()
    return model.to(torch.float32)


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().norm() ** 2)
    return total**0.5


def _batch(batch_size: int = 2, seq_len: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(SEED)
    input_ids = torch.randint(0, 128, (batch_size, seq_len), generator=g)
    labels = input_ids.clone()
    return input_ids, labels


def run_hard_routing() -> dict[str, float]:
    """Stage 2 style: deterministic mask, first half = vision (expert 0)."""
    model = _build_model()
    input_ids, labels = _batch()
    batch_size, seq_len = input_ids.shape

    routing_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    routing_mask[:, : seq_len // 2] = 0  # 0 = vision expert, 1 = text expert
    for layer in model.model.layers:
        layer.mlp.routing_mode = "hard"
        layer.mlp.routing_mask = routing_mask

    model.zero_grad()
    out = model(input_ids=input_ids, labels=labels)
    out.loss.backward()
    return {"loss": float(out.loss.detach()), "grad_norm": _grad_norm(model)}


def run_soft_routing() -> dict[str, float]:
    """Stage 2.5 style: learned gate + Gumbel-STE, training mode."""
    model = _build_model()
    input_ids, labels = _batch()

    for layer in model.model.layers:
        layer.mlp.routing_mode = "soft"
    model.train()

    torch.manual_seed(SEED)  # determinism for Gumbel noise
    model.zero_grad()
    out = model(input_ids=input_ids, labels=labels)
    out.loss.backward()
    return {"loss": float(out.loss.detach()), "grad_norm": _grad_norm(model)}


def collect() -> dict[str, dict[str, float]]:
    return {"hard": run_hard_routing(), "soft": run_soft_routing()}


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps(collect(), indent=2))
    sys.exit(0)
