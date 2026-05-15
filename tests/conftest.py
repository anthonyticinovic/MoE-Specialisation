"""Shared pytest fixtures and global test configuration."""

import os

import pytest
import torch

# Isolate from HuggingFace Hub — tests must never download weights.
os.environ["HF_HUB_OFFLINE"] = "1"
# Deterministic single-threaded ops for reproducibility on CI.
torch.set_num_threads(1)

from models import MistralMoEConfig, MistralMoEForCausalLM  # noqa: E402


@pytest.fixture(scope="session")
def tiny_config() -> MistralMoEConfig:
    """Minimal synthetic config — never loads real Mistral-7B weights."""
    return MistralMoEConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )


@pytest.fixture
def tiny_model(tiny_config) -> MistralMoEForCausalLM:
    torch.manual_seed(0)
    return MistralMoEForCausalLM(tiny_config).to(torch.float32)
