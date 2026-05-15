"""Tests for VisionLanguageConnector."""

import torch

from models.vl_connector import VisionLanguageConnector


def test_output_shape_default_dims():
    connector = VisionLanguageConnector()
    x = torch.randn(4, 1024)
    out = connector(x)
    assert out.shape == (4, 4096)


def test_output_shape_custom_dims():
    connector = VisionLanguageConnector(clip_hidden_size=512, llm_hidden_size=256)
    x = torch.randn(3, 512)
    out = connector(x)
    assert out.shape == (3, 256)


def test_batch_dimension_preserved():
    connector = VisionLanguageConnector(clip_hidden_size=64, llm_hidden_size=128)
    for batch in [1, 8, 32]:
        out = connector(torch.randn(batch, 64))
        assert out.shape == (batch, 128)


def test_dropout_active_in_training():
    """Dropout must produce different outputs on two identical train-mode passes."""
    torch.manual_seed(0)
    connector = VisionLanguageConnector(clip_hidden_size=64, llm_hidden_size=128, dropout_rate=0.5)
    connector.train()
    x = torch.randn(16, 64)
    out1 = connector(x)
    out2 = connector(x)
    assert not torch.equal(out1, out2), "Dropout should introduce stochasticity in train mode"


def test_dropout_inactive_in_eval():
    """Eval mode must be deterministic (dropout disabled)."""
    connector = VisionLanguageConnector(clip_hidden_size=64, llm_hidden_size=128, dropout_rate=0.5)
    connector.eval()
    x = torch.randn(8, 64)
    assert torch.equal(connector(x), connector(x))


def test_gradient_flows():
    connector = VisionLanguageConnector(clip_hidden_size=32, llm_hidden_size=64)
    x = torch.randn(4, 32, requires_grad=True)
    out = connector(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
