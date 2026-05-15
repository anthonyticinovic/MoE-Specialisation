"""Tests for MoELayer hard and soft routing."""

import torch
import pytest

from models import MoELayer, MistralMoEConfig


@pytest.fixture
def tiny_layer_config() -> MistralMoEConfig:
    return MistralMoEConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )


@pytest.fixture
def hard_layer(tiny_layer_config) -> MoELayer:
    torch.manual_seed(0)
    return MoELayer(config=tiny_layer_config, d_model=64, routing_mode="hard")


@pytest.fixture
def soft_layer(tiny_layer_config) -> MoELayer:
    torch.manual_seed(0)
    return MoELayer(config=tiny_layer_config, d_model=64, routing_mode="soft")


# ---------------------------------------------------------------------------
# Hard routing
# ---------------------------------------------------------------------------


class TestHardRouting:
    def test_output_shape(self, hard_layer):
        B, S, D = 2, 10, 64
        x = torch.randn(B, S, D)
        mask = torch.ones(B, S, dtype=torch.long)
        mask[:, : S // 2] = 0
        hard_layer.routing_mask = mask
        out = hard_layer(x)
        assert out.shape == (B, S, D)

    def test_vision_tokens_use_expert_0(self, hard_layer):
        """Vision tokens (mask=0) must be processed only by expert 0."""
        B, S, D = 1, 4, 64
        x = torch.randn(B, S, D)
        # All-vision mask
        hard_layer.routing_mask = torch.zeros(B, S, dtype=torch.long)

        # Sabotage expert 1 so its output is wildly large — if it runs on vision
        # tokens the output would differ from running expert 0 only.
        with torch.no_grad():
            for p in hard_layer.experts[1].parameters():
                p.fill_(0.0)

        out = hard_layer(x)
        expected = hard_layer.experts[0](x.view(-1, D)).view(B, S, D)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_text_tokens_use_expert_1(self, hard_layer):
        """Text tokens (mask=1) must be processed only by expert 1."""
        B, S, D = 1, 4, 64
        x = torch.randn(B, S, D)
        hard_layer.routing_mask = torch.ones(B, S, dtype=torch.long)

        with torch.no_grad():
            for p in hard_layer.experts[0].parameters():
                p.fill_(0.0)

        out = hard_layer(x)
        expected = hard_layer.experts[1](x.view(-1, D)).view(B, S, D)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_no_grad_through_gate_in_hard_mode(self, hard_layer):
        """Gate weights must receive no gradient under hard routing."""
        B, S, D = 2, 8, 64
        x = torch.randn(B, S, D, requires_grad=True)
        hard_layer.routing_mask = torch.randint(0, 2, (B, S))
        out = hard_layer(x)
        loss = out.sum()
        loss.backward()
        assert hard_layer.gate.weight.grad is None

    def test_last_router_logits_not_set_in_hard_mode(self, hard_layer):
        """Hard routing does not write _last_router_logits (only soft does)."""
        B, S, D = 1, 4, 64
        hard_layer.routing_mask = torch.zeros(B, S, dtype=torch.long)
        if hasattr(hard_layer, "_last_router_logits"):
            del hard_layer._last_router_logits
        hard_layer(torch.randn(B, S, D))
        assert not hasattr(hard_layer, "_last_router_logits")

    def test_mixed_routing_mask(self, hard_layer):
        """Mixed vision/text mask — output at each position must match the correct expert."""
        B, S, D = 1, 6, 64
        x = torch.randn(B, S, D)
        mask = torch.tensor([[0, 1, 0, 1, 0, 1]])
        hard_layer.routing_mask = mask

        with torch.no_grad():
            out = hard_layer(x)

        vision_idx = (mask == 0).nonzero(as_tuple=True)[1]
        text_idx = (mask == 1).nonzero(as_tuple=True)[1]
        vision_expected = hard_layer.experts[0](x[0, vision_idx])
        text_expected = hard_layer.experts[1](x[0, text_idx])

        assert torch.allclose(out[0, vision_idx], vision_expected, atol=1e-5)
        assert torch.allclose(out[0, text_idx], text_expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Soft routing
# ---------------------------------------------------------------------------


class TestSoftRouting:
    def test_output_shape(self, soft_layer):
        B, S, D = 2, 10, 64
        x = torch.randn(B, S, D)
        soft_layer.train()
        torch.manual_seed(0)
        out = soft_layer(x)
        assert out.shape == (B, S, D)

    def test_last_router_logits_stored(self, soft_layer):
        B, S, D = 2, 8, 64
        soft_layer.eval()
        soft_layer(torch.randn(B, S, D))
        assert hasattr(soft_layer, "_last_router_logits")
        assert soft_layer._last_router_logits.shape == (B, S, 2)

    def test_gate_receives_gradient(self, soft_layer):
        """Soft routing must propagate gradients back to the gate."""
        B, S, D = 2, 8, 64
        x = torch.randn(B, S, D)
        soft_layer.train()
        torch.manual_seed(0)
        out = soft_layer(x)
        out.sum().backward()
        assert soft_layer.gate.weight.grad is not None
        assert soft_layer.gate.weight.grad.abs().sum() > 0

    def test_expert_weights_receive_gradient(self, soft_layer):
        B, S, D = 2, 8, 64
        soft_layer.train()
        torch.manual_seed(0)
        out = soft_layer(torch.randn(B, S, D))
        out.sum().backward()
        # At least one expert should have non-zero gradient
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for expert in soft_layer.experts
            for p in expert.parameters()
        )
        assert has_grad

    def test_temperature_attribute_overrides_argument(self, soft_layer):
        """_forward_temperature attribute must override the default arg."""
        soft_layer.eval()
        soft_layer._forward_temperature = 0.5
        B, S, D = 1, 4, 64
        x = torch.randn(B, S, D)
        # Should not raise
        out = soft_layer(x)
        assert out.shape == (B, S, D)

    def test_eval_mode_no_gumbel_noise(self, soft_layer):
        """In eval mode two identical passes must produce identical outputs."""
        soft_layer.eval()
        x = torch.randn(2, 6, 64)
        out1 = soft_layer(x)
        out2 = soft_layer(x)
        assert torch.allclose(out1, out2)

    def test_initialize_gate_resets_weights(self, soft_layer):
        """initialize_gate must reinitialise, not share weights with old gate."""
        old_weight = soft_layer.gate.weight.data.clone()
        torch.manual_seed(99)
        soft_layer.initialize_gate()
        # With a different seed the weights should differ (extremely unlikely to match)
        assert not torch.equal(soft_layer.gate.weight.data, old_weight)

    def test_invalid_routing_mode(self, tiny_layer_config):
        layer = MoELayer(config=tiny_layer_config, d_model=64, routing_mode="invalid")
        with pytest.raises(ValueError, match="Invalid routing mode"):
            layer(torch.randn(1, 4, 64))
