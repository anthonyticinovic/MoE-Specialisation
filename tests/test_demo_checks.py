"""Tests that the demo's invariants actually fire.

A check that can never fail is worse than no check: it looks like coverage and
provides none. Each test here corrupts exactly the property one invariant is
supposed to protect, and asserts that the invariant notices.
"""

import json
import math

import pytest
import torch

from demo import checks


@pytest.fixture
def moe_dir(tmp_path):
    """A real Stage 0 output built from a tiny base model."""
    from transformers import MistralConfig, MistralForCausalLM

    from models.utils.create_moe_model import create_moe_model

    config = MistralConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    base = tmp_path / "base"
    MistralForCausalLM(config).save_pretrained(str(base))

    out = tmp_path / "moe"
    create_moe_model(str(base), str(out))
    return out


class TestStage0Checks:
    def test_identical_experts_passes_on_real_output(self, moe_dir):
        assert checks.check_experts_identical_at_init(moe_dir).passed

    def test_identical_experts_fails_when_experts_differ(self, moe_dir, tmp_path):
        """Perturb one expert; the check must notice they no longer match."""
        from safetensors.torch import load_file, save_file

        weights = moe_dir / "model.safetensors"
        tensors = load_file(str(weights))
        key = next(k for k in tensors if ".mlp.experts.1." in k)
        tensors[key] = tensors[key] + 1.0
        save_file(tensors, str(weights))

        result = checks.check_experts_identical_at_init(moe_dir)
        assert not result.passed
        assert "differs" in result.detail

    def test_config_check_fails_on_dense_model_type(self, moe_dir):
        """The exact regression that made checkpoints load as dense models."""
        config_path = moe_dir / "config.json"
        config = json.loads(config_path.read_text())
        config["model_type"] = "mistral"
        config_path.write_text(json.dumps(config))

        result = checks.check_config_declares_moe(moe_dir)
        assert not result.passed
        assert "dense" in result.detail

    def test_hard_routing_check_passes(self, moe_dir):
        assert checks.check_hard_routing_is_exact(moe_dir).passed


class TestStage2Checks:
    def _write_checkpoint(self, path, state):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": state}, path)

    def test_divergence_fails_when_experts_never_trained(self, moe_dir, tmp_path):
        """Experts start identical — if training is a no-op they stay identical."""
        from safetensors.torch import load_file

        state = load_file(str(moe_dir / "model.safetensors"))
        ckpt = tmp_path / "stage2.pth"
        self._write_checkpoint(ckpt, state)

        result = checks.check_experts_diverged(moe_dir, ckpt)
        assert not result.passed
        assert "bit-identical" in result.detail

    def test_divergence_passes_once_experts_differ(self, moe_dir, tmp_path):
        from safetensors.torch import load_file

        state = load_file(str(moe_dir / "model.safetensors"))
        key = next(k for k in state if ".mlp.experts.0." in k)
        state[key] = state[key] + 0.01
        ckpt = tmp_path / "stage2.pth"
        self._write_checkpoint(ckpt, state)

        assert checks.check_experts_diverged(moe_dir, ckpt).passed

    def test_frozen_weights_check_catches_a_moved_attention_weight(self, moe_dir, tmp_path):
        """Stage 2 must not touch attention; simulate a freezing bug."""
        from safetensors.torch import load_file

        state = load_file(str(moe_dir / "model.safetensors"))
        key = next(k for k in state if "self_attn" in k)
        state[key] = state[key] + 0.5
        ckpt = tmp_path / "stage2.pth"
        self._write_checkpoint(ckpt, state)

        result = checks.check_only_experts_trained(moe_dir, ckpt)
        assert not result.passed
        assert "frozen tensors changed" in result.detail


class TestMetricChecks:
    def _metrics(self, entropy):
        return {
            "per_layer": [
                {
                    "layer": 0,
                    "avg_routing_entropy": entropy,
                    "expert_load_distribution": {"expert_0": 40.0, "expert_1": 60.0},
                }
            ],
            "aggregate": {"avg_routing_entropy": entropy},
        }

    def test_entropy_bounds_pass_within_ln2(self):
        assert checks.check_routing_entropy_bounds(self._metrics(0.5)).passed

    def test_entropy_bounds_fail_above_ln2(self):
        """The summed-instead-of-averaged signature."""
        result = checks.check_routing_entropy_bounds(self._metrics(2 * math.log(2)))
        assert not result.passed
        assert "ln(2)" in result.detail

    def test_loads_must_sum_to_100(self):
        broken = self._metrics(0.5)
        broken["per_layer"][0]["expert_load_distribution"] = {"expert_0": 40.0, "expert_1": 40.0}
        result = checks.check_expert_loads_sum_to_100(broken)
        assert not result.passed

    def test_layer_count_mismatch_is_caught(self):
        result = checks.check_all_layers_reported(self._metrics(0.5), expected_layers=32)
        assert not result.passed
        assert "32" in result.detail


class TestNumericalHealth:
    def test_nan_in_checkpoint_is_caught(self, tmp_path):
        runs = tmp_path / "runs"
        runs.mkdir()
        torch.save({"model_state_dict": {"w": torch.tensor([1.0, float("nan")])}}, runs / "a.pth")

        result = checks.check_checkpoints_finite(runs)
        assert not result.passed
        assert "non-finite" in result.detail

    def test_finite_checkpoint_passes(self, tmp_path):
        runs = tmp_path / "runs"
        runs.mkdir()
        torch.save({"model_state_dict": {"w": torch.tensor([1.0, 2.0])}}, runs / "a.pth")
        assert checks.check_checkpoints_finite(runs).passed

    def test_nan_loss_is_caught(self, tmp_path):
        runs = tmp_path / "runs"
        runs.mkdir()
        (runs / "m.json").write_text(json.dumps({"train_loss": [1.0, float("nan")]}))

        result = checks.check_losses_finite(runs, {"Stage X": "m.json"})
        assert not result.passed


class TestSkipBehaviour:
    def test_missing_artifacts_skip_rather_than_fail(self, tmp_path):
        """A partial run (--stages 0 1) must not report spurious failures."""
        result = checks.check_experts_diverged(tmp_path, tmp_path / "absent.pth")
        assert result.skipped
        assert result.passed
