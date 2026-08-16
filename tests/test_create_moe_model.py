"""Tests for create_moe_model weight copying.

The script copies the base Mistral FFN weights into *both* experts of every
layer; a regression here silently corrupts the initial state of every
downstream experiment. CPU-only, no real weights, no network — a tiny synthetic
base model is written to a tmp dir and the real script runs against it.

Weight copying is verified by inspecting the saved safetensors directly. The
saved ``config.json`` is verified separately: it must declare
``model_type="mistral_moe"`` so that ``from_pretrained`` re-dispatches to the
MoE classes. An earlier version built the MoE model from the *base*
``MistralConfig``; because ``PretrainedConfig.to_dict()`` reads ``model_type``
from the config class rather than the instance, that wrote
``model_type="mistral"`` and any load without ``trust_remote_code=True``
silently produced a dense ``MistralForCausalLM`` with randomly initialised
FFNs. ``test_config_declares_moe_model_type`` and
``test_reloads_as_moe_without_trust_remote_code`` pin that fix.
"""

import json

import pytest
import torch
from safetensors.torch import load_file
from transformers import MistralConfig, MistralForCausalLM


@pytest.fixture
def tiny_base_model() -> MistralForCausalLM:
    """Tiny real MistralForCausalLM — same dims as the conftest tiny_config."""
    cfg = MistralConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    return MistralForCausalLM(cfg)


@pytest.fixture
def moe_output_dir(tiny_base_model, tmp_path):
    """Run the real create_moe_model end-to-end on a tiny on-disk base model.

    The base model is saved to disk rather than monkeypatching
    ``from_pretrained``: patching it would also intercept the *reload* in
    TestCheckpointReload, because MistralMoEForCausalLM inherits that
    classmethod from MistralForCausalLM.
    """
    import models.utils.create_moe_model as cmm

    base_dir = tmp_path / "base_model"
    tiny_base_model.save_pretrained(str(base_dir))

    out_dir = tmp_path / "moe_model"
    cmm.create_moe_model(str(base_dir), str(out_dir))
    return out_dir


@pytest.fixture
def saved_state_dict(moe_output_dir) -> dict[str, torch.Tensor]:
    return load_file(str(moe_output_dir / "model.safetensors"))


class TestCreateMoEModel:
    def test_ffn_copied_into_both_experts(self, tiny_base_model, saved_state_dict):
        """Every base FFN parameter must be byte-identical in both saved experts."""
        n_layers = len(tiny_base_model.model.layers)
        assert n_layers > 0
        for layer_idx, layer_base in enumerate(tiny_base_model.model.layers):
            base_ffn = dict(layer_base.mlp.named_parameters())
            assert base_ffn, "Base FFN should expose parameters"
            for expert_idx in (0, 1):
                for name, base_param in base_ffn.items():
                    key = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{name}"
                    assert key in saved_state_dict, f"Missing saved weight: {key}"
                    assert torch.equal(base_param, saved_state_dict[key]), (
                        f"Expert {expert_idx} param {name} (layer {layer_idx}) "
                        f"differs from base FFN"
                    )

    def test_shared_weights_preserved(self, tiny_base_model, saved_state_dict):
        """strict=False load_state_dict must still copy non-FFN shared weights."""
        base_sd = tiny_base_model.state_dict()
        for key in ("model.embed_tokens.weight", "lm_head.weight"):
            assert key in saved_state_dict, f"{key} missing from saved MoE checkpoint"
            assert torch.equal(base_sd[key], saved_state_dict[key]), f"{key} not preserved"

    def test_auto_map_patched(self, moe_output_dir):
        """auto_map is the mechanism real loads rely on — it must be patched in."""
        cfg = json.loads((moe_output_dir / "config.json").read_text())
        assert cfg["auto_map"]["AutoConfig"] == "custom_mistral.MistralMoEConfig"
        assert cfg["auto_map"]["AutoModelForCausalLM"] == "custom_mistral.MistralMoEForCausalLM"

    def test_trust_remote_code_sources_copied(self, moe_output_dir):
        """The custom class sources must be copied beside the checkpoint."""
        for fname in ("custom_mistral.py", "moe_layer.py", "__init__.py"):
            assert (moe_output_dir / fname).exists(), f"Missing source file: {fname}"

    def test_config_declares_moe_model_type(self, moe_output_dir):
        """config.json must say mistral_moe, not the base mistral.

        Regression: to_dict() takes model_type from the config *class*, so
        constructing the MoE model from a plain MistralConfig silently wrote
        "mistral" here and produced a checkpoint that loads as a dense model.
        """
        cfg = json.loads((moe_output_dir / "config.json").read_text())
        assert cfg["model_type"] == "mistral_moe", (
            f"config.json declares model_type={cfg['model_type']!r}; a checkpoint "
            "declaring 'mistral' loads as a dense MistralForCausalLM with randomly "
            "initialised FFNs unless trust_remote_code=True is passed."
        )
        assert cfg["architectures"] == ["MistralMoEForCausalLM"]


class TestReproducibility:
    """Stage 0 must be deterministic given the same base model and seed."""

    def _build(self, tiny_base_model, tmp_path, name, seed=42):
        from safetensors.torch import load_file

        import models.utils.create_moe_model as cmm

        base_dir = tmp_path / f"base_{name}"
        tiny_base_model.save_pretrained(str(base_dir))
        out_dir = tmp_path / f"moe_{name}"
        cmm.create_moe_model(str(base_dir), str(out_dir), seed=seed)
        return load_file(str(out_dir / "model.safetensors"))

    def test_same_seed_gives_identical_gates(self, tiny_base_model, tmp_path):
        """The router gate is the only random draw in Stage 0.

        Regression: it was initialised from the unseeded global RNG, so every
        invocation produced a different checkpoint. Hard routing never reads the
        gate, so Stage 2 losses matched and the drift only surfaced downstream.
        """
        first = self._build(tiny_base_model, tmp_path, "a")
        second = self._build(tiny_base_model, tmp_path, "b")

        gate_keys = [key for key in first if ".mlp.gate." in key]
        assert gate_keys, "No router gate weights found in the Stage 0 checkpoint"
        for key in gate_keys:
            assert torch.equal(first[key], second[key]), f"{key} differs between identical builds"

    def test_different_seed_gives_different_gates(self, tiny_base_model, tmp_path):
        """The seed must actually be wired through, not ignored."""
        first = self._build(tiny_base_model, tmp_path, "c", seed=1)
        second = self._build(tiny_base_model, tmp_path, "d", seed=2)

        gate_keys = [key for key in first if ".mlp.gate." in key]
        assert any(not torch.equal(first[key], second[key]) for key in gate_keys), (
            "Changing the seed did not change the gate initialisation"
        )


class TestCheckpointReload:
    """The saved checkpoint must come back as a real MoE model."""

    def test_reloads_as_moe_without_trust_remote_code(self, moe_output_dir):
        """Registered classes alone must be enough to reload as an MoE model.

        This is the load path that silently degraded before the model_type fix:
        no exception, just a dense model with random FFN weights.
        """
        from transformers import AutoModelForCausalLM

        from models.moe_layer import MoELayer
        from models.utils.common import register_moe_model

        register_moe_model()
        model = AutoModelForCausalLM.from_pretrained(str(moe_output_dir), local_files_only=True)

        assert type(model).__name__ == "MistralMoEForCausalLM", (
            f"Reloaded as {type(model).__name__}, expected MistralMoEForCausalLM"
        )
        for idx, layer in enumerate(model.model.layers):
            assert isinstance(layer.mlp, MoELayer), f"Layer {idx} mlp is {type(layer.mlp).__name__}"
            assert len(layer.mlp.experts) == 2

    def test_reloaded_expert_weights_match_disk(self, moe_output_dir, saved_state_dict):
        """Reloading must actually restore the expert weights, not reinitialise them."""
        from transformers import AutoModelForCausalLM

        from models.utils.common import register_moe_model

        register_moe_model()
        model = AutoModelForCausalLM.from_pretrained(str(moe_output_dir), local_files_only=True)

        reloaded = model.state_dict()
        expert_keys = [k for k in saved_state_dict if ".mlp.experts." in k]
        assert expert_keys, "No expert weights found in the saved checkpoint"
        for key in expert_keys:
            assert key in reloaded, f"Expert weight {key} missing after reload"
            assert torch.equal(saved_state_dict[key], reloaded[key]), (
                f"Expert weight {key} changed on reload — weights were reinitialised"
            )
