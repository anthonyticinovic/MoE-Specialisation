"""Tests for create_moe_model weight copying.

The script copies the base Mistral FFN weights into *both* experts of every
layer; a regression here silently corrupts the initial state of every
downstream experiment. CPU-only, no real weights, no network — the heavy 7B
load is monkeypatched to a tiny synthetic base model.

The saved checkpoint is inspected directly (safetensors), not reloaded through
``from_pretrained``: that path re-dispatches on config.json's ``model_type``
(left as the base ``mistral`` because create_moe_model builds the MoE model
from the base config object) and is exercised by the trust_remote_code /
registration tests elsewhere. Here we only care that the bytes on disk are
correct.
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
def moe_output_dir(tiny_base_model, tmp_path, monkeypatch):
    """Run the real create_moe_model with the 7B load stubbed out."""
    import models.utils.create_moe_model as cmm

    monkeypatch.setattr(
        cmm.MistralForCausalLM,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: tiny_base_model),
    )
    out_dir = tmp_path / "moe_model"
    cmm.create_moe_model("ignored-base-path", str(out_dir))
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
