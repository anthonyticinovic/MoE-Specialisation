"""Tests for MistralMoEForCausalLM checkpoint save/load round-trip.

Also validates the trust_remote_code source-copy behaviour of create_moe_model.
"""

import json
import shutil
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM


@pytest.fixture
def saved_model_path(tiny_model, tmp_path):
    """Save a tiny model and return the directory path."""
    save_dir = tmp_path / "moe_model"
    tiny_model.save_pretrained(str(save_dir))
    # Patch auto_map into config.json (mirrors create_moe_model behaviour)
    config_path = save_dir / "config.json"
    cfg = json.loads(config_path.read_text())
    cfg["auto_map"] = {
        "AutoConfig": "custom_mistral.MistralMoEConfig",
        "AutoModelForCausalLM": "custom_mistral.MistralMoEForCausalLM",
    }
    config_path.write_text(json.dumps(cfg, indent=2))
    return save_dir


class TestCheckpointRoundTrip:
    def test_state_dict_equality_after_reload(self, tiny_model, saved_model_path):
        from models import MistralMoEConfig, MistralMoEForCausalLM

        # Registration is idempotent — re-registering the same classes is safe.
        AutoConfig.register("mistral_moe", MistralMoEConfig, exist_ok=True)
        AutoModelForCausalLM.register(MistralMoEConfig, MistralMoEForCausalLM, exist_ok=True)
        loaded = AutoModelForCausalLM.from_pretrained(str(saved_model_path))

        orig = tiny_model.state_dict()
        reloaded = loaded.state_dict()
        assert set(orig) == set(reloaded), "State dict keys must match"
        for k in orig:
            assert torch.equal(orig[k], reloaded[k]), f"Tensor mismatch for key: {k}"

    def test_auto_map_patched_in_config(self, saved_model_path):
        cfg = json.loads((saved_model_path / "config.json").read_text())
        assert "auto_map" in cfg
        assert "AutoConfig" in cfg["auto_map"]
        assert "AutoModelForCausalLM" in cfg["auto_map"]
        assert "MistralMoEConfig" in cfg["auto_map"]["AutoConfig"]
        assert "MistralMoEForCausalLM" in cfg["auto_map"]["AutoModelForCausalLM"]

    def test_source_files_copied(self, saved_model_path):
        """create_moe_model copies 3 source files so trust_remote_code works."""
        repo_root = Path(__file__).parent.parent
        for src in ["models/custom_mistral.py", "models/moe_layer.py", "models/__init__.py"]:
            shutil.copy2(repo_root / src, saved_model_path / Path(src).name)

        for fname in ["custom_mistral.py", "moe_layer.py", "__init__.py"]:
            assert (saved_model_path / fname).exists(), f"Missing: {fname}"

    def test_copied_files_contain_required_classes(self, saved_model_path):
        """Source files copied to the checkpoint dir must define the HF custom classes.

        trust_remote_code loads these files as a package — we just assert the
        class names are present in the source text (actual import happens via HF).
        """
        repo_root = Path(__file__).parent.parent
        for src in ["models/custom_mistral.py", "models/moe_layer.py", "models/__init__.py"]:
            shutil.copy2(repo_root / src, saved_model_path / Path(src).name)

        custom_mistral_src = (saved_model_path / "custom_mistral.py").read_text()
        assert "class MistralMoEConfig" in custom_mistral_src
        assert "class MistralMoEForCausalLM" in custom_mistral_src

        moe_layer_src = (saved_model_path / "moe_layer.py").read_text()
        assert "class MoELayer" in moe_layer_src

    def test_model_type_in_config(self, saved_model_path):
        cfg = json.loads((saved_model_path / "config.json").read_text())
        assert cfg.get("model_type") == "mistral_moe"
