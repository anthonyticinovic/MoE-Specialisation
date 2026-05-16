"""Tests for models.utils.common shared helpers."""

import pytest
import yaml


@pytest.fixture
def valid_config(tmp_path):
    cfg = {
        "paths": {
            "mistral_local_path": "/real/path/Mistral",
            "clip_local_path": "/real/path/clip",
            "image_dir": "/real/path/coco/images",
            "annotations_file": "/real/path/coco/annotations.json",
            "llava_annotations_file": "/real/path/llava.json",
            "llava_image_dir": "/real/path/llava/images",
            "moe_model_path": "/real/path/moe",
            "output_dir": "/real/path/outputs",
        }
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(cfg))
    return str(p)


@pytest.fixture
def placeholder_config(tmp_path):
    cfg = {
        "paths": {
            "mistral_local_path": "YOUR_PATH_HERE/Mistral-7B-v0.3",
            "clip_local_path": "/real/path/clip",
        }
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(cfg))
    return str(p)


class TestLoadConfig:
    def test_loads_valid_config(self, valid_config):
        from models.utils.common import load_config

        cfg = load_config(valid_config)
        assert "paths" in cfg
        assert "mistral_local_path" in cfg["paths"]

    def test_raises_on_missing_file(self, tmp_path):
        from models.utils.common import load_config

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config(str(tmp_path / "nonexistent.yaml"))

    def test_raises_on_placeholder(self, placeholder_config):
        from models.utils.common import load_config

        with pytest.raises(ValueError, match="YOUR_PATH_HERE"):
            load_config(placeholder_config)

    def test_error_message_lists_all_bad_keys(self, tmp_path):
        """All unfilled keys should appear in the error message."""
        cfg = {
            "paths": {
                "key_a": "YOUR_PATH_HERE/a",
                "key_b": "YOUR_PATH_HERE/b",
                "key_c": "/real/path",
            }
        }
        p = tmp_path / "config.yaml"
        p.write_text(yaml.dump(cfg))

        from models.utils.common import load_config

        with pytest.raises(ValueError) as exc_info:
            load_config(str(p))
        msg = str(exc_info.value)
        assert "key_a" in msg
        assert "key_b" in msg
        assert "key_c" not in msg  # real path should not appear


class TestValidateConfig:
    def test_passes_on_real_paths(self):
        from models.utils.common import validate_config

        validate_config({"paths": {"a": "/real/path", "b": "/another"}})

    def test_raises_single_placeholder(self):
        from models.utils.common import validate_config

        with pytest.raises(ValueError):
            validate_config({"paths": {"a": "YOUR_PATH_HERE/foo"}})

    def test_passes_with_no_paths_section(self):
        from models.utils.common import validate_config

        # Config without a paths section should not raise
        validate_config({"training_stage1": {"num_epochs": 5}})


class TestSetSeed:
    def test_produces_same_tensors_with_same_seed(self):
        import torch

        from models.utils.common import set_seed

        set_seed(123)
        t1 = torch.randn(10)
        set_seed(123)
        t2 = torch.randn(10)
        assert torch.equal(t1, t2)

    def test_different_seeds_produce_different_tensors(self):
        import torch

        from models.utils.common import set_seed

        set_seed(1)
        t1 = torch.randn(10)
        set_seed(2)
        t2 = torch.randn(10)
        assert not torch.equal(t1, t2)


class TestRegisterMoEModel:
    def test_registration_is_idempotent(self):
        from models.utils.common import register_moe_model

        # Calling twice must not raise
        register_moe_model()
        register_moe_model()

    def test_auto_model_resolves_after_registration(self):
        from transformers import AutoConfig

        from models import MistralMoEConfig
        from models.utils.common import register_moe_model

        register_moe_model()
        resolved = AutoConfig.for_model("mistral_moe")
        assert isinstance(resolved, MistralMoEConfig)
