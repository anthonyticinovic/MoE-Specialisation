"""Smoke tests: verify all public packages and modules import cleanly."""

import importlib


def test_models_package():
    models = importlib.import_module("models")
    assert hasattr(models, "MoELayer")
    assert hasattr(models, "MistralMoEForCausalLM")
    assert hasattr(models, "MistralMoEConfig")
    assert hasattr(models, "VisionLanguageConnector")


def test_data_package():
    data = importlib.import_module("data")
    assert hasattr(data, "COCO_Loader")
    assert hasattr(data, "LLaVA_Loader")


def test_version():
    models = importlib.import_module("models")
    assert isinstance(models.__version__, str)
    assert len(models.__version__) > 0


def test_all_model_submodules():
    for module in [
        "models.moe_layer",
        "models.custom_mistral",
        "models.vl_connector",
        "models.utils.generation",
        "models.utils.create_moe_model",
        "models.utils.common",
    ]:
        assert importlib.import_module(module) is not None


def test_all_data_submodules():
    for module in ["data.COCO_loader", "data.LLaVA_loader"]:
        assert importlib.import_module(module) is not None
