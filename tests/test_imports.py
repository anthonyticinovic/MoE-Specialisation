"""Smoke tests: the public packages import cleanly, and nothing in them is dead.

The import checks alone are weak — they prove a module parses, not that anything
uses it. `models/utils/generation.py` passed them for months while being both
unreferenced and, had it ever been called, broken. `TestNoDeadModules` is the
check that would have caught it.
"""

import ast
import importlib
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
CORE_MODULES = sorted(
    path
    for package in ("models", "data")
    for path in (REPO / package).rglob("*.py")
    if path.name != "__init__.py"
)


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


@pytest.mark.parametrize("path", CORE_MODULES, ids=lambda p: p.stem)
def test_core_module_imports_cleanly(path):
    dotted = path.relative_to(REPO).with_suffix("").as_posix().replace("/", ".")
    assert importlib.import_module(dotted) is not None


class TestNoDeadModules:
    """Every module in the maintained core must have a real consumer.

    `models/` is the first thing a reviewer opens, so an unused module there
    costs more than the same module anywhere else — especially one that would
    raise on the first call, as the deleted `CaptionGenerator` would have.
    """

    @pytest.mark.parametrize("path", CORE_MODULES, ids=lambda p: p.stem)
    def test_module_is_imported_somewhere(self, path):
        stem = path.stem
        consumers = []

        for candidate in REPO.rglob("*.py"):
            if candidate == path or ".venv" in candidate.parts:
                continue
            # This file imports every core module by construction, so counting
            # it as a consumer would restore exactly the blind spot above.
            if candidate.name == "test_imports.py":
                continue
            try:
                tree = ast.parse(candidate.read_text())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and stem in (node.module or "").split("."):
                    consumers.append(candidate.name)
                elif isinstance(node, ast.Import):
                    if any(stem in alias.name.split(".") for alias in node.names):
                        consumers.append(candidate.name)

        assert consumers, (
            f"{path.relative_to(REPO)} is imported by nothing. Either wire it "
            "into the code that needs it, or delete it — an unused module in "
            "the maintained core reads as abandoned work."
        )
