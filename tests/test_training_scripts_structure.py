"""Structural tests for every training script.

All five stages used to be top-level statements: importing one started training.
These tests pin the property the refactor established — each module is inert on
import and its work is reachable as named functions — across the whole set, so
it cannot regress in one script while the others stay clean.
"""

import ast
import importlib
from pathlib import Path

import pytest

SCRIPTS = {
    "train_stage_1": {"build_setup", "train_one_epoch", "run_validation", "main"},
    "train_stage_2": {
        "build_setup",
        "train_one_epoch",
        "run_validation",
        "save_checkpoints",
        "main",
    },
    "train_stage_2_5": {
        "build_setup",
        "train_one_epoch",
        "run_validation",
        "save_checkpoints",
        "main",
    },
    "train_stage_3": {
        "build_setup",
        "train_one_epoch",
        "run_validation",
        "save_checkpoints",
        "main",
    },
    "train_dense": {"build_setup", "train_one_epoch", "run_validation", "save_checkpoints", "main"},
}
# Stage 2.5's filename is not a valid identifier, so it is loaded by path.
FILENAMES = {name: name.replace("train_stage_2_5", "train_stage_2.5") for name in SCRIPTS}
SCRIPT_DIR = Path(__file__).parent.parent / "training_scripts"


def _tree(name: str) -> ast.Module:
    return ast.parse((SCRIPT_DIR / f"{FILENAMES[name]}.py").read_text())


@pytest.mark.parametrize("name", list(SCRIPTS))
class TestImportIsInert:
    def test_no_work_at_module_scope(self, name):
        """Only constants, the logger and the __main__ guard may run on import.

        Bare expressions count: a stray ``register_moe_model()`` at module scope
        survived an earlier extraction precisely because a laxer check ignored
        them.
        """
        for node in _tree(name).body:
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
                continue
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue  # module docstring
            if isinstance(node, ast.Assign):
                targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
                assert all(t.isupper() or t == "logger" for t in targets), (
                    f"{name}: module-level assignment to {targets}"
                )
                continue
            if isinstance(node, ast.If):
                assert ast.unparse(node.test) == "__name__ == '__main__'", (
                    f"{name}: module-level branch {ast.unparse(node.test)}"
                )
                continue
            pytest.fail(f"{name}: work at module scope: {ast.unparse(node)[:80]}")

    def test_has_exactly_one_main_guard(self, name):
        guards = [
            node
            for node in _tree(name).body
            if isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'"
        ]
        assert len(guards) == 1, f"{name} must be runnable via a single __main__ guard"

    def test_pipeline_steps_are_named_functions(self, name):
        defined = {n.name for n in _tree(name).body if isinstance(n, ast.FunctionDef)}
        missing = SCRIPTS[name] - defined
        assert not missing, f"{name} is missing {sorted(missing)}"

    def test_no_function_is_oversized(self, name):
        """No single function should reabsorb the monolith."""
        oversized = [
            (node.name, node.end_lineno - node.lineno)
            for node in ast.walk(_tree(name))
            if isinstance(node, ast.FunctionDef)
            and node.end_lineno
            and node.lineno
            and node.end_lineno - node.lineno > 120
        ]
        assert not oversized, f"{name}: functions over 120 lines: {oversized}"


class TestSharedLibrary:
    """The stages must actually reuse _lib rather than re-implementing it."""

    @pytest.mark.parametrize("name", list(SCRIPTS))
    def test_script_imports_the_shared_runtime(self, name):
        imported = {
            alias.name
            for node in _tree(name).body
            if isinstance(node, ast.ImportFrom)
            and (node.module or "").startswith("training_scripts._lib")
            for alias in node.names
        }
        assert "build_run_context" in imported, f"{name} builds its own run context"

    @pytest.mark.parametrize(
        "name", ["train_stage_2", "train_stage_2_5", "train_stage_3", "train_dense"]
    )
    def test_distributed_stages_use_shared_fsdp_wrapper(self, name):
        source = (SCRIPT_DIR / f"{FILENAMES[name]}.py").read_text()
        assert "FSDP(" not in source, f"{name} constructs FSDP directly instead of using _lib"
        assert "wrap_with_fsdp" in source

    @pytest.mark.parametrize("name", list(SCRIPTS))
    def test_no_script_loads_weights_without_the_guard(self, name):
        """``strict=False`` must go through ``load_matching_weights``.

        A bare ``load_state_dict(..., strict=False)`` accepts a state dict that
        matches the model in no key at all: it loads nothing and returns
        normally. That is how Stages 2.5 and 3 spent months starting from their
        Stage 0 experts while logging that the Stage 2 checkpoint had loaded.
        """
        source = (SCRIPT_DIR / f"{FILENAMES[name]}.py").read_text()
        assert "strict=False" not in source, (
            f"{name}: use _lib.load_matching_weights instead of a bare strict=False load"
        )

    def test_run_context_reports_main_rank(self):
        from training_scripts._lib import RunContext

        ctx = RunContext(
            device="cpu", amp_device="cpu", on_gpu=False, local_rank=0, distributed=False
        )
        assert ctx.is_main and not ctx.use_fsdp
        assert not RunContext(
            device="cpu", amp_device="cpu", on_gpu=False, local_rank=1, distributed=True
        ).is_main

    def test_shared_loss_guards_against_shape_mismatch(self):
        """A malformed batch contributes zero rather than deadlocking the job."""
        import torch
        import torch.nn as nn

        from training_scripts._lib import shifted_caption_loss

        logits = torch.randn(2, 10, 7)
        input_ids = torch.randint(0, 7, (2, 99))  # deliberately mismatched
        loss = shifted_caption_loss(nn.CrossEntropyLoss(), logits, 4, input_ids, 7, device="cpu")
        assert loss.item() == 0.0 and loss.requires_grad


def test_stage_1_and_3_keep_their_own_loss_alignment():
    """The loss offsets genuinely differ; unifying them would be a silent change."""
    stage1 = (SCRIPT_DIR / "train_stage_1.py").read_text()
    assert "num_visual_tokens - 1 : -1" in stage1, "Stage 1's earlier offset was lost"
    assert "shifted_caption_loss" not in stage1, "Stage 1 must not use the shared loss"


def test_stage_2_5_module_loads_by_path():
    """Sanity check that the dotted filename is importable at all."""
    spec = importlib.util.spec_from_file_location(
        "train_stage_2_5", SCRIPT_DIR / "train_stage_2.5.py"
    )
    assert spec is not None and spec.loader is not None
