"""Structural tests for the refactored Stage 3 script.

Stage 3 used to be ~950 lines of top-level statements: importing it started
training. These tests pin the property that refactor established — the module
is inert on import and its work is reachable as named functions — so it cannot
silently regress.
"""

import ast
import inspect
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "training_scripts" / "train_stage_3.py"


@pytest.fixture(scope="module")
def module():
    import training_scripts.train_stage_3 as stage3

    return stage3


@pytest.fixture(scope="module")
def tree():
    return ast.parse(SCRIPT.read_text())


class TestImportIsInert:
    def test_module_imports_without_running_anything(self, module):
        """Importing must not load models, read config or start training."""
        assert callable(module.main)

    def test_no_work_at_module_scope(self, tree):
        """Only constants, the logger and the __main__ guard may run on import.

        Bare expressions count: a stray ``register_moe_model()`` at module scope
        survived an earlier extraction precisely because a laxer check ignored
        them.
        """
        allowed_assignments = {"logger", "DIST_TIMEOUT", "MAX_VAL_BATCHES", "NUM_EXPERTS"}

        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
                continue
            # Module docstring.
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue
            if isinstance(node, ast.Assign):
                targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
                assert targets <= allowed_assignments, f"Unexpected module-level assign: {targets}"
                continue
            if isinstance(node, ast.If):
                assert ast.unparse(node.test) == "__name__ == '__main__'", (
                    f"Unexpected module-level branch: {ast.unparse(node.test)}"
                )
                continue
            pytest.fail(f"Work at module scope: {ast.unparse(node)[:80]}")

    def test_has_main_guard(self, tree):
        guards = [
            node
            for node in tree.body
            if isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'"
        ]
        assert len(guards) == 1, "Script must be runnable via a single __main__ guard"


class TestDecomposition:
    EXPECTED = [
        "build_backbones",
        "build_llm",
        "configure_trainable_parameters",
        "wrap_with_fsdp",
        "load_stage2_experts",
        "load_stage1_connector",
        "build_datasets",
        "build_dataloaders",
        "maybe_resume",
        "train_one_epoch",
        "run_validation",
        "save_checkpoints",
        "main",
    ]

    @pytest.mark.parametrize("name", EXPECTED)
    def test_pipeline_step_is_a_named_function(self, module, name):
        assert callable(getattr(module, name)), f"{name} missing"

    def test_no_function_is_oversized(self, tree):
        """No single function should reabsorb the monolith."""
        oversized = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.end_lineno and node.lineno:
                length = node.end_lineno - node.lineno
                if length > 120:
                    oversized.append((node.name, length))
        assert not oversized, f"Functions over 120 lines: {oversized}"

    def test_routing_metrics_live_in_their_own_module(self, module):
        """The tracker was extracted to training_scripts/_lib."""
        assert module.ExpertUsageTracker.__module__ == "training_scripts._lib.expert_metrics"
        assert module.save_expert_metrics.__module__ == "training_scripts._lib.expert_metrics"


class TestSignatures:
    def test_train_one_epoch_returns_a_loss(self, module):
        """Contract check: the epoch function reports its own average loss."""
        signature = inspect.signature(module.train_one_epoch)
        assert list(signature.parameters) == ["setup", "ctx", "epoch", "num_epochs"]

    def test_run_validation_returns_loss_metrics_and_steps(self, module):
        signature = inspect.signature(module.run_validation)
        assert list(signature.parameters) == ["setup", "ctx"]

    def test_run_context_reports_main_rank(self, module):
        ctx = module.RunContext(device="cpu", amp_device="cpu", use_fsdp=False, local_rank=0)
        assert ctx.is_main
        assert not module.RunContext(
            device="cpu", amp_device="cpu", use_fsdp=False, local_rank=1
        ).is_main
