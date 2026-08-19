"""Unit tests for the shared analysis_scripts._lib helpers."""

import ast
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from analysis_scripts._lib import (
    compute_cosine_similarity_matrix,
    load_analysis_config,
    majority_vote_expert,
)


class TestCosineSimilarityMatrix:
    def test_identity_diagonal_and_symmetry(self):
        reps = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])]
        m = compute_cosine_similarity_matrix(reps)
        assert m.shape == (3, 3)
        for i in range(3):
            assert m[i, i] == 1.0
        assert np.allclose(m, m.T, atol=1e-6)

    def test_orthogonal_and_parallel(self):
        m = compute_cosine_similarity_matrix(
            [np.array([1.0, 0.0]), np.array([0.0, 5.0]), np.array([2.0, 0.0])]
        )
        assert m[0, 1] == pytest.approx(0.0, abs=1e-6)
        assert m[0, 2] == pytest.approx(1.0, abs=1e-6)


class TestMajorityVoteExpert:
    def test_empty_is_unknown(self):
        assert majority_vote_expert([]) == ("unknown", 0.0)

    def test_decisive_winner(self):
        label, frac = majority_vote_expert([1, 1, 1, 0], confidence_threshold=0.6)
        assert label == "Expert 1"
        assert frac == pytest.approx(0.75)

    def test_below_threshold_is_mixed(self):
        label, frac = majority_vote_expert([0, 0, 1, 1, 1], confidence_threshold=0.9)
        assert label == "mixed"
        assert frac == pytest.approx(0.6)


class TestLoadAnalysisConfig:
    def test_applies_defaults_and_returns(self, tmp_path):
        p = tmp_path / "c.json"
        p.write_text(json.dumps({"concepts": ["cat"]}))
        cfg = load_analysis_config(
            str(p), required_fields=["concepts"], defaults={"layers": [31], "pooling": "mean"}
        )
        assert cfg["concepts"] == ["cat"]
        assert cfg["layers"] == [31]
        assert cfg["pooling"] == "mean"

    def test_missing_required_field_raises(self, tmp_path):
        p = tmp_path / "c.json"
        p.write_text(json.dumps({"foo": 1}))
        with pytest.raises(ValueError, match="missing required field"):
            load_analysis_config(str(p), required_fields=["concepts"])


def test_lib_public_api_imports():
    import analysis_scripts._lib as lib

    for name in lib.__all__:
        assert hasattr(lib, name), f"_lib missing exported name: {name}"


ANALYSIS_DIR = Path(__file__).parent.parent / "analysis_scripts"
ANALYSIS_MODULES = sorted(ANALYSIS_DIR.rglob("*.py"))


class TestLoggingMigration:
    """The analysis scripts log like the rest of the repo, not via print().

    Before this migration the core used ``logging`` while these 838 call sites
    used ``print`` — the inconsistency was more noticeable than either choice
    would have been alone.
    """

    @pytest.mark.parametrize("path", ANALYSIS_MODULES, ids=lambda p: p.name)
    def test_no_print_calls(self, path):
        calls = [
            node
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ]
        assert not calls, (
            f"{path.name}: {len(calls)} print() call(s) at lines "
            f"{[c.lineno for c in calls]} — use the module logger"
        )

    @pytest.mark.parametrize("path", ANALYSIS_MODULES, ids=lambda p: p.name)
    def test_modules_that_log_define_a_logger(self, path):
        source = path.read_text()
        if "logger." not in source:
            return
        assert "logger = logging.getLogger(__name__)" in source, (
            f"{path.name} logs without defining a module logger"
        )

    @pytest.mark.parametrize("path", ANALYSIS_MODULES, ids=lambda p: p.name)
    def test_entry_points_configure_logging(self, path):
        """A script that logs but never configures a handler emits nothing."""
        source = path.read_text()
        if "__main__" not in source or "logger." not in source:
            return
        assert "setup_logging()" in source, (
            f"{path.name} is an entry point that logs but never calls setup_logging()"
        )


class TestFailuresAreVisible:
    """An analysis that fails must say so at a level someone reads.

    `run_comprehensive_analysis` used to wrap five diagnostics in
    `except Exception` and report each failure with `logger.info`, so a run in
    which every diagnostic failed still printed "complete" and exited zero. A
    similar loop swallowed a whole layer. Both are the same shape as the bare
    `except:` and the `strict=False` before it.
    """

    @pytest.mark.parametrize("path", ANALYSIS_MODULES, ids=lambda p: p.name)
    def test_no_error_is_reported_at_info_or_debug(self, path):
        offenders = []
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ExceptHandler):
                continue
            for call in ast.walk(node):
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr in ("info", "debug")
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "logger"
                ):
                    text = ast.unparse(call).lower()
                    if any(word in text for word in ("error", "failed", "exception")):
                        offenders.append(f"line {call.lineno}")
        assert not offenders, (
            f"{path.name}: reports a failure at INFO/DEBUG ({offenders}). Use "
            "logger.error or logger.exception — an error logged as information "
            "is an error nobody sees."
        )

    @pytest.mark.parametrize("path", ANALYSIS_MODULES, ids=lambda p: p.name)
    def test_no_traceback_printed_outside_logging(self, path):
        """`traceback.print_exc()` writes to stderr, bypassing the handler.

        The logging migration test looks for `print()`; this is a print by
        another name, and it was the last thing still writing outside the
        logging system.
        """
        source = path.read_text()
        assert "print_exc" not in source, (
            f"{path.name}: use logger.exception() so the traceback goes through logging"
        )

    @pytest.mark.parametrize("path", ANALYSIS_MODULES, ids=lambda p: p.name)
    def test_no_leftover_debug_scaffolding(self, path):
        """`DEBUG:`-prefixed messages logged at INFO are someone's debugging run.

        Sixteen of them survived in two analysers — "About to call save_results",
        "Exception caught, continuing" — printed on every run at INFO.
        """
        offenders = [
            call.lineno
            for call in ast.walk(ast.parse(path.read_text()))
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "info"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "logger"
            and "DEBUG:" in ast.unparse(call)
        ]
        assert not offenders, (
            f"{path.name}: DEBUG: messages logged at INFO on line(s) {offenders}. "
            "Delete them, or log them at logger.debug."
        )


class TestConfigAndDeviceSeam:
    """The analysis scripts must resolve config and device the same way the
    training scripts do.

    An explicit ``configs/training_config.yaml`` default reaches
    ``load_config`` as a real argument, which skips the ``MOE_CONFIG``
    environment variable entirely. That is why the CPU demo could drive all
    five training scripts but not one analysis script: pointing one at the demo
    fixtures failed on the repo config's ``YOUR_PATH_HERE`` placeholders.
    """

    @pytest.mark.parametrize("path", ANALYSIS_MODULES, ids=lambda p: p.name)
    def test_no_hardcoded_training_config_default(self, path):
        defaults = [
            node.lineno
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.endswith("configs/training_config.yaml")
        ]
        assert not defaults, (
            f"{path.name}: hardcodes the training config at line(s) {defaults}. "
            "Pass None and let models.utils.common.load_config resolve MOE_CONFIG."
        )

    @pytest.mark.parametrize("path", ANALYSIS_MODULES, ids=lambda p: p.name)
    def test_no_analysis_script_loads_weights_without_the_guard(self, path):
        """The analysis loaders read the same checkpoints the stages write.

        `strict=False` accepts a state dict that matches the model in no key at
        all: it loads nothing and returns normally. That is how Stages 2.5 and 3
        spent months starting from their Stage 0 experts while logging success,
        and `_lib/model_loading.py` had the same call.
        """
        assert "strict=False" not in path.read_text(), (
            f"{path.name}: use models.utils.checkpoints.load_matching_weights "
            "instead of a bare strict=False load"
        )

    @pytest.mark.parametrize("path", ANALYSIS_MODULES, ids=lambda p: p.name)
    def test_no_unconditional_cuda_default(self, path):
        """``device="cuda"`` as a default breaks the programmatic API on CPU.

        Two analysers guarded the argparse default with ``is_available()`` and
        then hardcoded ``"cuda"`` in the constructor, so they worked from the
        command line and failed when imported.
        """
        offenders = []
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.FunctionDef):
                continue
            defaults = node.args.defaults + [d for d in node.args.kw_defaults if d is not None]
            offenders += [
                f"{node.name}() line {d.lineno}"
                for d in defaults
                if isinstance(d, ast.Constant) and d.value == "cuda"
            ]
        assert not offenders, (
            f"{path.name}: {offenders} default to CUDA. Default to None and "
            "resolve with models.utils.common.get_device()."
        )


class TestAnalysisModulesAreImportable:
    """Every analysis module must import as part of the package.

    Five karpathy scripts used ``from karpathy_utils import ...``, which
    resolves only because Python puts a script's *own* directory on
    ``sys.path`` when you run it as a script. They ran, but no test could
    reach ``main()`` — and the repo-root ``sys.path`` edit lived in
    ``karpathy_utils`` itself, so whether a sibling's ``from models...``
    import worked depended on which import ran first.
    """

    @pytest.mark.parametrize("path", ANALYSIS_MODULES, ids=lambda p: p.name)
    def test_module_imports(self, path):
        dotted = path.relative_to(ANALYSIS_DIR.parent).with_suffix("").as_posix().replace("/", ".")
        assert importlib.import_module(dotted) is not None

    @pytest.mark.parametrize("path", ANALYSIS_MODULES, ids=lambda p: p.name)
    def test_no_flat_sibling_imports(self, path):
        """``from karpathy_utils import`` only works by accident of cwd."""
        siblings = {p.stem for p in ANALYSIS_DIR.rglob("*.py") if p.name != "__init__.py"}
        flat = [
            node.module
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.ImportFrom)
            and node.level == 0
            and (node.module or "").split(".")[0] in siblings
        ]
        assert not flat, (
            f"{path.name} imports {flat} by bare module name. Use the full "
            "package path so the module is importable from anywhere."
        )

    @pytest.mark.parametrize("path", ANALYSIS_MODULES, ids=lambda p: p.name)
    def test_entry_points_expose_main(self, path):
        """A script whose work is trapped in ``if __name__ == "__main__"`` can
        only be tested by launching a subprocess and reading files off disk."""
        source = path.read_text()
        if "__main__" not in source or "ArgumentParser" not in source:
            return
        tree = ast.parse(source)
        assert any(isinstance(n, ast.FunctionDef) and n.name == "main" for n in tree.body), (
            f"{path.name} builds its parser but has no main() to call"
        )


REPO = ANALYSIS_DIR.parent
SOURCE_FILES = sorted(
    path
    for directory in ("models", "data", "training_scripts", "analysis_scripts", "demo", "tests")
    for path in (REPO / directory).rglob("*.py")
)
MAX_LINES = 800


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.name)
def test_no_source_file_is_oversized(path):
    """No file over 800 lines.

    Four files were over when this was added — two analysers of 1382 and 1071
    lines, a plotting module of 843 and `train_stage_3.py` at 839. The guideline
    is worth enforcing rather than rediscovering: a reviewer opening a
    1,400-line module learns nothing about the design from it, and a file that
    large is where dead code and mangled docstrings survive unnoticed.
    """
    count = len(path.read_text().splitlines())
    assert count <= MAX_LINES, (
        f"{path.relative_to(REPO)} is {count} lines. Split it along a seam that "
        "already exists — extraction vs metrics, analysis vs plotting, setup vs loop."
    )
