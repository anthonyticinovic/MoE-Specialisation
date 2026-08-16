"""Unit tests for the shared analysis_scripts._lib helpers."""

import ast
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
