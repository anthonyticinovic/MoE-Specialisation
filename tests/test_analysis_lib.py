"""Unit tests for the shared analysis_scripts._lib helpers."""

import json

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
