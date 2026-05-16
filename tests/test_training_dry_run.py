"""Behavioural oracle regression test.

Loads the recorded baseline and re-runs the oracle. Any numeric drift means
a refactor changed training-relevant code — stop and investigate before merging.
"""

import json
import math
from pathlib import Path

import pytest

from tests.oracle import collect

BASELINE_PATH = Path(__file__).parent / "_fixtures" / "oracle_baseline.json"
TOLERANCE = 1e-5  # float32 precision; identical code should be bit-exact


@pytest.fixture(scope="module")
def baseline():
    return json.loads(BASELINE_PATH.read_text())


@pytest.fixture(scope="module")
def current():
    return collect()


def test_hard_routing_loss(baseline, current):
    assert math.isclose(current["hard"]["loss"], baseline["hard"]["loss"], rel_tol=TOLERANCE), (
        f"Hard-routing loss drifted: {current['hard']['loss']} vs {baseline['hard']['loss']}"
    )


def test_hard_routing_grad_norm(baseline, current):
    assert math.isclose(
        current["hard"]["grad_norm"], baseline["hard"]["grad_norm"], rel_tol=TOLERANCE
    ), (
        f"Hard-routing grad_norm drifted: "
        f"{current['hard']['grad_norm']} vs baseline {baseline['hard']['grad_norm']}"
    )


def test_soft_routing_loss(baseline, current):
    assert math.isclose(current["soft"]["loss"], baseline["soft"]["loss"], rel_tol=TOLERANCE), (
        f"Soft-routing loss drifted: {current['soft']['loss']} vs {baseline['soft']['loss']}"
    )


def test_soft_routing_grad_norm(baseline, current):
    assert math.isclose(
        current["soft"]["grad_norm"], baseline["soft"]["grad_norm"], rel_tol=TOLERANCE
    ), (
        f"Soft-routing grad_norm drifted: "
        f"{current['soft']['grad_norm']} vs baseline {baseline['soft']['grad_norm']}"
    )
