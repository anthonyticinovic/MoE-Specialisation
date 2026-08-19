"""Validate result files against the schema the plots expect.

`make figures` is the only way to see a published number without a GPU, so the
JSON it reads has to be right. A truncated copy or a renamed key is caught here
rather than as an exception inside a plotting routine.

Two sets of files go through the same validators:

- **`paper_metrics/`** — the committed Stage 3 run. These tests skip while it is
  empty, which it currently is; they are what will check the files when added.
- **The CPU demo's own Stage 3 output** — see
  `tests/test_expert_metrics_pipeline.py`. That runs on every push, so the
  schema itself is exercised continuously even with nothing committed.

The validators live here and are imported there, so the two can never drift.
"""

import json
import math
import re
from pathlib import Path

import pytest

METRICS = Path(__file__).parent.parent / "paper_metrics" / "stage3"
EXPERT_METRICS = sorted(METRICS.glob("expert_metrics/expert_metrics_epoch_*.json"))
TRAINING_METRICS = METRICS / "training_metrics_stage3.json"

NUM_EXPERTS = 2
ENTROPY_CEILING = math.log(NUM_EXPERTS)  # nats, for a two-expert router
EXPERT_KEYS = {f"expert_{i}" for i in range(NUM_EXPERTS)}


def shares_sum_to_100(shares: dict, where: str) -> None:
    assert set(shares) == EXPERT_KEYS, f"{where}: expected {EXPERT_KEYS}, got {set(shares)}"
    total = sum(shares.values())
    assert total == pytest.approx(100.0, abs=0.5), f"{where}: shares sum to {total}, not 100"


def validate_expert_metrics(data: dict, where: str) -> None:
    """Every schema rule for one epoch's expert metrics, in one place."""
    assert set(data) >= {"per_layer", "aggregate"}, f"{where}: missing per_layer/aggregate"
    assert isinstance(data["per_layer"], list) and data["per_layer"], f"{where}: no layers"

    layers = [entry["layer"] for entry in data["per_layer"]]
    assert layers == list(range(len(layers))), f"{where}: layers are not 0..N-1 in order"

    for entry in data["per_layer"]:
        at = f"{where} layer {entry['layer']}"
        shares_sum_to_100(entry["expert_load_distribution"], f"{at} load")
        shares_sum_to_100(entry["visual_vs_text_routing"]["visual"], f"{at} visual")
        shares_sum_to_100(entry["visual_vs_text_routing"]["text"], f"{at} text")
        assert 0.0 <= entry["avg_routing_entropy"] <= ENTROPY_CEILING + 1e-6, (
            f"{at}: entropy {entry['avg_routing_entropy']} outside [0, ln 2]. "
            "A layer-summed total was reported as an entropy once before."
        )
        assert 0.0 <= entry["high_confidence_fraction"] <= 1.0, f"{at}: fraction out of [0,1]"

    aggregate = data["aggregate"]
    shares_sum_to_100(aggregate["expert_load_distribution"], f"{where} aggregate load")
    shares_sum_to_100(aggregate["visual_routing"], f"{where} aggregate visual")
    shares_sum_to_100(aggregate["text_routing"], f"{where} aggregate text")
    assert 0.0 <= aggregate["avg_routing_entropy"] <= ENTROPY_CEILING + 1e-6


def validate_training_metrics(data: dict, where: str) -> None:
    """The loss history the specialisation-vs-loss plot reads."""
    expected = {"epoch", "train_loss", "val_loss", "learning_rate"}
    assert set(data) >= expected, f"{where}: missing {expected - set(data)}"

    lengths = {key: len(data[key]) for key in expected}
    assert len(set(lengths.values())) == 1, f"{where}: columns have different lengths: {lengths}"

    for key in expected - {"epoch"}:
        assert all(math.isfinite(v) for v in data[key]), f"{where}: {key} contains NaN or Inf"
    assert data["epoch"] == list(range(1, len(data["epoch"]) + 1)), (
        f"{where}: epochs are not 1..N: {data['epoch']}"
    )


@pytest.mark.skipif(not EXPERT_METRICS, reason="no expert metrics committed yet")
@pytest.mark.parametrize("path", EXPERT_METRICS, ids=lambda p: p.name)
class TestExpertMetrics:
    def test_filename_carries_the_epoch(self, path):
        """plot_expert_metrics.py reads the epoch out of the filename."""
        assert re.fullmatch(r"expert_metrics_epoch_\d+\.json", path.name)

    def test_matches_the_schema(self, path):
        validate_expert_metrics(json.loads(path.read_text()), path.name)


@pytest.mark.skipif(not EXPERT_METRICS, reason="no expert metrics committed yet")
def test_every_epoch_covers_the_same_layers():
    """Epochs are plotted against each other, so they must be commensurable."""
    counts = {p.name: len(json.loads(p.read_text())["per_layer"]) for p in EXPERT_METRICS}
    assert len(set(counts.values())) == 1, f"epochs disagree on layer count: {counts}"


@pytest.mark.skipif(not TRAINING_METRICS.exists(), reason="no training metrics committed yet")
def test_training_metrics_match_the_schema():
    validate_training_metrics(json.loads(TRAINING_METRICS.read_text()), TRAINING_METRICS.name)


@pytest.mark.skipif(not EXPERT_METRICS, reason="no expert metrics committed yet")
def test_training_metrics_accompany_the_expert_metrics():
    """The loss plot needs both; committing one without the other half-works."""
    assert TRAINING_METRICS.exists(), (
        f"{len(EXPERT_METRICS)} epochs of expert metrics are committed but "
        f"{TRAINING_METRICS.name} is missing — the loss plot will be skipped"
    )
