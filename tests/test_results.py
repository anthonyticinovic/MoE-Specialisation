"""Validate the committed result files against the schema the plots expect.

`make figures` is the only way to see a published number without a GPU, so the
JSON it reads has to be right. These tests skip when `paper_metrics/` is empty — the
files are added from a completed Stage 3 run — and check every file that is
present. A truncated copy or a renamed key is caught here rather than as an
exception inside a plotting routine.
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


def _shares_sum_to_100(shares: dict, where: str) -> None:
    assert set(shares) == EXPERT_KEYS, f"{where}: expected {EXPERT_KEYS}, got {set(shares)}"
    total = sum(shares.values())
    assert total == pytest.approx(100.0, abs=0.5), f"{where}: shares sum to {total}, not 100"


@pytest.mark.skipif(not EXPERT_METRICS, reason="no expert metrics committed yet")
@pytest.mark.parametrize("path", EXPERT_METRICS, ids=lambda p: p.name)
class TestExpertMetrics:
    def test_filename_carries_the_epoch(self, path):
        """plot_expert_metrics.py reads the epoch out of the filename."""
        assert re.fullmatch(r"expert_metrics_epoch_\d+\.json", path.name)

    def test_top_level_shape(self, path):
        data = json.loads(path.read_text())
        assert set(data) >= {"per_layer", "aggregate"}
        assert isinstance(data["per_layer"], list) and data["per_layer"]

    def test_layers_are_complete_and_in_order(self, path):
        """Every layer must appear once, ascending — the plots index by position."""
        layers = [entry["layer"] for entry in json.loads(path.read_text())["per_layer"]]
        assert layers == sorted(layers), f"{path.name}: layers are out of order"
        assert layers == list(range(len(layers))), f"{path.name}: layers are not 0..N-1"

    def test_per_layer_entries_are_well_formed(self, path):
        for entry in json.loads(path.read_text())["per_layer"]:
            where = f"{path.name} layer {entry['layer']}"
            _shares_sum_to_100(entry["expert_load_distribution"], f"{where} load")
            _shares_sum_to_100(entry["visual_vs_text_routing"]["visual"], f"{where} visual")
            _shares_sum_to_100(entry["visual_vs_text_routing"]["text"], f"{where} text")
            assert 0.0 <= entry["avg_routing_entropy"] <= ENTROPY_CEILING + 1e-6, (
                f"{where}: entropy {entry['avg_routing_entropy']} outside [0, ln 2]. "
                "A layer-summed total was reported as an entropy once before."
            )
            assert 0.0 <= entry["high_confidence_fraction"] <= 1.0, (
                f"{where}: fraction out of [0,1]"
            )

    def test_aggregate_is_well_formed(self, path):
        aggregate = json.loads(path.read_text())["aggregate"]
        _shares_sum_to_100(aggregate["expert_load_distribution"], f"{path.name} aggregate load")
        _shares_sum_to_100(aggregate["visual_routing"], f"{path.name} aggregate visual")
        _shares_sum_to_100(aggregate["text_routing"], f"{path.name} aggregate text")
        assert 0.0 <= aggregate["avg_routing_entropy"] <= ENTROPY_CEILING + 1e-6


@pytest.mark.skipif(not EXPERT_METRICS, reason="no expert metrics committed yet")
def test_every_epoch_covers_the_same_layers():
    """Epochs are plotted against each other, so they must be commensurable."""
    counts = {p.name: len(json.loads(p.read_text())["per_layer"]) for p in EXPERT_METRICS}
    assert len(set(counts.values())) == 1, f"epochs disagree on layer count: {counts}"


@pytest.mark.skipif(not TRAINING_METRICS.exists(), reason="no training metrics committed yet")
class TestTrainingMetrics:
    def test_columns_are_parallel_and_finite(self):
        data = json.loads(TRAINING_METRICS.read_text())
        expected = {"epoch", "train_loss", "val_loss", "learning_rate"}
        assert set(data) >= expected, f"missing {expected - set(data)}"

        lengths = {key: len(data[key]) for key in expected}
        assert len(set(lengths.values())) == 1, f"columns have different lengths: {lengths}"

        for key in expected - {"epoch"}:
            assert all(math.isfinite(v) for v in data[key]), f"{key} contains NaN or Inf"

    def test_epochs_are_sequential(self):
        epochs = json.loads(TRAINING_METRICS.read_text())["epoch"]
        assert epochs == list(range(1, len(epochs) + 1)), f"epochs are not 1..N: {epochs}"


@pytest.mark.skipif(not EXPERT_METRICS, reason="no expert metrics committed yet")
def test_training_metrics_accompany_the_expert_metrics():
    """The loss plot needs both; committing one without the other half-works."""
    assert TRAINING_METRICS.exists(), (
        f"{len(EXPERT_METRICS)} epochs of expert metrics are committed but "
        f"{TRAINING_METRICS.name} is missing — the loss plot will be skipped"
    )
