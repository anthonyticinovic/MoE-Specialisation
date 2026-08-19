"""Tests that the demo's invariants actually fire.

A check that can never fail is worse than no check: it looks like coverage and
provides none. Each test here corrupts exactly the property one invariant is
supposed to protect, and asserts that the invariant notices.

The two module-level tests at the bottom keep that true. The claim "each
invariant has a test that proves it can fail" was made in `docs/design.md` while
four of the sixteen had no such test — the count had drifted twice and nothing
was watching. It is enforced now rather than asserted.
"""

import inspect
import json
import math
from pathlib import Path

import pytest
import torch

from demo import checks


@pytest.fixture
def moe_dir(tmp_path):
    """A real Stage 0 output built from a tiny base model."""
    from transformers import MistralConfig, MistralForCausalLM

    from models.utils.create_moe_model import create_moe_model

    config = MistralConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    base = tmp_path / "base"
    MistralForCausalLM(config).save_pretrained(str(base))

    out = tmp_path / "moe"
    create_moe_model(str(base), str(out))
    return out


class TestStage0Checks:
    def test_identical_experts_passes_on_real_output(self, moe_dir):
        assert checks.check_experts_identical_at_init(moe_dir).passed

    def test_identical_experts_fails_when_experts_differ(self, moe_dir, tmp_path):
        """Perturb one expert; the check must notice they no longer match."""
        from safetensors.torch import load_file, save_file

        weights = moe_dir / "model.safetensors"
        tensors = load_file(str(weights))
        key = next(k for k in tensors if ".mlp.experts.1." in k)
        tensors[key] = tensors[key] + 1.0
        save_file(tensors, str(weights))

        result = checks.check_experts_identical_at_init(moe_dir)
        assert not result.passed
        assert "differs" in result.detail

    def test_config_check_fails_on_dense_model_type(self, moe_dir):
        """The exact regression that made checkpoints load as dense models."""
        config_path = moe_dir / "config.json"
        config = json.loads(config_path.read_text())
        config["model_type"] = "mistral"
        config_path.write_text(json.dumps(config))

        result = checks.check_config_declares_moe(moe_dir)
        assert not result.passed
        assert "dense" in result.detail

    def test_hard_routing_check_passes(self, moe_dir):
        assert checks.check_hard_routing_is_exact(moe_dir).passed


class TestStage2Checks:
    def _write_checkpoint(self, path, state):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": state}, path)

    def test_divergence_fails_when_experts_never_trained(self, moe_dir, tmp_path):
        """Experts start identical — if training is a no-op they stay identical."""
        from safetensors.torch import load_file

        state = load_file(str(moe_dir / "model.safetensors"))
        ckpt = tmp_path / "stage2.pth"
        self._write_checkpoint(ckpt, state)

        result = checks.check_experts_diverged(moe_dir, ckpt)
        assert not result.passed
        assert "bit-identical" in result.detail

    def test_divergence_passes_once_experts_differ(self, moe_dir, tmp_path):
        from safetensors.torch import load_file

        state = load_file(str(moe_dir / "model.safetensors"))
        key = next(k for k in state if ".mlp.experts.0." in k)
        state[key] = state[key] + 0.01
        ckpt = tmp_path / "stage2.pth"
        self._write_checkpoint(ckpt, state)

        assert checks.check_experts_diverged(moe_dir, ckpt).passed

    def test_frozen_weights_check_catches_a_moved_attention_weight(self, moe_dir, tmp_path):
        """Stage 2 must not touch attention; simulate a freezing bug."""
        from safetensors.torch import load_file

        state = load_file(str(moe_dir / "model.safetensors"))
        key = next(k for k in state if "self_attn" in k)
        state[key] = state[key] + 0.5
        ckpt = tmp_path / "stage2.pth"
        self._write_checkpoint(ckpt, state)

        result = checks.check_only_experts_trained(moe_dir, ckpt)
        assert not result.passed
        assert "frozen tensors changed" in result.detail


class TestStage25Checks:
    """Stage 2.5 trains the gate and nothing else — both halves must be checked.

    The invariant has two failure modes and they mean opposite things: a moved
    expert is a freezing bug, an unmoved gate is a training bug. A check that
    only noticed one would pass on a Stage 2.5 that did nothing at all.
    """

    def _checkpoint(self, path, state):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": {k: v.clone() for k, v in state.items()}}, path)
        return path

    @pytest.fixture
    def stage2(self, moe_dir, tmp_path):
        """A Stage 2 checkpoint, and the state to derive Stage 2.5 ones from."""
        from safetensors.torch import load_file

        state = load_file(str(moe_dir / "model.safetensors"))
        return state, self._checkpoint(tmp_path / "stage2.pth", state)

    def test_passes_when_only_the_gates_moved(self, stage2, tmp_path):
        state, before = stage2
        after = {k: v.clone() for k, v in state.items()}
        key = next(k for k in after if checks.GATE_KEY in k)
        after[key] = after[key] + 0.1

        result = checks.check_only_gates_trained(
            before, self._checkpoint(tmp_path / "b.pth", after)
        )
        assert result.passed and not result.skipped

    def test_fails_when_an_expert_moved(self, stage2, tmp_path):
        """The freezing bug: Stage 2.5 must not update the experts."""
        state, before = stage2
        after = {k: v.clone() for k, v in state.items()}
        after[next(k for k in after if checks.GATE_KEY in k)] += 0.1
        after[next(k for k in after if checks.EXPERT_KEY in k)] += 0.1

        result = checks.check_only_gates_trained(
            before, self._checkpoint(tmp_path / "b.pth", after)
        )
        assert not result.passed
        assert "expert tensors changed" in result.detail

    def test_fails_when_no_gate_moved(self, stage2, tmp_path):
        """The silent no-op: the router never trained and nothing said so."""
        state, before = stage2
        result = checks.check_only_gates_trained(
            before, self._checkpoint(tmp_path / "b.pth", state)
        )
        assert not result.passed
        assert "router did not train" in result.detail


class TestReportedEntropyBounds:
    """The per-stage metrics history has its own entropy, and its own ceiling.

    Separate from ``check_routing_entropy_bounds``: that one reads the
    ExpertUsageTracker dump, this one reads what Stage 2.5 recorded per epoch.
    Both can be wrong independently.
    """

    def _history(self, tmp_path, entropy):
        runs = tmp_path / "runs"
        runs.mkdir(exist_ok=True)
        (runs / "training_metrics_stage2.5.json").write_text(json.dumps({"entropy": entropy}))
        return runs

    def _check(self, runs):
        return checks.check_reported_entropy_bounds(
            runs, "training_metrics_stage2.5.json", "stage2.5"
        )

    def test_passes_within_ln2(self, tmp_path):
        assert self._check(self._history(tmp_path, [0.69, 0.65, 0.4])).passed

    def test_fails_above_ln2(self, tmp_path):
        """A per-layer entropy summed instead of averaged scales with depth."""
        runs = self._history(tmp_path, [0.65, 2 * math.log(2)])
        result = self._check(runs)
        assert not result.passed
        assert "summed rather than averaged" in result.detail

    def test_negative_entropy_is_caught(self, tmp_path):
        assert not self._check(self._history(tmp_path, [-0.1])).passed

    def test_skips_when_the_stage_never_ran(self, tmp_path):
        runs = tmp_path / "runs"
        runs.mkdir()
        result = self._check(runs)
        assert result.skipped and result.passed

    def test_skips_when_no_entropy_was_recorded(self, tmp_path):
        result = self._check(self._history(tmp_path, []))
        assert result.skipped and result.passed


class TestExpertMetricFigures:
    """The figure pipeline must produce every output, and none of them a stub."""

    def _figures(self, tmp_path, *, omit=None, stub=None):
        directory = tmp_path / "expert_metrics"
        directory.mkdir()
        for name in checks.EXPECTED_FIGURES - {omit}:
            size = 10 if name == stub else checks.MIN_FIGURE_BYTES * 2
            (directory / name).write_bytes(b"x" * size)
        return directory

    def test_passes_when_every_output_is_present_and_substantial(self, tmp_path):
        assert checks.check_expert_metric_figures(self._figures(tmp_path)).passed

    def test_fails_when_a_figure_is_missing(self, tmp_path):
        """A plotting routine raised, or a metric key was renamed under it."""
        result = checks.check_expert_metric_figures(
            self._figures(tmp_path, omit="routing_entropy.png")
        )
        assert not result.passed
        assert "routing_entropy.png" in result.detail

    def test_fails_on_a_stub_left_by_a_figure_that_raised(self, tmp_path):
        """The failure mode 'the file exists' would miss: a truncated PNG."""
        result = checks.check_expert_metric_figures(
            self._figures(tmp_path, stub="aggregate_summary.png")
        )
        assert not result.passed
        assert "suspiciously small" in result.detail

    def test_skips_when_the_figures_stage_did_not_run(self, tmp_path):
        result = checks.check_expert_metric_figures(tmp_path / "absent", stage_ran=False)
        assert result.skipped and result.passed

    def test_fails_when_the_stage_ran_and_wrote_nothing(self, tmp_path):
        result = checks.check_expert_metric_figures(tmp_path / "absent", stage_ran=True)
        assert not result.passed and not result.skipped


class TestReportMatchesTheMetrics:
    """The figures are images; the report is the only output that can be read back.

    So it stands in for all of them. If the plotting code starts reading the
    wrong key, every figure is wrong and only this notices.
    """

    AGGREGATE = {
        "expert_load_distribution": {"expert_0": 43.44, "expert_1": 56.56},
        "avg_routing_entropy": 0.6603,
    }

    def _report(self, tmp_path, text):
        directory = tmp_path / "expert_metrics"
        directory.mkdir(exist_ok=True)
        (directory / "expert_metrics_report.txt").write_text(text)
        return directory

    def _metrics(self):
        return {"per_layer": [], "aggregate": self.AGGREGATE}

    def test_passes_when_the_report_quotes_the_json(self, tmp_path):
        directory = self._report(tmp_path, "expert_0: 43.44%\nexpert_1: 56.56%\nentropy: 0.6603\n")
        assert checks.check_report_matches_the_metrics(directory, self._metrics()).passed

    def test_fails_when_a_load_does_not_match(self, tmp_path):
        """The wrong-key regression: plausible numbers, from the wrong place."""
        directory = self._report(tmp_path, "expert_0: 34.44%\nexpert_1: 56.56%\nentropy: 0.6603\n")
        result = checks.check_report_matches_the_metrics(directory, self._metrics())
        assert not result.passed
        assert "expert_0 load" in result.detail

    def test_fails_when_the_entropy_does_not_match(self, tmp_path):
        directory = self._report(tmp_path, "expert_0: 43.44%\nexpert_1: 56.56%\nentropy: 0.1234\n")
        result = checks.check_report_matches_the_metrics(directory, self._metrics())
        assert not result.passed
        assert "routing entropy" in result.detail

    def test_skips_without_a_report(self, tmp_path):
        result = checks.check_report_matches_the_metrics(tmp_path, self._metrics())
        assert result.skipped and result.passed

    def test_skips_without_metrics(self, tmp_path):
        directory = self._report(tmp_path, "anything")
        result = checks.check_report_matches_the_metrics(directory, None)
        assert result.skipped and result.passed


class TestMetricChecks:
    def _metrics(self, entropy):
        return {
            "per_layer": [
                {
                    "layer": 0,
                    "avg_routing_entropy": entropy,
                    "expert_load_distribution": {"expert_0": 40.0, "expert_1": 60.0},
                }
            ],
            "aggregate": {"avg_routing_entropy": entropy},
        }

    def test_entropy_bounds_pass_within_ln2(self):
        assert checks.check_routing_entropy_bounds(self._metrics(0.5)).passed

    def test_entropy_bounds_fail_above_ln2(self):
        """The summed-instead-of-averaged signature."""
        result = checks.check_routing_entropy_bounds(self._metrics(2 * math.log(2)))
        assert not result.passed
        assert "ln(2)" in result.detail

    def test_loads_must_sum_to_100(self):
        broken = self._metrics(0.5)
        broken["per_layer"][0]["expert_load_distribution"] = {"expert_0": 40.0, "expert_1": 40.0}
        result = checks.check_expert_loads_sum_to_100(broken)
        assert not result.passed

    def test_layer_count_mismatch_is_caught(self):
        result = checks.check_all_layers_reported(self._metrics(0.5), expected_layers=32)
        assert not result.passed
        assert "32" in result.detail


class TestNumericalHealth:
    def test_nan_in_checkpoint_is_caught(self, tmp_path):
        runs = tmp_path / "runs"
        runs.mkdir()
        torch.save({"model_state_dict": {"w": torch.tensor([1.0, float("nan")])}}, runs / "a.pth")

        result = checks.check_checkpoints_finite(runs)
        assert not result.passed
        assert "non-finite" in result.detail

    def test_finite_checkpoint_passes(self, tmp_path):
        runs = tmp_path / "runs"
        runs.mkdir()
        torch.save({"model_state_dict": {"w": torch.tensor([1.0, 2.0])}}, runs / "a.pth")
        assert checks.check_checkpoints_finite(runs).passed

    def test_nan_loss_is_caught(self, tmp_path):
        runs = tmp_path / "runs"
        runs.mkdir()
        (runs / "m.json").write_text(json.dumps({"train_loss": [1.0, float("nan")]}))

        result = checks.check_losses_finite(runs, {"Stage X": "m.json"})
        assert not result.passed


class TestSkipBehaviour:
    def test_missing_artifacts_skip_rather_than_fail(self, tmp_path):
        """A partial run (--stages 0 1) must not report spurious failures."""
        result = checks.check_experts_diverged(tmp_path, tmp_path / "absent.pth")
        assert result.skipped
        assert result.passed


class TestAnalysisChecks:
    """The two invariants covering the analysis half of the repo."""

    def _results(self, tmp_path, normal, flipped):
        payload = {
            "num_samples": len(normal),
            "normal_routing": {"losses": normal},
            "flipped_routing": {"losses": flipped},
            "delta": {"percent": 1.0},
        }
        path = tmp_path / "routing_ablation_results.json"
        path.write_text(json.dumps(payload))
        return path

    def test_missing_results_file_fails(self, tmp_path):
        """The analysis stage running is itself the check: a silent skip here
        would restore the state N2 fixed, where the demo covered no analysis
        code at all."""
        result = checks.check_routing_ablation_ran(tmp_path / "absent.json")
        assert not result.passed

    def test_non_finite_loss_is_caught(self, tmp_path):
        path = self._results(tmp_path, [1.0, float("nan")], [1.1, 1.2])
        assert not checks.check_routing_ablation_ran(path).passed

    def test_mismatched_series_lengths_are_caught(self, tmp_path):
        path = self._results(tmp_path, [1.0, 1.1], [1.2])
        assert not checks.check_routing_ablation_ran(path).passed

    def test_specialised_experts_pass(self, tmp_path):
        path = self._results(tmp_path, [1.0, 1.1], [1.2, 1.3])
        assert checks.check_flipped_routing_is_worse(path).passed

    def test_a_single_sample_that_prefers_the_flip_fails(self, tmp_path):
        path = self._results(tmp_path, [1.0, 1.1], [1.2, 0.9])
        result = checks.check_flipped_routing_is_worse(path)
        assert not result.passed
        assert "did not specialise" in result.detail

    def test_identical_losses_fail(self, tmp_path):
        """Equal losses mean routing made no difference — the experts are copies."""
        path = self._results(tmp_path, [1.0, 1.1], [1.0, 1.1])
        assert not checks.check_flipped_routing_is_worse(path).passed


class TestPartialRunSkips:
    """A partial run must not invent failures for stages it never executed.

    `--stages 0 1` is a normal way to use the demo. The analysis checks were
    added after this convention existed and initially failed the whole run
    when the analysis stage had simply not been asked for.
    """

    def test_analysis_check_skips_when_the_stage_did_not_run(self, tmp_path):
        result = checks.check_routing_ablation_ran(tmp_path / "absent.json", stage_ran=False)
        assert result.skipped and result.passed

    def test_analysis_check_fails_when_the_stage_ran_and_wrote_nothing(self, tmp_path):
        result = checks.check_routing_ablation_ran(tmp_path / "absent.json", stage_ran=True)
        assert not result.passed and not result.skipped

    def test_partial_run_reports_no_failures(self, tmp_path):
        """The end-to-end version of the above, through run_all()."""
        results = checks.run_all(tmp_path, {}, stages=["0", "1"])
        assert not [r for r in results if not r.passed]


def _invariants() -> set[str]:
    return {
        name
        for name, obj in inspect.getmembers(checks, inspect.isfunction)
        if name.startswith("check_") and obj.__module__ == checks.__name__
    }


def test_every_invariant_is_wired_into_run_all():
    """A check nobody calls protects nothing, however good it looks."""
    called = inspect.getsource(checks.run_all)
    orphans = sorted(name for name in _invariants() if f"{name}(" not in called)
    assert not orphans, f"defined but never run by run_all(): {orphans}"


def test_every_invariant_is_covered_by_a_test_here():
    """The ratchet on this file. See the module docstring for why it exists.

    Naming the check is the minimum bar, not proof the test is a good one — but
    it is the bar that was silently missed, and it is the one a reviewer counts.
    """
    source = Path(__file__).read_text()
    uncovered = sorted(name for name in _invariants() if f"checks.{name}(" not in source)
    assert not uncovered, (
        f"{len(uncovered)} invariant(s) with no test proving they can fail: {uncovered}. "
        "Corrupt the property each one protects and assert it notices."
    )
