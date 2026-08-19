"""The Stage 3 metrics → figures path, end to end on CPU.

`plot_expert_metrics.py` is the second analysis entry point the demo executes,
and this is the same path driven as functions rather than subprocesses: Stage 3
emits routing metrics, the schema validators check them, and `main()` turns them
into the figures the paper's routing analysis is built from.

Nothing here needs data you have to supply. The metrics come from the demo's own
Stage 3 run against synthetic fixtures, which is why this covers the schema on
every push while `tests/test_results.py` — the same validators applied to the
*committed* `paper_metrics/` — is still skipped for want of those files.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tests.pipeline import build_setup, load_stage, train_epoch
from tests.test_results import validate_expert_metrics, validate_training_metrics


@pytest.fixture(scope="session")
def stage3_metrics(tmp_path_factory, fixtures_config, prior_stages, run_context) -> Path:
    """A directory of real Stage 3 expert metrics, one file per epoch.

    Runs an epoch and a validation pass on the demo fixtures — the validation
    pass is what populates the tracker, so training alone is not enough.
    """
    import shutil

    output_dir = tmp_path_factory.mktemp("stage3_metrics")
    shutil.copytree(prior_stages, output_dir, dirs_exist_ok=True)

    config = copy.deepcopy(fixtures_config)
    config["paths"]["output_dir"] = str(output_dir)

    stage_3 = load_stage("stage_3")
    setup = build_setup(stage_3, "stage_3", config, run_context)
    train_epoch(stage_3, setup, run_context)
    _, expert_metrics, _ = stage_3.run_validation(setup, run_context)
    assert expert_metrics is not None, "validation produced no routing metrics"

    from training_scripts._lib import save_expert_metrics

    save_expert_metrics(expert_metrics, str(output_dir), 0, setup.num_visual_tokens)
    return output_dir


class TestStage3EmitsValidMetrics:
    """What Stage 3 writes must satisfy the schema the plots assume.

    `tests/test_results.py` holds these rules and is skipped until
    `paper_metrics/` is populated. Applying them to the demo's own output is
    what makes the schema continuously enforced instead of aspirational.
    """

    def test_metrics_file_is_written_with_the_expected_name(self, stage3_metrics):
        """`plot_expert_metrics.py` reads the epoch out of the filename."""
        files = sorted((stage3_metrics / "expert_metrics").glob("expert_metrics_epoch_*.json"))
        assert files, "Stage 3 wrote no expert metrics"

    def test_metrics_match_the_published_schema(self, stage3_metrics):
        for path in sorted((stage3_metrics / "expert_metrics").glob("*.json")):
            validate_expert_metrics(json.loads(path.read_text()), path.name)

    def test_one_entry_per_model_layer(self, stage3_metrics, fixtures_config):
        """A tracker sized from the wrong constant silently drops layers.

        `ExpertUsageTracker` used to default to the 7B model's 32; it is now
        sized from the loaded config, and the demo's model has 2.
        """
        config_path = Path(fixtures_config["paths"]["moe_model_path"]) / "config.json"
        expected = json.loads(config_path.read_text())["num_hidden_layers"]

        path = next(iter(sorted((stage3_metrics / "expert_metrics").glob("*.json"))))
        layers = json.loads(path.read_text())["per_layer"]
        assert len(layers) == expected, f"{len(layers)} layers reported, model has {expected}"

    def test_training_metrics_match_the_published_schema(self, prior_stages, fixtures_config):
        """The loss history the specialisation-vs-loss plot reads.

        Produced by a full `main()` run rather than a single epoch, so it is
        checked against the demo's output rather than rebuilt here.
        """
        path = Path("demo_output/runs/training_metrics_stage3.json")
        if not path.exists():
            pytest.skip("no demo_output — run `make demo` to cover this")
        validate_training_metrics(json.loads(path.read_text()), path.name)


class TestPlotExpertMetricsMain:
    """The plotting entry point, called as a function rather than a subprocess.

    The demo runs it as a subprocess, which proves the command line works; this
    gives a failure a traceback instead of an exit code.
    """

    def test_main_produces_every_figure(self, stage3_metrics, tmp_path):
        from analysis_scripts.plot_expert_metrics import main

        output = tmp_path / "figures"
        assert (
            main(
                [
                    "--metrics_dir",
                    str(stage3_metrics / "expert_metrics"),
                    "--output_dir",
                    str(output),
                    "--layers",
                    "all_layers",
                ]
            )
            == 0
        )

        produced = {path.name for path in output.iterdir()}
        assert "expert_load_distribution.png" in produced
        assert "routing_entropy.png" in produced
        assert "expert_metrics_report.txt" in produced
        assert all(path.stat().st_size > 1024 for path in output.glob("*.png"))

    def test_a_layer_selection_beyond_the_model_is_rejected(self, stage3_metrics, tmp_path):
        """The demo's model has 2 layers; the default selection is `0 7 15 23 31`.

        This used to raise `IndexError` from inside a plotting routine on any
        model that was not the 7B one.
        """
        from analysis_scripts.plot_expert_metrics import main

        output = tmp_path / "figures"
        code = main(
            [
                "--metrics_dir",
                str(stage3_metrics / "expert_metrics"),
                "--output_dir",
                str(output),
                "--layers",
                "0 7 15 23 31",
            ]
        )
        assert code == 1, "refusing to plot must be reported as a failure, not a silent return"
        assert not output.exists() or not list(output.glob("*.png"))
