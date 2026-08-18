"""Behavioural tests for ``analysis_scripts._lib.model_loading``.

Every analysis script that touches a checkpoint goes through this module, and
until now its whole test coverage was a structure check asserting the string
``strict=False`` does not appear in the file. That is the same shape of gap that
let ``models/utils/generation.py`` sit broken for months: a check that proves a
file parses, standing in for one that proves it works.

These drive the real loaders against the real checkpoints the training stages
produce, on CPU, in a couple of seconds. Each one pins a property that was
either recently fixed or is silently load-bearing:

- the checkpoint guard refuses a state dict that matches nothing (the bug that
  cost Stages 2.5 and 3 their Stage 2 experts);
- the connector is sized from the loaded backbones, not from the CLIP-L /
  Mistral-7B defaults, which could only ever have loaded one model pair;
- Stage 3 sets ``_forward_temperature`` and not only ``temperature`` — the MoE
  layer never reads the latter, so a Stage 3 analysis was silently running at
  the default temperature;
- device and dtype follow the machine.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from analysis_scripts._lib import load_stage2_models, load_stage3_models
from models.utils.common import get_model_dtype
from tests.pipeline import build_setup, load_stage, train_epoch


@pytest.fixture(scope="session")
def stage3_artifacts(tmp_path_factory, fixtures_config, prior_stages, run_context) -> Path:
    """Stage 1, 2 and 3 checkpoints in one directory.

    Stage 3 needs the Stage 1 connector and the Stage 2 experts, so it runs on
    a copy of the shared ``prior_stages`` output rather than beside it.
    """
    import shutil

    output_dir = tmp_path_factory.mktemp("stage3_artifacts")
    shutil.copytree(prior_stages, output_dir, dirs_exist_ok=True)

    config = copy.deepcopy(fixtures_config)
    config["paths"]["output_dir"] = str(output_dir)

    stage_3 = load_stage("stage_3")
    setup = build_setup(stage_3, "stage_3", config, run_context)
    train_epoch(stage_3, setup, run_context)
    checkpoint_dir = output_dir / "stage3_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stage_3.save_checkpoints(setup, run_context, str(checkpoint_dir), 0, 1.0, float("inf"))
    return output_dir


@pytest.fixture
def analysis_config(fixtures_config, prior_stages) -> dict:
    """The demo config pointed at the directory Stages 1 and 2 wrote to."""
    config = copy.deepcopy(fixtures_config)
    config["paths"]["output_dir"] = str(prior_stages)
    return config


@pytest.fixture
def stage3_config(fixtures_config, stage3_artifacts) -> dict:
    config = copy.deepcopy(fixtures_config)
    config["paths"]["output_dir"] = str(stage3_artifacts)
    return config


class TestStage2Loader:
    def test_loads_every_component(self, analysis_config):
        models = load_stage2_models(analysis_config)
        assert models.llm is not None
        assert models.vision_encoder is not None
        assert models.vision_connector is not None
        assert models.tokenizer is not None
        assert models.clip_processor is not None

    def test_every_moe_layer_is_in_hard_routing_mode(self, analysis_config):
        """Stage 2's whole premise is the position-derived mask. A layer left in
        soft mode would route by a gate that Stage 2 never trained."""
        models = load_stage2_models(analysis_config)
        modes = {layer.mlp.routing_mode for layer in models.llm.model.layers}
        assert modes == {"hard"}

    def test_the_connector_is_sized_from_the_loaded_backbones(self, analysis_config):
        """The demo's CLIP is 32-wide and its LLM 64-wide.

        The loader used to construct ``VisionLanguageConnector()`` with the
        CLIP-L/Mistral-7B defaults (1024 → 4096), so loading a connector trained
        against any other pair raised on the state-dict shapes.
        """
        models = load_stage2_models(analysis_config)
        first = models.vision_connector.mlp[0]
        assert first.in_features == models.vision_encoder.config.hidden_size
        assert first.out_features == models.llm.config.hidden_size

    def test_the_connector_carries_the_stage_1_weights(self, analysis_config):
        """Constructing the right shape is not the same as loading into it."""
        models = load_stage2_models(analysis_config)
        trained = torch.load(
            Path(analysis_config["paths"]["output_dir"]) / "vision_connector_stage1_best.pth",
            map_location="cpu",
        )
        for name, param in models.vision_connector.state_dict().items():
            assert torch.equal(param.cpu(), trained[name].cpu()), f"{name} was not loaded"

    def test_the_experts_carry_the_stage_2_weights(self, analysis_config):
        """The regression proper: with a bare ``strict=False`` a changed
        checkpoint format loaded *nothing* and reported success."""
        models = load_stage2_models(analysis_config)
        checkpoint = torch.load(
            Path(analysis_config["paths"]["output_dir"])
            / "stage2_checkpoints"
            / "llm_stage2_best.pth",
            map_location="cpu",
            weights_only=False,
        )
        trained = checkpoint["model_state_dict"]
        expert_keys = [k for k in trained if ".mlp.experts." in k]
        assert expert_keys, "the Stage 2 checkpoint has no expert weights to compare"

        loaded = models.llm.state_dict()
        for key in expert_keys:
            assert torch.equal(loaded[key].cpu().float(), trained[key].cpu().float()), (
                f"{key} kept its Stage 0 value — the checkpoint did not apply"
            )

    def test_a_checkpoint_that_matches_nothing_raises(self, analysis_config, tmp_path):
        """A state dict overlapping the model in no key is what a changed
        format looks like. It must not load silently."""
        bogus = tmp_path / "bogus.pth"
        torch.save({"not_a_real_parameter": torch.zeros(3)}, bogus)

        with pytest.raises(RuntimeError, match="matched none of the model"):
            load_stage2_models(analysis_config, stage2_checkpoint=str(bogus))

    def test_dtype_and_device_follow_the_machine(self, analysis_config):
        """bfloat16 on a GPU node exactly as the paper runs used, float32 on CPU
        where bfloat16 matmuls are unsupported or far slower."""
        models = load_stage2_models(analysis_config)
        assert models.llm.dtype == get_model_dtype()
        assert not torch.cuda.is_available() or next(models.llm.parameters()).is_cuda

    def test_the_model_is_in_eval_mode(self, analysis_config):
        """Dropout during analysis would make every measurement irreproducible."""
        models = load_stage2_models(analysis_config)
        assert not models.llm.training
        assert not models.vision_encoder.training
        assert not models.vision_connector.training


class TestStage3Loader:
    def test_every_moe_layer_is_in_soft_routing_mode(self, stage3_config, stage3_artifacts):
        models = load_stage3_models(
            stage3_config,
            None,
            str(stage3_artifacts / "stage3_checkpoints" / "llm_stage3_best.pth"),
        )
        modes = {layer.mlp.routing_mode for layer in models.llm.model.layers}
        assert modes == {"soft"}

    def test_the_temperature_reaches_the_attribute_the_layer_reads(
        self, stage3_config, stage3_artifacts
    ):
        """``MoELayer`` reads ``_forward_temperature``, never ``temperature``.

        ``cross_modality_purity`` used to set only the latter, so its Stage 3
        analyses silently ran at the default temperature. The shared loader sets
        both; this is what stops that regressing back to one.
        """
        models = load_stage3_models(
            stage3_config,
            None,
            str(stage3_artifacts / "stage3_checkpoints" / "llm_stage3_best.pth"),
            temperature=0.25,
        )
        for layer in models.llm.model.layers:
            assert layer.mlp._forward_temperature == pytest.approx(0.25)
            assert layer.mlp.temperature == pytest.approx(0.25)

    def test_the_full_checkpoint_updates_the_connector(self, stage3_config, stage3_artifacts):
        """Stage 3 trains the connector too, so a full checkpoint must apply it —
        otherwise the analysis runs Stage 3's LLM against Stage 1's projection."""
        checkpoint_path = stage3_artifacts / "stage3_checkpoints" / "llm_stage3_best.pth"
        models = load_stage3_models(stage3_config, None, str(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        assert "connector_state_dict" in checkpoint
        for name, param in models.vision_connector.state_dict().items():
            assert torch.equal(param.cpu(), checkpoint["connector_state_dict"][name].cpu())

    def test_the_portable_checkpoint_loads_too(self, stage3_config, stage3_artifacts):
        """The portable format is a bare-ish dict without optimiser state; both
        shapes go through the same guard."""
        portable = stage3_artifacts / "stage3_checkpoints" / "llm_stage3_best_portable.pth"
        models = load_stage3_models(stage3_config, None, str(portable))
        assert {layer.mlp.routing_mode for layer in models.llm.model.layers} == {"soft"}

    def test_a_stage_3_checkpoint_that_matches_nothing_raises(self, stage3_config, tmp_path):
        bogus = tmp_path / "bogus_stage3.pth"
        torch.save({"model_state_dict": {"nonsense": torch.zeros(3)}}, bogus)

        with pytest.raises(RuntimeError, match="matched none of the model"):
            load_stage3_models(stage3_config, None, str(bogus))

    def test_a_loaded_model_produces_a_finite_forward_pass(self, stage3_config, stage3_artifacts):
        """The end of the chain: everything above is shape and mode checking, and
        none of it proves the assembled model runs."""
        models = load_stage3_models(
            stage3_config,
            None,
            str(stage3_artifacts / "stage3_checkpoints" / "llm_stage3_best.pth"),
        )
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        with torch.no_grad():
            logits = models.llm(input_ids=input_ids, use_cache=False).logits
        assert torch.isfinite(logits).all()


class TestRoutingAblationMain:
    """The one analysis entry point the demo runs, called as a function.

    The demo drives this script as a subprocess and reads its JSON off disk.
    That is the right thing for the demo — it proves the command line works —
    but it means a failure surfaces as an exit code. Calling ``main()`` gives
    the assertion a traceback instead.
    """

    def test_main_runs_and_writes_its_results(self, fixtures_config, prior_stages, tmp_path):
        import json

        from analysis_scripts.routing_ablation_experiment import main

        paths = fixtures_config["paths"]
        results = main(
            [
                "--num-samples",
                "4",
                "--image-dir",
                paths["image_dir"],
                "--annotations",
                paths["annotations_file"],
                "--output-dir",
                str(tmp_path / "ablation"),
                "--training-config",
                _config_pointing_at(fixtures_config, prior_stages, tmp_path),
            ]
        )

        assert results["normal_routing"]["losses"], "no samples were evaluated"
        written = json.loads((tmp_path / "ablation" / "routing_ablation_results.json").read_text())
        assert written["delta"]["percent"] == pytest.approx(results["delta"]["percent"])

    def test_flipped_routing_costs_more_on_every_sample(
        self, fixtures_config, prior_stages, tmp_path
    ):
        """The demo's headline invariant, reached without a subprocess."""
        from analysis_scripts.routing_ablation_experiment import main

        paths = fixtures_config["paths"]
        results = main(
            [
                "--num-samples",
                "4",
                "--image-dir",
                paths["image_dir"],
                "--annotations",
                paths["annotations_file"],
                "--output-dir",
                str(tmp_path / "ablation"),
                "--training-config",
                _config_pointing_at(fixtures_config, prior_stages, tmp_path),
            ]
        )
        pairs = list(
            zip(
                results["normal_routing"]["losses"],
                results["flipped_routing"]["losses"],
                strict=True,
            )
        )
        assert pairs
        assert all(flipped > normal for normal, flipped in pairs)


def _config_pointing_at(fixtures_config: dict, output_dir: Path, tmp_path: Path) -> str:
    """Write a config identical to the demo's but with a chosen output_dir."""
    import yaml

    config = copy.deepcopy(fixtures_config)
    config["paths"]["output_dir"] = str(output_dir)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return str(path)
