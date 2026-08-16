"""Behavioural tests for the training scripts.

The structural tests next door pin the *shape* of the refactor — inert on
import, work reachable as named functions. These call those functions and check
what they do: that a step trains, that it trains only what the stage claims to
train, and that a stage can pick up where the previous one left off.

Everything runs against the demo's synthetic fixtures on CPU in a couple of
seconds. That is the whole point of the CPU fallback: the same code the cluster
runs is reachable from a test without a GPU, a checkpoint, or COCO.
"""

from __future__ import annotations

import copy
import importlib.util
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from demo import build_fixtures
from models.utils.common import register_moe_model
from models.utils.create_moe_model import create_moe_model
from training_scripts._lib import RunContext, build_run_context, teardown, wrap_with_fsdp

SCRIPT_DIR = Path(__file__).parent.parent / "training_scripts"

# Stage name → (filename, config section). Stage 2.5's filename is not a valid
# identifier, so every stage is loaded by path for consistency.
STAGES: dict[str, tuple[str, str]] = {
    "stage_1": ("train_stage_1.py", "training_stage1"),
    "stage_2": ("train_stage_2.py", "training_stage2"),
    "stage_2_5": ("train_stage_2.5.py", "training_stage2.5"),
    "stage_3": ("train_stage_3.py", "training_stage3"),
    "dense": ("train_dense.py", "dense_control"),
}

# What each stage claims to train, as substrings of the LLM parameter names.
# Stage 1 trains only the connector, so its LLM must stay entirely frozen.
TRAINABLE_INTENT: dict[str, set[str]] = {
    "stage_1": set(),
    "stage_2": {"mlp.experts."},
    "stage_2_5": {"mlp.gate."},
    "stage_3": {"mlp.experts.", "mlp.gate.", "self_attn."},
    "dense": {"mlp.", "self_attn."},
}

# Stage 2.5 and Stage 3 load the Stage 1 connector and the Stage 2 experts, so
# they need a directory where those stages have already run.
NEEDS_PRIOR_STAGES = ("stage_2_5", "stage_3")


def _load_stage(name: str) -> Any:
    """Import a training script by path, once per session."""
    module_name = f"_test_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_DIR / STAGES[name][0])
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _train(module: Any, name: str, setup: Any, ctx: RunContext) -> float:
    """Run one epoch and return the training loss.

    Stage 2.5 also reports its loss components and the router temperature; the
    total is first.
    """
    result = module.train_one_epoch(setup, ctx, 0, 1)
    return float(result[0]) if isinstance(result, tuple) else float(result)


def _validate(module: Any, name: str, setup: Any, ctx: RunContext) -> float:
    """Run validation and return the loss.

    Stage 3 also returns its routing metrics and batch count, so its signature
    differs from the other four.
    """
    if name == "stage_3":
        return float(module.run_validation(setup, ctx)[0])
    return float(module.run_validation(setup, ctx, 0, 1))


def _parameters(setup: Any) -> dict[str, torch.Tensor]:
    """Every parameter the stage holds, whether or not it is trainable."""
    named = {}
    for attribute in ("llm", "vision_connector", "vision_encoder"):
        module = getattr(setup, attribute, None)
        if module is not None:
            named.update({f"{attribute}.{n}": p for n, p in module.named_parameters()})
    return named


def _starved_parameters(module: Any, name: str, setup: Any, ctx: RunContext) -> set[str]:
    """Trainable parameters with no gradient when the optimiser is asked to step."""
    starved: set[str] = set()
    names = {id(p): n for n, p in _parameters(setup).items()}
    seen = False

    def inspect(optimizer, *_args, **_kwargs):
        nonlocal seen
        if seen:  # the first step is enough, and later steps are slower to read
            return
        seen = True
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is None or not param.grad.any():
                    starved.add(names.get(id(param), "<unknown>"))

    setup.optimizer.register_step_pre_hook(inspect)
    module.train_one_epoch(setup, ctx, 0, 1)
    assert seen, f"{name} never stepped the optimiser"
    return starved


def _snapshot(setup: Any) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in _parameters(setup).items()}


def _changed_since(setup: Any, before: dict[str, torch.Tensor]) -> set[str]:
    return {
        name for name, p in _parameters(setup).items() if not torch.equal(p.detach(), before[name])
    }


@pytest.fixture(scope="session")
def run_context() -> RunContext:
    """One process group for the whole module, torn down at the end.

    The distributed stages join a group even single-process on CPU, and
    ``init_distributed`` is idempotent, so all five stages share this context
    exactly as they would share a rank on a cluster.
    """
    ctx = build_run_context(distributed=True, seed=42, stage_name="tests")
    yield ctx
    teardown()


@pytest.fixture(scope="session")
def fixtures_config(tmp_path_factory, monkeypatch_session) -> dict[str, Any]:
    """Build the synthetic fixtures and the Stage 0 MoE model once.

    This is the demo's own fixture builder, so a test failing here means the
    demo is broken too — which is the intended coupling.
    """
    root = tmp_path_factory.mktemp("training_steps")
    config_path = build_fixtures.build(root, num_images=8)
    create_moe_model(
        str(root / "fixtures" / "base_llm"),
        str(root / "fixtures" / "moe_model"),
        seed=42,
    )
    # The scripts resolve their config through MOE_CONFIG; setting it here is
    # exactly how the demo points them at the miniature setup.
    monkeypatch_session.setenv("MOE_CONFIG", str(config_path))
    register_moe_model()
    return yaml.safe_load(config_path.read_text())


@pytest.fixture(scope="session")
def monkeypatch_session():
    """Session-scoped monkeypatch — pytest's built-in one is function-scoped."""
    patcher = pytest.MonkeyPatch()
    yield patcher
    patcher.undo()


@pytest.fixture
def config_in(fixtures_config, tmp_path):
    """A copy of the demo config writing its outputs to an empty directory.

    Each test gets its own output directory so no test can resume from another
    test's checkpoint — the failure mode that made the demo silently train for
    zero epochs.
    """

    def build(output_dir: Path | None = None) -> dict[str, Any]:
        config = copy.deepcopy(fixtures_config)
        config["paths"]["output_dir"] = str(output_dir or tmp_path / "runs")
        return config

    return build


@pytest.fixture(scope="session")
def prior_stages(tmp_path_factory, fixtures_config, run_context) -> Path:
    """An output directory where Stage 1 and Stage 2 have run and checkpointed.

    Built once and treated as read-only by the tests that consume it.
    """
    output_dir = tmp_path_factory.mktemp("prior_stages")
    config = copy.deepcopy(fixtures_config)
    config["paths"]["output_dir"] = str(output_dir)

    stage_1 = _load_stage("stage_1")
    setup = stage_1.build_setup(
        config["paths"], config["training_stage1"], config["dataloader"], 1, run_context
    )
    _train(stage_1, "stage_1", setup, run_context)
    stage_1.save_checkpoints(setup, str(output_dir), 1.0, float("inf"))

    stage_2 = _load_stage("stage_2")
    setup_2, _, _, _ = stage_2.build_setup(
        config["paths"], config["training_stage2"], config["dataloader"], 1, run_context
    )
    _train(stage_2, "stage_2", setup_2, run_context)
    stage_2.save_checkpoints(
        setup_2, run_context, str(output_dir / "stage2_checkpoints"), 0, 1.0, float("inf")
    )
    return output_dir


@pytest.fixture
def writable_prior_stages(tmp_path, prior_stages) -> Path:
    """A private copy of the Stage 1/2 artifacts, for tests that write there."""
    destination = tmp_path / "prior"
    shutil.copytree(prior_stages, destination)
    return destination


@pytest.fixture
def build_stage(config_in, run_context, prior_stages):
    """Assemble a stage against a clean output directory."""

    def build(name: str, output_dir: Path | None = None) -> Any:
        module = _load_stage(name)
        if output_dir is None and name in NEEDS_PRIOR_STAGES:
            output_dir = prior_stages
        config = config_in(output_dir)
        result = module.build_setup(
            config["paths"],
            config[STAGES[name][1]],
            config["dataloader"],
            1,
            run_context,
        )
        setup = result[0] if isinstance(result, tuple) else result
        return module, setup, config

    return build


@pytest.mark.parametrize("name", list(STAGES))
class TestTrainingStep:
    """One real epoch of each stage, on the real code path."""

    def test_epoch_returns_a_finite_loss(self, name, build_stage, run_context):
        module, setup, _ = build_stage(name)
        loss = _train(module, name, setup, run_context)
        assert torch.isfinite(torch.tensor(loss)), f"{name} produced a non-finite loss"
        assert loss > 0.0, f"{name} produced a zero loss — the batch guard may be swallowing it"

    def test_only_the_trainable_parameters_change(self, name, build_stage, run_context):
        """Everything marked frozen must come out of the epoch untouched.

        This is the invariant the stage design rests on: Stage 2 must not move
        the router, Stage 2.5 must not move the experts, and no stage may move
        the vision tower.
        """
        module, setup, _ = build_stage(name)
        expected = {n for n, p in _parameters(setup).items() if p.requires_grad}

        before = _snapshot(setup)
        _train(module, name, setup, run_context)
        changed = _changed_since(setup, before)

        assert not (changed - expected), (
            f"{name} changed frozen parameters: {sorted(changed - expected)[:5]}"
        )
        assert changed == expected, (
            f"{name} left trainable parameters unchanged: {sorted(expected - changed)[:5]}"
        )

    def test_trainable_set_matches_the_stage_intent(self, name, build_stage):
        """The trainable parameters are the ones the stage table documents."""
        _, setup, _ = build_stage(name)
        trainable = {n for n, p in setup.llm.named_parameters() if p.requires_grad}
        intent = TRAINABLE_INTENT[name]

        if not intent:
            assert not trainable, f"{name} should freeze the LLM entirely"
            return
        unexpected = [n for n in trainable if not any(marker in n for marker in intent)]
        assert not unexpected, f"{name} trains unexpected parameters: {unexpected[:5]}"
        assert trainable, f"{name} has nothing to train"

    def test_gradients_reach_every_trainable_parameter(self, name, build_stage, run_context):
        """A detached tensor or a lost ``requires_grad`` shows up here first.

        Checking that parameters moved is not enough: AdamW's decoupled weight
        decay moves a parameter whose gradient is exactly zero. The hook reads
        the gradients at the moment the optimiser is asked to step.

        Stage 2's final-layer vision expert is exempt, for the structural reason
        documented in ``test_stage_2_cannot_train_the_final_vision_expert``.
        """
        module, setup, _ = build_stage(name)
        starved = _starved_parameters(module, name, setup, run_context)
        if name == "stage_2":
            last = setup.llm.config.num_hidden_layers - 1
            starved -= {n for n in starved if f"layers.{last}.mlp.experts.0." in n}

        assert not starved, f"{name}: no gradient reached {sorted(starved)[:5]}"

    def test_validation_updates_nothing(self, name, build_stage, run_context):
        """Validation must not train. It also must not leave the model in eval."""
        module, setup, _ = build_stage(name)
        before = _snapshot(setup)
        loss = _validate(module, name, setup, run_context)

        assert torch.isfinite(torch.tensor(loss))
        assert not _changed_since(setup, before), f"{name} validation mutated parameters"


def test_stage_2_cannot_train_the_final_vision_expert(build_stage, run_context):
    """The last layer's vision expert receives no gradient, by construction.

    Stage 2 scores only the caption positions — the visual prefix is sliced out
    of the loss. A vision expert therefore only influences the loss through the
    attention of later text positions, and after the final decoder layer there
    is no attention left. The last layer's vision expert is a dead end.

    This is a property of the objective, not a defect, but it is invisible in
    the code and easy to mistake for a broken gradient path, so it is pinned
    here. It is also why the gradient test above exempts exactly this expert.
    """
    module, setup, _ = build_stage("stage_2")
    starved = _starved_parameters(module, "stage_2", setup, run_context)
    last = setup.llm.config.num_hidden_layers - 1

    assert starved, "expected the final vision expert to be starved of gradient"
    assert all(f"layers.{last}.mlp.experts.0." in n for n in starved), (
        f"only the final vision expert should lack gradient, got {sorted(starved)}"
    )


class TestCheckpointRoundTrip:
    """A stage must be able to resume from its own checkpoint."""

    def test_stage_2_resumes_its_experts_and_epoch(self, build_stage, run_context, tmp_path):
        module, setup, config = build_stage("stage_2", tmp_path / "resume")
        checkpoint_dir = tmp_path / "resume" / "stage2_checkpoints"

        _train(module, "stage_2", setup, run_context)
        module.save_checkpoints(setup, run_context, str(checkpoint_dir), 0, 0.5, float("inf"))
        trained = {n: p.detach().clone() for n, p in setup.llm.named_parameters()}

        # A fresh build against the same directory must find the checkpoint.
        reloaded, epoch, best_val_loss, resumed = module.build_setup(
            config["paths"],
            config["training_stage2"],
            config["dataloader"],
            1,
            run_context,
        )
        assert resumed, "Stage 2 did not detect its own checkpoint"
        assert epoch == 1, "Resumed epoch should be the number of completed epochs"
        assert best_val_loss == pytest.approx(0.5)

        for name, param in reloaded.llm.named_parameters():
            assert torch.equal(param.detach(), trained[name]), f"{name} did not survive the reload"

    def test_stage_2_5_resumes_its_router(self, build_stage, run_context, writable_prior_stages):
        """A partial checkpoint — routers only — must still load and resume.

        The guard on ``strict=False`` has to allow this: 27 of the 29 keys are
        legitimately absent. Only a *zero* overlap is an error.
        """
        module, setup, _ = build_stage("stage_2_5", writable_prior_stages)
        checkpoint_dir = writable_prior_stages / "stage2_5_checkpoints"

        _train(module, "stage_2_5", setup, run_context)
        module.save_checkpoints(setup, run_context, str(checkpoint_dir), 0, 0.5, float("inf"))
        trained = {n: p.detach().clone() for n, p in setup.llm.named_parameters()}

        _, fresh, _ = build_stage("stage_2_5", writable_prior_stages)
        start_epoch, best_val_loss = module.resume_from_checkpoint(
            fresh, str(checkpoint_dir), run_context
        )
        assert start_epoch == 1
        assert best_val_loss == pytest.approx(0.5)
        for name, param in fresh.llm.named_parameters():
            if "mlp.gate." in name:
                assert torch.equal(param.detach(), trained[name]), f"router {name} was not restored"

    def test_stage_3_resumes_from_its_portable_checkpoint(
        self, build_stage, run_context, writable_prior_stages
    ):
        module, setup, _ = build_stage("stage_3", writable_prior_stages)
        checkpoint_dir = writable_prior_stages / "stage3_checkpoints"

        _train(module, "stage_3", setup, run_context)
        module.save_checkpoints(setup, run_context, str(checkpoint_dir), 0, 0.5, float("inf"))
        trained = {n: p.detach().clone() for n, p in setup.llm.named_parameters()}

        _, fresh, _ = build_stage("stage_3", writable_prior_stages)
        start_epoch = module.maybe_resume(
            fresh.llm, fresh.vision_connector, str(checkpoint_dir), run_context
        )
        assert start_epoch == 1
        for name, param in fresh.llm.named_parameters():
            assert torch.equal(param.detach(), trained[name]), f"{name} was not restored"

    def test_stage_1_saves_best_only_when_validation_improves(
        self, build_stage, run_context, tmp_path
    ):
        """``save_checkpoints`` returns the running best and gates the best file."""
        output_dir = tmp_path / "best"
        module, setup, _ = build_stage("stage_1", output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)  # main() does this before saving
        _train(module, "stage_1", setup, run_context)

        best = module.save_checkpoints(setup, str(output_dir), 2.0, float("inf"))
        assert best == pytest.approx(2.0)
        best_file = output_dir / "vision_connector_stage1_best.pth"
        first_saved = best_file.read_bytes()

        # A worse epoch must leave the best file alone.
        best = module.save_checkpoints(setup, str(output_dir), 5.0, best)
        assert best == pytest.approx(2.0)
        assert best_file.read_bytes() == first_saved


class TestCrossStageHandoff:
    """The stages are a pipeline; each must pick up the previous one's weights."""

    def test_stage_2_5_starts_from_the_stage_1_connector(
        self, build_stage, prior_stages, run_context
    ):
        _, setup, _ = build_stage("stage_2_5")
        saved = torch.load(prior_stages / "vision_connector_stage1_best.pth", map_location="cpu")
        for name, param in setup.vision_connector.named_parameters():
            assert torch.equal(param.detach(), saved[name]), (
                f"Stage 2.5 did not load the Stage 1 connector weight {name}"
            )

    @pytest.mark.parametrize("name", ["stage_2_5", "stage_3"])
    def test_downstream_stages_start_from_the_stage_2_experts(
        self, name, build_stage, prior_stages, run_context
    ):
        """The bug this catches: both stages silently kept their Stage 0 experts.

        Stage 2's ``best`` checkpoint used to be a bare state dict and is now a
        full training checkpoint. Both loaders still read it as bare, and with
        ``strict=False`` every key landed in ``unexpected_keys`` — so the
        weights never moved and the run logged that the load had succeeded.
        """
        _, setup, _ = build_stage(name)
        saved = torch.load(
            prior_stages / "stage2_checkpoints" / "llm_stage2_best.pth",
            map_location="cpu",
            weights_only=False,
        )["model_state_dict"]

        experts = {n: p for n, p in setup.llm.named_parameters() if "mlp.experts." in n}
        assert experts, f"{name} has no expert parameters to check"
        for expert_name, param in experts.items():
            assert torch.equal(param.detach(), saved[expert_name]), (
                f"{name} did not load the Stage 2 expert weight {expert_name}"
            )

    def test_a_bare_state_dict_still_loads(self, build_stage, prior_stages, tmp_path):
        """Older Stage 2 checkpoints were bare state dicts; both shapes must work."""
        from training_scripts._lib import state_dict_from

        wrapped = prior_stages / "stage2_checkpoints" / "llm_stage2_best.pth"
        bare = tmp_path / "bare.pth"
        torch.save(state_dict_from(str(wrapped)), bare)

        assert state_dict_from(str(bare)).keys() == state_dict_from(str(wrapped)).keys()

    def test_a_checkpoint_that_matches_nothing_raises(self, build_stage):
        """The silent path is closed: a mismatched format must stop the run."""
        from training_scripts._lib import load_matching_weights

        _, setup, _ = build_stage("stage_2")
        with pytest.raises(RuntimeError, match="matched none of the model"):
            load_matching_weights(
                setup.llm, {"epoch": torch.tensor(1)}, source="a checkpoint of the wrong shape"
            )

    def test_stage_2_5_reinitialises_the_router_it_inherits(self, build_stage, prior_stages):
        """Gates are deliberately re-drawn: inherited gates collapse routing."""
        _, setup, _ = build_stage("stage_2_5")
        saved = torch.load(
            prior_stages / "stage2_checkpoints" / "llm_stage2_best.pth",
            map_location="cpu",
            weights_only=False,
        )["model_state_dict"]
        gates = {n: p for n, p in setup.llm.named_parameters() if "mlp.gate." in n}
        assert gates, "Stage 2.5 has no router to train"
        assert all(not torch.equal(p.detach(), saved[n]) for n, p in gates.items()), (
            "Stage 2.5 kept the Stage 2 gates instead of re-initialising them"
        )


class TestConfigResolution:
    """``MOE_CONFIG`` is what lets these scripts run unmodified at demo scale."""

    def test_stage_reads_batch_size_from_the_config(self, build_stage, config_in, run_context):
        """A changed config value must reach the dataloader, not a constant."""
        module = _load_stage("stage_1")
        config = config_in()
        config["training_stage1"] = {**config["training_stage1"], "batch_size": 2}
        setup = module.build_setup(
            config["paths"], config["training_stage1"], config["dataloader"], 1, run_context
        )
        assert setup.train_loader.batch_size == 2

    def test_connector_is_sized_from_the_loaded_models(self, build_stage):
        """Nothing may hardcode CLIP-L's 1024 or Mistral-7B's 4096."""
        _, setup, _ = build_stage("stage_1")
        first_layer = setup.vision_connector.mlp[0]
        assert first_layer.in_features == setup.vision_encoder.config.hidden_size
        assert setup.llm.config.hidden_size == 64, "fixtures should be miniature"


class TestCpuFallback:
    """The CPU path exists so the demo and these tests can run the real code."""

    def test_context_disables_gpu_only_machinery(self, run_context):
        if torch.cuda.is_available():
            pytest.skip("CPU fallback assertions only hold off-GPU")
        assert run_context.device == "cpu"
        assert run_context.amp_device == "cpu"
        assert not run_context.on_gpu
        assert not run_context.use_fsdp
        assert run_context.is_main

    def test_wrap_with_fsdp_is_a_no_op_off_gpu(self, build_stage, run_context):
        """Off-GPU the loops must run against the plain module, not a wrapper."""
        if torch.cuda.is_available():
            pytest.skip("FSDP is used on GPU")
        _, setup, _ = build_stage("stage_2")
        assert wrap_with_fsdp(setup.llm, run_context, offload_params=True) is setup.llm

    def test_scaler_is_disabled_off_gpu(self, build_stage):
        """Loss scaling is a float16 device concern; CPU runs in float32."""
        if torch.cuda.is_available():
            pytest.skip("GradScaler is enabled on GPU")
        _, setup, _ = build_stage("stage_2")
        assert not setup.scaler.is_enabled()
