"""Shared pytest fixtures and global test configuration."""

import os

import pytest
import torch

# Isolate from HuggingFace Hub — tests must never download weights.
os.environ["HF_HUB_OFFLINE"] = "1"
# Deterministic single-threaded ops for reproducibility on CI.
torch.set_num_threads(1)

from models import MistralMoEConfig, MistralMoEForCausalLM  # noqa: E402


@pytest.fixture(scope="session")
def tiny_config() -> MistralMoEConfig:
    """Minimal synthetic config — never loads real Mistral-7B weights."""
    return MistralMoEConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )


@pytest.fixture
def tiny_model(tiny_config) -> MistralMoEForCausalLM:
    torch.manual_seed(0)
    return MistralMoEForCausalLM(tiny_config).to(torch.float32)


# ---------------------------------------------------------------------------
# Pipeline fixtures: real stages run against the demo's synthetic fixtures.
#
# Session-scoped and shared, so the training tests and the analysis-loader
# tests pay for Stage 1 and Stage 2 once between them.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def monkeypatch_session():
    """Session-scoped monkeypatch — pytest's built-in one is function-scoped."""
    patcher = pytest.MonkeyPatch()
    yield patcher
    patcher.undo()


@pytest.fixture(scope="session")
def run_context():
    """One process group for the whole session, torn down at the end.

    The distributed stages join a group even single-process on CPU, and
    ``init_distributed`` is idempotent, so every stage shares this context
    exactly as they would share a rank on a cluster.
    """
    from training_scripts._lib import build_run_context, teardown

    ctx = build_run_context(distributed=True, seed=42, stage_name="tests")
    yield ctx
    teardown()


@pytest.fixture(scope="session")
def fixtures_config(tmp_path_factory, monkeypatch_session):
    """Build the synthetic fixtures and the Stage 0 MoE model once.

    This is the demo's own fixture builder, so a test failing here means the
    demo is broken too — which is the intended coupling.
    """
    import yaml

    from demo import build_fixtures
    from models.utils.common import register_moe_model
    from models.utils.create_moe_model import create_moe_model

    root = tmp_path_factory.mktemp("pipeline")
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
def prior_stages(tmp_path_factory, fixtures_config, run_context):
    """An output directory where Stage 1 and Stage 2 have run and checkpointed.

    Built once and treated as read-only by the tests that consume it.
    """
    import copy

    from tests.pipeline import build_setup, load_stage, train_epoch

    output_dir = tmp_path_factory.mktemp("prior_stages")
    config = copy.deepcopy(fixtures_config)
    config["paths"]["output_dir"] = str(output_dir)

    stage_1 = load_stage("stage_1")
    setup = build_setup(stage_1, "stage_1", config, run_context)
    train_epoch(stage_1, setup, run_context)
    stage_1.save_checkpoints(setup, str(output_dir), 1.0, float("inf"))

    stage_2 = load_stage("stage_2")
    setup_2 = build_setup(stage_2, "stage_2", config, run_context)
    train_epoch(stage_2, setup_2, run_context)
    stage_2.save_checkpoints(
        setup_2, run_context, str(output_dir / "stage2_checkpoints"), 0, 1.0, float("inf")
    )
    return output_dir
