"""Shared library for the training scripts.

Mirrors ``analysis_scripts/_lib``: code used by more than one stage, or large
enough to crowd a stage script, lives here rather than being copy-pasted.
"""

from training_scripts._lib.checkpoints import load_matching_weights, state_dict_from
from training_scripts._lib.expert_metrics import ExpertUsageTracker, save_expert_metrics
from training_scripts._lib.pipeline import (
    build_backbones,
    build_distributed_loaders,
    build_vision_connector,
    combine_sequence,
    move_batch,
    shifted_caption_loss,
)
from training_scripts._lib.runtime import (
    RunContext,
    broadcast_flag,
    build_run_context,
    teardown,
    wrap_with_fsdp,
)

__all__ = [
    "ExpertUsageTracker",
    "RunContext",
    "broadcast_flag",
    "build_backbones",
    "build_distributed_loaders",
    "build_run_context",
    "build_vision_connector",
    "combine_sequence",
    "load_matching_weights",
    "move_batch",
    "save_expert_metrics",
    "state_dict_from",
    "shifted_caption_loss",
    "teardown",
    "wrap_with_fsdp",
]
