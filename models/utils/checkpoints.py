"""Reading checkpoints written by an earlier stage.

Stage 2's ``best`` checkpoint was originally a bare state dict, and the loaders
in Stages 2.5 and 3 were written against that shape. It later became a full
training checkpoint with the weights nested under ``model_state_dict``, and the
loaders were not updated. Nothing raised: they load with ``strict=False``, so
every top-level key landed in ``unexpected_keys``, the model silently kept its
Stage 0 weights, and the run logged that the checkpoint had loaded.

These two helpers close that hole. ``state_dict_from`` accepts either shape, and
``load_matching_weights`` refuses to continue when a non-empty state dict
matches nothing — the case that used to pass unnoticed.

They live in the core rather than in ``training_scripts/_lib`` because the
analysis scripts read the same checkpoint files and can fail the same way;
``training_scripts._lib`` re-exports them so the stages import one name.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def state_dict_from(path: str, map_location: Any = "cpu") -> dict[str, torch.Tensor]:
    """Return the tensor state dict from a checkpoint, wrapped or bare."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload


def load_matching_weights(
    model: nn.Module, state_dict: dict[str, torch.Tensor], *, source: str
) -> tuple[list[str], list[str]]:
    """``load_state_dict(strict=False)`` that fails when nothing matched.

    ``strict=False`` is genuinely needed: a stage loads a subset of the model
    (Stage 3 takes the experts, not the attention) and under FSDP's
    ``rank0_only`` gather the non-zero ranks are handed an empty dict on
    purpose. What it must not do is accept a state dict that overlaps the model
    in *no* key, which is what a changed checkpoint format looks like.

    Args:
        model: Target module, FSDP-wrapped or plain.
        state_dict: Weights to apply. Empty is allowed — that is the
            ``rank0_only`` case, where the other ranks intentionally load
            nothing.
        source: Path or description used in the error, so a failure says which
            checkpoint disagreed with the model.
    """
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if state_dict and len(missing) == len(model.state_dict()):
        raise RuntimeError(
            f"{source} matched none of the model's {len(missing)} parameters. "
            f"Its top-level keys are {sorted(state_dict)[:6]}. This is what a "
            "changed checkpoint format looks like: loading would have left the "
            "model on its previous weights while reporting success."
        )

    if unexpected:
        logger.warning(
            "%s carried %d key(s) the model does not have: %s",
            source,
            len(unexpected),
            unexpected[:5],
        )
    return missing, unexpected
