"""Driving the real training scripts from a test.

The stage scripts are files, not an importable package — Stage 2.5's name is not
a valid identifier — so they are loaded by path. Anything that needs to *run* a
stage goes through here rather than importing a script directly, so the training
tests and the analysis tests drive the pipeline the same way.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).parent.parent / "training_scripts"

# Stage name → (filename, config section).
STAGES: dict[str, tuple[str, str]] = {
    "stage_1": ("train_stage_1.py", "training_stage1"),
    "stage_2": ("train_stage_2.py", "training_stage2"),
    "stage_2_5": ("train_stage_2.5.py", "training_stage2.5"),
    "stage_3": ("train_stage_3.py", "training_stage3"),
    "dense": ("train_dense.py", "dense_control"),
}

# Stages that load an earlier stage's checkpoints and so need a directory where
# those stages have already run.
NEEDS_PRIOR_STAGES = ("stage_2_5", "stage_3")


def load_stage(name: str) -> Any:
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


def train_epoch(module: Any, setup: Any, ctx: Any) -> float:
    """Run one epoch and return the training loss.

    Stage 2.5 also reports its loss components and the router temperature; the
    total is first.
    """
    result = module.train_one_epoch(setup, ctx, 0, 1)
    return float(result[0]) if isinstance(result, tuple) else float(result)


def build_setup(module: Any, name: str, config: dict[str, Any], ctx: Any) -> Any:
    """Call a stage's ``build_setup``, unwrapping the stages that return a tuple."""
    result = module.build_setup(
        config["paths"],
        config[STAGES[name][1]],
        config["dataloader"],
        1,
        ctx,
    )
    return result[0] if isinstance(result, tuple) else result
