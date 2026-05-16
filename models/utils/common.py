"""Shared utilities for training and analysis scripts.

Extracts boilerplate that was previously copy-pasted across the five training
scripts (and several analysis scripts) without changing any behaviour.
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

_DEFAULT_CONFIG_PATH = "./configs/training_config.yaml"


def setup_logging(rank: int | None = None) -> logging.Logger:
    """Configure stdlib logging for training/analysis scripts.

    In single-GPU scripts pass rank=None (or omit it).
    In multi-GPU scripts pass local_rank after dist.init_process_group so that
    non-zero ranks are silenced to WARNING, preserving the current implicit
    "only rank 0 prints" convention.

    Returns the root logger; individual modules should use
    `logging.getLogger(__name__)` as usual.
    """
    level = logging.INFO if (rank is None or rank == 0) else logging.WARNING
    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    datefmt = "%H:%M:%S"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)
    else:
        # Replace any existing handlers so re-calling is safe
        root.handlers = [handler]
    return root


def load_config(path: str = _DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load training_config.yaml and return the parsed dict.

    Raises FileNotFoundError with a clear message if the file is missing, and
    calls validate_config() to catch unfilled YOUR_PATH_HERE placeholders early.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path!r}\n"
            "Run scripts from the repo root or pass the correct path."
        )
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    """Raise ValueError listing every unfilled YOUR_PATH_HERE placeholder."""
    unfilled = []
    paths = cfg.get("paths", {})
    for key, value in paths.items():
        if isinstance(value, str) and "YOUR_PATH_HERE" in value:
            unfilled.append(f"  paths.{key}: {value!r}")
    if unfilled:
        raise ValueError(
            "Config has unfilled placeholders — edit configs/training_config.yaml:\n"
            + "\n".join(unfilled)
        )


def set_seed(seed: int = 42) -> None:
    """Seed all RNGs for reproducibility.

    Verbatim from train_stage_1.py — do not modify this body without re-running
    the oracle (tests/test_training_dry_run.py) to confirm numeric identity.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def register_moe_model() -> None:
    """Register custom MoE classes with HuggingFace AutoModel/AutoConfig.

    Must be called before any AutoModelForCausalLM.from_pretrained() on a
    saved MoE checkpoint. Idempotent: safe to call multiple times.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    from models.custom_mistral import MistralMoEConfig, MistralMoEForCausalLM

    AutoConfig.register("mistral_moe", MistralMoEConfig, exist_ok=True)
    AutoModelForCausalLM.register(MistralMoEConfig, MistralMoEForCausalLM, exist_ok=True)
