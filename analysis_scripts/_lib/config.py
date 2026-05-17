"""Configuration loading for analysis scripts.

Thin layer over ``models.utils.common`` for the training YAML, plus a generic
JSON loader for the per-analysis config files in ``configs/``.
"""

from __future__ import annotations

import json
from typing import Any

from models.utils.common import load_config as _load_training_config

_DEFAULT_TRAINING_CONFIG = "configs/training_config.yaml"


def load_training_config(path: str = _DEFAULT_TRAINING_CONFIG) -> dict[str, Any]:
    """Load and validate ``training_config.yaml`` (placeholder check included)."""
    return _load_training_config(path)


def get_paths(path: str = _DEFAULT_TRAINING_CONFIG) -> dict[str, str]:
    """Return just the ``paths:`` section of the training config."""
    return load_training_config(path)["paths"]


def load_analysis_config(
    config_file: str,
    required_fields: list[str] | None = None,
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a per-analysis JSON config, validating required keys and applying defaults.

    Args:
        config_file: Path to the JSON config file.
        required_fields: Keys that must be present; a missing key raises ValueError.
        defaults: Optional-field defaults applied via ``dict.setdefault``.
    """
    with open(config_file) as f:
        config = json.load(f)

    for field in required_fields or []:
        if field not in config:
            raise ValueError(
                f"Config file missing required field: {field}\nRequired fields: {required_fields}"
            )

    for key, value in (defaults or {}).items():
        config.setdefault(key, value)

    return config
