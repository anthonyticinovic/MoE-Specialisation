"""Shared utilities for training and analysis scripts.

Extracts boilerplate that was previously copy-pasted across the five training
scripts (and several analysis scripts) without changing any behaviour.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

_DEFAULT_CONFIG_PATH = "./configs/training_config.yaml"
_CONFIG_ENV_VAR = "MOE_CONFIG"


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


def load_config(path: str | None = None) -> dict[str, Any]:
    """Load training_config.yaml and return the parsed dict.

    Resolution order: explicit ``path`` argument, then the ``MOE_CONFIG``
    environment variable, then the default ``configs/training_config.yaml``.
    The environment variable is what lets the CPU demo point the unmodified
    training scripts at ``configs/demo_config.yaml`` without editing them.

    Raises FileNotFoundError with a clear message if the file is missing, and
    calls validate_config() to catch unfilled YOUR_PATH_HERE placeholders early.
    """
    if path is None:
        path = os.environ.get(_CONFIG_ENV_VAR, _DEFAULT_CONFIG_PATH)
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


def get_device() -> str:
    """Return the compute device: ``'cuda'`` when available, otherwise ``'cpu'``.

    Every device-dependent helper below keys off this, so a GPU node behaves
    exactly as it did before these helpers existed; CPU is purely a fallback
    path used by the demo and by local smoke testing.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_dist_backend() -> str:
    """Return the torch.distributed backend appropriate to the device.

    NCCL is GPU-only; a CPU run (single- or multi-process) must use gloo.
    """
    return "nccl" if torch.cuda.is_available() else "gloo"


def get_model_dtype() -> torch.dtype:
    """Return the dtype to load large models in.

    bfloat16 on GPU (as the paper runs used); float32 on CPU, where bfloat16
    matmuls are either unsupported or drastically slower.
    """
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def get_attn_implementation() -> str:
    """Return the attention kernel to request from transformers.

    FlashAttention-2 is CUDA-only and is an optional ``hpc`` extra, so CPU runs
    fall back to the portable eager implementation.
    """
    return "flash_attention_2" if torch.cuda.is_available() else "eager"


def supports_fsdp() -> bool:
    """Whether to wrap the model in FSDP.

    FSDP sharding is only meaningful (and only tested here) on multi-GPU CUDA
    runs. On CPU the demo runs the same training logic unsharded, which keeps
    the loop identical apart from the wrapper.
    """
    return torch.cuda.is_available()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module, unwrapping FSDP/DDP if present.

    FSDP exposes the wrapped model as ``.module``; an unsharded CPU model has no
    such attribute. Scripts that reach into ``llm.module.model.layers`` to touch
    the MoE layers go through this so the same line works in both cases.
    """
    return getattr(model, "module", model)


def init_distributed(local_rank: int = 0, timeout: timedelta | None = None) -> None:
    """Initialise the default process group for this run, if not already done.

    Under torchrun the standard ``env://`` rendezvous is used, exactly as
    before. A bare ``python train_stage_N.py`` (the CPU demo) has no rendezvous
    to join, so a single-rank group is created over a file store — which also
    avoids binding a TCP port, so concurrent demo runs cannot collide.
    """
    import tempfile

    import torch.distributed as dist

    if dist.is_initialized():
        return

    kwargs: dict[str, Any] = {"timeout": timeout} if timeout is not None else {}
    backend = get_dist_backend()
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend, **kwargs)
        return

    store = Path(tempfile.gettempdir()) / f"moe_pg_{os.getpid()}"
    store.unlink(missing_ok=True)
    dist.init_process_group(
        backend, init_method=f"file://{store}", rank=local_rank, world_size=1, **kwargs
    )


@contextmanager
def full_state_dict_context(
    model: torch.nn.Module, offload_to_cpu: bool = True, rank0_only: bool = True
) -> Iterator[None]:
    """Enter FSDP's FULL_STATE_DICT mode, or do nothing if the model isn't sharded.

    Checkpoint save/load must gather the sharded parameters onto rank 0 under
    FSDP. An unwrapped model (the CPU demo) already has full parameters, so the
    context is a no-op there and the surrounding save/load code is identical in
    both cases.
    """
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if isinstance(model, FSDP):
        policy = FullStateDictConfig(offload_to_cpu=offload_to_cpu, rank0_only=rank0_only)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, policy):
            yield
    else:
        yield


def register_moe_model() -> None:
    """Register custom MoE classes with HuggingFace AutoModel/AutoConfig.

    Must be called before any AutoModelForCausalLM.from_pretrained() on a
    saved MoE checkpoint. Idempotent: safe to call multiple times.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    from models.custom_mistral import MistralMoEConfig, MistralMoEForCausalLM

    AutoConfig.register("mistral_moe", MistralMoEConfig, exist_ok=True)
    AutoModelForCausalLM.register(MistralMoEConfig, MistralMoEForCausalLM, exist_ok=True)
