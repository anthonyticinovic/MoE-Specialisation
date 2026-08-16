"""Device, distribution and sharding setup shared by every training stage.

Each stage previously repeated the same twenty-odd lines: work out the device,
initialise the process group, silence non-zero ranks, seed the RNGs, then wrap
the model in FSDP with an identical policy. That block is here once, so a change
to the CPU fallback or the wrap policy lands in all five scripts at once.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.mistral.modeling_mistral import MistralMLP

from models.utils.common import get_device, init_distributed, set_seed, setup_logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunContext:
    """Where this process runs and how it talks to its peers.

    ``on_gpu`` is the single switch behind every device-dependent choice:
    autocast, loss scaling, pinned memory, FSDP sharding and CUDA cache clears.
    On CPU all of them turn off together, which is what makes the demo able to
    run these scripts unmodified.
    """

    device: str | int
    amp_device: str
    on_gpu: bool
    local_rank: int
    distributed: bool

    @property
    def is_main(self) -> bool:
        """True on the rank that owns logging, checkpointing and metrics."""
        return self.local_rank == 0

    @property
    def use_fsdp(self) -> bool:
        """Whether to shard the model. Sharding is only meaningful on GPU."""
        return self.on_gpu


def build_run_context(
    *,
    distributed: bool,
    seed: int = 42,
    timeout: timedelta | None = None,
    stage_name: str = "",
) -> RunContext:
    """Resolve the device, join the process group, configure logging and seed.

    Args:
        distributed: True for the FSDP stages (2, 2.5, 3, dense), which join a
            process group even when running single-process on CPU. Stage 1 is
            single-GPU and passes False.
        seed: Seeds Python, NumPy and torch. Every stage seeds; without it the
            dropout, sampler order and Gumbel noise differ run to run.
        timeout: Process-group timeout; Stage 3 raises it for slow checkpoint
            gathers.
        stage_name: Used only in the CPU-fallback warning.
    """
    on_gpu = get_device() == "cuda"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if distributed:
        init_distributed(local_rank, timeout=timeout)
    setup_logging(local_rank if distributed else None)

    if on_gpu:
        torch.cuda.set_device(local_rank)
        device: str | int = local_rank if distributed else "cuda"
    else:
        device = "cpu"

    ctx = RunContext(
        device=device,
        # autocast and GradScaler take a device *type*, but `device` may be an
        # ordinal under FSDP.
        amp_device="cuda" if on_gpu else "cpu",
        on_gpu=on_gpu,
        local_rank=local_rank,
        distributed=distributed,
    )

    set_seed(seed)

    if not on_gpu and ctx.is_main:
        logger.warning(
            "CUDA not available — running %s on CPU. Mixed precision, FSDP "
            "sharding and FlashAttention are disabled. This path exists for the "
            "CPU demo and local smoke tests; real training requires a GPU.",
            stage_name or "this stage",
        )
    return ctx


def wrap_with_fsdp(
    llm: nn.Module,
    ctx: RunContext,
    *,
    offload_params: bool | None,
    ignore_embeddings: bool = True,
) -> nn.Module:
    """Shard the model across ranks, or return it unchanged on CPU.

    Args:
        offload_params: Passed straight to ``CPUOffload``. Stage 2 and the dense
            baseline use True; Stages 2.5 and 3 use None. The difference is
            load-bearing and preserved per stage rather than unified.
        ignore_embeddings: Keep ``embed_tokens`` unsharded because the training
            loops call it directly. It must already be on the target device
            before wrapping, or FSDP reports it as a newly-added parameter.
    """
    if ignore_embeddings:
        llm.model.embed_tokens.to(ctx.device)

    if not ctx.use_fsdp:
        # CPU: sharding is a no-op at one rank, so the loops run against the
        # plain module. Nothing else about them changes.
        return llm.to(ctx.device)

    return FSDP(
        llm,
        device_id=ctx.device,
        auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={MistralMLP}),
        cpu_offload=CPUOffload(offload_params=offload_params),
        mixed_precision=torch.distributed.fsdp.MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        use_orig_params=True,
        ignored_modules=[llm.model.embed_tokens] if ignore_embeddings else None,
    )


def broadcast_flag(condition: bool, ctx: RunContext) -> bool:
    """Agree across ranks on a decision rank 0 makes from the filesystem.

    Every rank must take the same branch — a rank that skips a checkpoint load
    the others perform will desynchronise the collectives that follow.
    """
    flag = torch.tensor(1.0 if (ctx.is_main and condition) else 0.0, device=ctx.device)
    dist.broadcast(flag, src=0)
    return bool(flag.item() == 1.0)


def teardown() -> None:
    """Tear the process group down at the end of a run."""
    if dist.is_initialized():
        dist.destroy_process_group()
