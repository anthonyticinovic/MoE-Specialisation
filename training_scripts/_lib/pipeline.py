"""Model, data and loss pieces shared by the training stages.

Every stage builds the same frozen CLIP tower, the same connector sized from the
loaded backbones, and the same distributed dataloaders. Those are here.

What is deliberately *not* shared is the forward pass. The stages differ in
where the ``no_grad`` and ``autocast`` boundaries fall, and those differences are
load-bearing rather than incidental:

- Stage 1 trains the connector, so it must sit outside ``no_grad``; the encoder
  and token embedding run outside ``autocast`` as well.
- Stages 2, 2.5 and the dense baseline freeze the connector and run it inside
  ``no_grad``; Stage 2 alone keeps the token embedding outside.
- Stage 3 runs only the encoder under ``no_grad``.

Folding those into one parameterised helper would need three flags and would
make an accidental change to a boundary — a GPU-only numerics change that a CPU
run cannot detect — much easier to introduce. The two genuinely identical steps,
building the combined sequence and the shifted loss, are shared below.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoProcessor, AutoTokenizer, CLIPVisionModel

from models import VisionLanguageConnector
from training_scripts._lib.runtime import RunContext

logger = logging.getLogger(__name__)


def build_backbones(paths: dict[str, Any], ctx: RunContext) -> tuple[Any, Any, Any, int]:
    """Load the frozen CLIP tower, its processor and the tokenizer.

    Returns the encoder, processor, tokenizer, and the number of visual tokens
    the tower emits — patch grid plus CLS, so 257 for ViT-L/14 at 224px and
    fewer for the demo's tiny tower. Nothing downstream should hardcode it.
    """
    if ctx.is_main:
        logger.info("Loading foundational models...")

    vision_encoder = CLIPVisionModel.from_pretrained(paths["clip_local_path"]).to(ctx.device)
    num_visual_tokens = (
        vision_encoder.config.image_size // vision_encoder.config.patch_size
    ) ** 2 + 1
    for param in vision_encoder.parameters():
        param.requires_grad = False

    clip_processor = AutoProcessor.from_pretrained(paths["clip_local_path"])
    tokenizer = AutoTokenizer.from_pretrained(paths["mistral_local_path"])
    tokenizer.pad_token = tokenizer.eos_token

    return vision_encoder, clip_processor, tokenizer, num_visual_tokens


def build_vision_connector(
    vision_encoder: Any, llm: Any, ctx: RunContext, *, trainable: bool
) -> VisionLanguageConnector:
    """Build the connector with dims taken from the loaded backbones.

    Sizing from the models rather than the CLIP-L/Mistral-7B defaults is what
    lets these scripts also run against the tiny demo models.
    """
    connector = VisionLanguageConnector(
        clip_hidden_size=vision_encoder.config.hidden_size,
        llm_hidden_size=llm.config.hidden_size,
    ).to(ctx.device)

    if not trainable:
        for param in connector.parameters():
            param.requires_grad = False
    return connector


def build_distributed_loaders(
    train_dataset: Any,
    val_dataset: Any,
    train_params: dict[str, Any],
    loader_params: dict[str, Any],
    ctx: RunContext,
    *,
    drop_last_train: bool = False,
) -> tuple[DataLoader, DataLoader, DistributedSampler]:
    """Wrap datasets in distributed samplers and loaders.

    ``drop_last_train`` matters under FSDP: a padded final batch gives ranks
    different execution paths, which FSDP reports as execution-order divergence.
    Validation never updates FSDP state, so it keeps its remainder.
    """
    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=drop_last_train)
    val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)

    loader_kwargs = {
        "batch_size": train_params["batch_size"],
        "num_workers": loader_params["num_workers"],
        "pin_memory": ctx.on_gpu,
    }
    train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, **loader_kwargs)
    return train_loader, val_loader, train_sampler


def move_batch(batch: tuple, device: str | int) -> tuple[tuple, bool]:
    """Move a batch to the device and report whether it carries LLaVA labels.

    LLaVA yields four tensors — the labels have question tokens masked to -100 —
    while COCO yields three and the loss covers all text.
    """
    moved = tuple(tensor.to(device) for tensor in batch)
    return moved, len(batch) == 4


def combine_sequence(
    visual_soft_tokens: torch.Tensor,
    text_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    ctx: RunContext,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepend the projected image to the text and extend the attention mask.

    Identical in every stage: the visual tokens are always fully attended, so
    the mask is extended with ones.
    """
    combined_embeddings = torch.cat([visual_soft_tokens, text_embeddings], dim=1)
    combined_attention_mask = torch.cat(
        [torch.ones(visual_soft_tokens.shape[:2], device=ctx.device), attention_mask], dim=1
    )
    return combined_embeddings, combined_attention_mask


def shifted_caption_loss(
    loss_fn: nn.Module,
    logits: torch.Tensor,
    num_visual_tokens: int,
    input_ids: torch.Tensor,
    vocab_size: int,
    *,
    device: str | int,
) -> torch.Tensor:
    """Next-token loss over the caption, dropping the visual prefix.

    Shared by Stage 2, Stage 2.5 and the dense baseline, which align the loss
    identically. Stage 1 uses a different offset (it starts one position earlier
    and so also scores the first caption token) and Stage 3 honours LLaVA's
    masked labels, so both keep their own version.

    A batch whose logits and labels disagree on length contributes zero rather
    than raising: under FSDP a rank that errors out while its peers continue
    deadlocks the whole job on the next collective.
    """
    text_logits = logits[..., num_visual_tokens:-1, :].contiguous()
    text_labels = input_ids[..., 1:].contiguous()

    if text_logits.shape[1] != text_labels.shape[1]:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return loss_fn(text_logits.view(-1, vocab_size), text_labels.view(-1))
