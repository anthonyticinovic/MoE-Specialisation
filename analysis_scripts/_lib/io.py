"""Shared I/O, image preprocessing and pooling helpers for analysis scripts."""

from __future__ import annotations

import logging
import json
from pathlib import Path

import torch
from PIL import Image

logger = logging.getLogger(__name__)


def load_and_preprocess_image(image_path: str, processor) -> torch.Tensor:
    """Load an image and return a single (3, 224, 224) tensor (not batched)."""
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"]
    return pixel_values.squeeze(0)


def mean_pool_embeddings(
    embeddings: torch.Tensor, attention_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Mean pool embeddings across the sequence dimension.

    Args:
        embeddings: (batch_size, seq_len, hidden_dim)
        attention_mask: Optional mask (batch_size, seq_len)

    Returns:
        pooled: (batch_size, hidden_dim)
    """
    if attention_mask is not None:
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask
    return embeddings.mean(dim=1)


def save_json(data: dict, filepath: str) -> None:
    """Save a dict to a JSON file, creating parent directories."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved: %s", filepath)


def load_json(filepath: str) -> dict:
    """Load a JSON file."""
    with open(filepath) as f:
        return json.load(f)


def format_time(seconds: float) -> str:
    """Format seconds as a human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def log_banner(text: str, char: str = "=") -> None:
    """Log a section heading as a single record.

    One record rather than three, so the rule/title/rule cannot be split apart
    by interleaved output from another logger.
    """
    width = 80
    rule = char * width
    logger.info("\n%s\n%s\n%s\n", rule, text.center(width), rule)
