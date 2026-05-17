"""The vision→language projection trained in Stage 1."""

import torch
import torch.nn as nn


class VisionLanguageConnector(nn.Module):
    """Two-layer MLP projecting CLIP visual embeddings into the LLM's latent space.

    This is the only trainable module in Stage 1 (CLIP and the LLM are frozen).
    The default dimensions are fixed by the chosen backbones rather than tuned:

    - ``clip_hidden_size=1024`` — CLIP ViT-L/14 patch-embedding width.
    - ``llm_hidden_size=4096`` — Mistral-7B hidden size; the output must match
      the token-embedding width so projected visual tokens can be concatenated
      with text-token embeddings.

    The hidden layer is kept at the LLM width (4096→4096) so the connector has
    enough capacity to align modalities without a bottleneck. ``dropout_rate``
    defaults to ``0.3``: comparatively high because Stage 1 trains a small
    module on limited caption data and overfits easily.
    """

    def __init__(
        self, clip_hidden_size: int = 1024, llm_hidden_size: int = 4096, dropout_rate: float = 0.3
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(clip_hidden_size, llm_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """Project ``(..., clip_hidden_size)`` features to ``(..., llm_hidden_size)``."""
        return self.mlp(visual_features)
