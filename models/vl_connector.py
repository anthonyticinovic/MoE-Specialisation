import torch.nn as nn


class VisionLanguageConnector(nn.Module):
    """Two-layer MLP projecting CLIP visual embeddings into the LLM's latent space."""

    def __init__(
        self, clip_hidden_size: int = 1024, llm_hidden_size: int = 4096, dropout_rate: float = 0.3
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(clip_hidden_size, llm_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

    def forward(self, visual_features):
        return self.mlp(visual_features)
