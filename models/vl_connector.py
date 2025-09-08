import torch.nn as nn

# nn.Module is the base class for all neural network modules in PyTorch.
# This gives our class acces to the functionality of a standard layer like parameter tracking, etc.
class VisionLanguageConnector(nn.Module):
    """
    A two-layer MLP to project CLIP's visual embeddings into the
    LLM's (Mistral 7B) latent space.
    """
    def __init__(self, clip_hidden_size: int = 1024, llm_hidden_size: int = 4096, dropout_rate: float = 0.1):
        super().__init__()

        # nn.Sequential is a container that allows us to stack layers in a sequence.
        self.mlp = nn.Sequential(
            nn.Linear(clip_hidden_size, llm_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

    # visual_features is the output of the CLIP model.
    # It's a tensor of shape (batch_size, clip_hidden_size). Passed through our mlp. 
    def forward(self, visual_features):
        return self.mlp(visual_features)