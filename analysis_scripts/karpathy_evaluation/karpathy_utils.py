#!/usr/bin/env python3
"""
Utility functions for Karpathy COCO evaluation.
Shared code for model loading, preprocessing, etc.
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from PIL import Image
from torchvision import transforms

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def _load_paths() -> dict:
    """Load model/data paths from training_config.yaml."""
    config_path = project_root / "configs" / "training_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)["paths"]


def load_model_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """
    Load a trained model checkpoint (Stage 2 or Stage 3).
    Returns a wrapper object containing vision encoder, connector, and LLM.

    Args:
        checkpoint_path: Path to .pth checkpoint file
        device: Device to load model on

    Returns:
        model_wrapper: Object with vision_encoder, connector, llm attributes
        processor: Vision processor (CLIP)
        tokenizer: Text tokenizer (Mistral)
    """
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoProcessor,
        AutoTokenizer,
        CLIPVisionModel,
    )

    from models import VisionLanguageConnector
    from models.custom_mistral import MistralMoEConfig, MistralMoEForCausalLM

    # Register custom architecture
    AutoConfig.register("mistral_moe", MistralMoEConfig)
    AutoModelForCausalLM.register(MistralMoEConfig, MistralMoEForCausalLM)

    print(f"\n🔄 Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    paths = _load_paths()
    moe_model_path = paths["moe_model_path"]
    clip_path = paths["clip_local_path"]
    mistral_path = paths["mistral_local_path"]
    connector_path = str(Path(paths["output_dir"]) / "vision_connector_stage1_best.pth")

    # Load vision encoder
    print(f"📦 Loading CLIP vision encoder from {clip_path}...")
    vision_encoder = CLIPVisionModel.from_pretrained(clip_path, local_files_only=True).to(device)
    vision_encoder.eval()

    # Load vision-language connector with trained weights from Stage 1
    print(f"📦 Loading vision-language connector from {connector_path}...")
    vision_connector = VisionLanguageConnector()
    connector_state_dict = torch.load(connector_path, map_location="cpu", weights_only=False)
    vision_connector.load_state_dict(connector_state_dict)
    vision_connector = vision_connector.to(device)
    vision_connector.eval()
    print("✅ Vision connector loaded successfully")

    # Load base MoE model architecture
    print(f"📦 Loading base MoE LLM from {moe_model_path}...")
    llm = AutoModelForCausalLM.from_pretrained(
        moe_model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    # Load LLM checkpoint state dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Remove FSDP/DDP wrapper prefixes if present
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "").replace("_forward_module.", "")
        cleaned_state_dict[new_key] = value

    llm.load_state_dict(cleaned_state_dict, strict=True)
    llm = llm.to(device)
    llm.eval()

    print("✅ LLM loaded successfully")
    print(f"   Device: {device}")
    print(f"   Parameters: {sum(p.numel() for p in llm.parameters()) / 1e9:.2f}B")

    # Load processor and tokenizer (same paths as training)
    print(f"📦 Loading CLIP processor from {clip_path}...")
    processor = AutoProcessor.from_pretrained(clip_path, local_files_only=True)

    print(f"📦 Loading Mistral tokenizer from {mistral_path}...")
    tokenizer = AutoTokenizer.from_pretrained(mistral_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("✅ Processor and tokenizer loaded")

    # Create a simple wrapper object
    class ModelWrapper:
        def __init__(self, vision_encoder, connector, llm):
            self.vision_encoder = vision_encoder
            self.connector = connector
            self.llm = llm

        def to(self, device):
            self.vision_encoder = self.vision_encoder.to(device)
            self.connector = self.connector.to(device)
            self.llm = self.llm.to(device)
            return self

        def eval(self):
            self.vision_encoder.eval()
            self.connector.eval()
            self.llm.eval()
            return self

    model_wrapper = ModelWrapper(vision_encoder, vision_connector, llm)

    return model_wrapper, processor, tokenizer


def get_image_transform():
    """Get standard image preprocessing transform."""
    return transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
            ),
        ]
    )


def load_and_preprocess_image(image_path: str, processor) -> torch.Tensor:
    """
    Load and preprocess a single image.

    Args:
        image_path: Path to image file
        processor: CLIP image processor

    Returns:
        preprocessed image tensor (3, 224, 224) - single image, not batched
    """
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"]
    # Return squeezed tensor: (3, 224, 224) instead of (1, 3, 224, 224)
    return pixel_values.squeeze(0)


def extract_layer_activations(
    model: nn.Module, inputs: dict[str, torch.Tensor], layer_idx: int, modality: str = "vision"
) -> torch.Tensor:
    """
    Extract activations from a specific layer.

    Args:
        model: The VLM model
        inputs: Input dict with pixel_values or input_ids
        layer_idx: Which layer to extract (0-31)
        modality: 'vision' or 'text'

    Returns:
        activations: (batch_size, seq_len, hidden_dim)
    """
    activations = {}

    def hook_fn(module, input, output):
        # output is tuple: (hidden_states, ...)
        if isinstance(output, tuple):
            activations["output"] = output[0].detach()
        else:
            activations["output"] = output.detach()

    # Register hook
    if modality == "vision":
        # Hook into vision encoder layer
        if hasattr(model, "vision_encoder"):
            target_layer = model.vision_encoder.vision_model.encoder.layers[layer_idx]
        elif hasattr(model, "model") and hasattr(model.model, "vision_tower"):
            target_layer = model.model.vision_tower.vision_model.encoder.layers[layer_idx]
        else:
            raise AttributeError("Cannot find vision encoder in model")
    else:  # text
        # Hook into text encoder layer (Mistral)
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            target_layer = model.model.layers[layer_idx]
        else:
            raise AttributeError("Cannot find text encoder layers in model")

    handle = target_layer.register_forward_hook(hook_fn)

    # Forward pass
    with torch.no_grad():
        if modality == "vision":
            _ = model.get_vision_features(inputs["pixel_values"])
        else:  # text
            _ = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

    # Remove hook
    handle.remove()

    if "output" not in activations:
        raise RuntimeError(f"Failed to extract activations from layer {layer_idx}")

    return activations["output"]


def mean_pool_embeddings(
    embeddings: torch.Tensor, attention_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Mean pool embeddings across sequence dimension.

    Args:
        embeddings: (batch_size, seq_len, hidden_dim)
        attention_mask: Optional mask (batch_size, seq_len)

    Returns:
        pooled: (batch_size, hidden_dim)
    """
    if attention_mask is not None:
        # Masked mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask
    else:
        # Simple mean pooling
        return embeddings.mean(dim=1)


def save_json(data: dict, filepath: str):
    """Save dict to JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved: {filepath}")


def load_json(filepath: str) -> dict:
    """Load JSON file."""
    with open(filepath) as f:
        return json.load(f)


def format_time(seconds: float) -> str:
    """Format seconds to human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def print_banner(text: str, char: str = "="):
    """Print a formatted banner."""
    width = 80
    print("\n" + char * width)
    print(text.center(width))
    print(char * width + "\n")


if __name__ == "__main__":
    # Test utilities
    print("Testing karpathy_utils.py...")
    print("✅ Utilities module loaded successfully")
