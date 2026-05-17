"""Shared model-loading for analysis scripts.

Consolidates the Stage-2 (hard routing) and Stage-3 (learned soft routing)
checkpoint-loading boilerplate that was copy-pasted across
``cross_modality_purity``, ``cross_concept_similarity_matrix`` and
``routing_ablation_experiment``.

Behaviour is preserved verbatim from those scripts with one deliberate fix:
for Stage 3 we set BOTH ``layer.mlp.temperature`` and
``layer.mlp._forward_temperature``. Previously ``cross_modality_purity`` only
set ``.temperature`` (which the MoE layer never reads — see CLAUDE.md: the
temperature must be supplied via ``_forward_temperature``), so its Stage-3
temperature was silently ignored. Setting both makes every caller correct.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from models.utils.common import register_moe_model


@dataclass
class LoadedModels:
    """Bundle of the four components every analysis script needs."""

    llm: object
    vision_encoder: object
    vision_connector: object
    tokenizer: object
    clip_processor: object


def _set_routing_mode(llm, mode: str, temperature: float | None = None) -> None:
    """Force every MoE layer into ``mode``; for soft routing set temperature.

    Sets both ``temperature`` and ``_forward_temperature`` — see module docstring.
    """
    for layer in llm.model.layers:
        if hasattr(layer.mlp, "routing_mode"):
            layer.mlp.routing_mode = mode
            if mode == "soft" and temperature is not None:
                layer.mlp.temperature = temperature
                layer.mlp._forward_temperature = temperature


def load_stage2_models(
    config: dict,
    device: str,
    stage2_checkpoint: str | None = None,
) -> LoadedModels:
    """Load CLIP + connector + Stage-2 MoE LLM with hard expert routing.

    If ``stage2_checkpoint`` is given it overrides the default
    ``<output_dir>/stage2_checkpoints/llm_stage2_best.pth`` and is loaded with
    ``map_location='cpu'`` (matching the original custom-checkpoint path);
    otherwise the default checkpoint is loaded with ``map_location=device``.
    """
    from transformers import (
        AutoModelForCausalLM,
        AutoProcessor,
        AutoTokenizer,
        CLIPVisionModel,
    )

    from models import VisionLanguageConnector

    register_moe_model()

    paths = config["paths"]
    output_dir = paths["output_dir"]

    print(f"  - Loading tokenizer from {paths['mistral_local_path']}")
    tokenizer = AutoTokenizer.from_pretrained(paths["mistral_local_path"])
    tokenizer.pad_token = tokenizer.eos_token

    print(f"  - Loading CLIP from {paths['clip_local_path']}")
    clip_processor = AutoProcessor.from_pretrained(paths["clip_local_path"])
    vision_encoder = CLIPVisionModel.from_pretrained(paths["clip_local_path"]).to(device)
    vision_encoder.eval()

    moe_model_path = paths["moe_model_path"]
    print(f"  - Loading base MoE model from {moe_model_path}")
    llm = AutoModelForCausalLM.from_pretrained(
        moe_model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # Required for output_attentions=True
    ).to(device)

    if stage2_checkpoint:
        print(f"  - Loading Stage 2 expert weights from {stage2_checkpoint}")
        expert_weights = torch.load(stage2_checkpoint, map_location="cpu")
    else:
        stage2_path = os.path.join(output_dir, "stage2_checkpoints", "llm_stage2_best.pth")
        print(f"  - Loading Stage 2 expert weights from {stage2_path}")
        expert_weights = torch.load(stage2_path, map_location=device)
    # The default llm_stage2_best.pth is a raw state_dict; some checkpoints wrap
    # it under "model_state_dict" (full training checkpoint) — handle both.
    if isinstance(expert_weights, dict) and "model_state_dict" in expert_weights:
        expert_weights = expert_weights["model_state_dict"]
    llm.load_state_dict(expert_weights, strict=False)
    llm.eval()

    _set_routing_mode(llm, "hard")

    print("  - Loading vision connector")
    vision_connector = VisionLanguageConnector().to(device)
    connector_path = os.path.join(output_dir, "vision_connector_stage1_best.pth")
    vision_connector.load_state_dict(torch.load(connector_path, map_location=device))
    vision_connector.eval()

    print("✅ All Stage 2 models loaded successfully")
    return LoadedModels(llm, vision_encoder, vision_connector, tokenizer, clip_processor)


def load_stage3_models(
    config: dict,
    device: str,
    stage3_checkpoint: str,
    temperature: float = 0.01,
) -> LoadedModels:
    """Load Stage-3 end-to-end model with learned soft routing.

    Loads the Stage-2 base first, then overrides with the Stage-3 checkpoint
    (full or portable format) and switches every MoE layer to soft routing.
    """
    models = load_stage2_models(config, device)

    print(f"  - Loading Stage 3 checkpoint from {stage3_checkpoint}")
    checkpoint = torch.load(stage3_checkpoint, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        print("      Detected FULL checkpoint format")
        models.llm.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(f"      ✓ Loaded LLM weights (epoch {checkpoint.get('epoch', 'unknown')})")
        if "connector_state_dict" in checkpoint:
            models.vision_connector.load_state_dict(checkpoint["connector_state_dict"])
            print("      ✓ Loaded vision connector weights (Stage 3 trained)")
    else:
        print("      Detected PORTABLE checkpoint format (state_dict only)")
        models.llm.load_state_dict(checkpoint, strict=False)
        print("      ✓ Loaded LLM weights (portable format)")
        print("      ⚠️  Note: Vision connector NOT updated (using Stage 1 weights)")

    _set_routing_mode(models.llm, "soft", temperature)

    models.llm.eval()
    models.vision_encoder.eval()
    models.vision_connector.eval()

    # Deterministic Gumbel noise for reproducible soft routing.
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print(f"✅ Stage 3 models loaded (soft routing, temperature={temperature})")
    return models
