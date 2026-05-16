"""
Create the custom MoE model from a base Mistral-7B checkpoint.

Replaces every FFN layer with a two-expert MoELayer. Both experts are
initialised with identical copies of the original FFN weights so the model
starts from a known-good, pretrained state.

Run from the repo root:
    python -m models.utils.create_moe_model \
        --base-model YOUR_PATH_HERE/Mistral-7B-v0.3 \
        --output    YOUR_PATH_HERE/Mistral-7B-MoE
"""

import argparse
import json
import logging
import os
import shutil

from transformers import MistralForCausalLM

from models.custom_mistral import MistralMoEForCausalLM

logger = logging.getLogger(__name__)


def create_moe_model(base_model_path: str, output_path: str):
    logger.info("Loading base model from %s...", base_model_path)
    llm_base = MistralForCausalLM.from_pretrained(base_model_path)

    logger.info("Creating MistralMoEForCausalLM...")
    llm_moe = MistralMoEForCausalLM(llm_base.config)

    logger.info("Copying weights...")
    llm_moe.load_state_dict(llm_base.state_dict(), strict=False)
    for layer_base, layer_moe in zip(llm_base.model.layers, llm_moe.model.layers, strict=False):
        layer_moe.mlp.experts[0].load_state_dict(layer_base.mlp.state_dict())
        layer_moe.mlp.experts[1].load_state_dict(layer_base.mlp.state_dict())
    logger.info("Weights copied.")

    llm_moe.config.model_type = "mistral_moe"
    llm_moe.config.architectures = ["MistralMoEForCausalLM"]

    logger.info("Saving MoE model to %s...", output_path)
    llm_moe.save_pretrained(output_path)

    # Patch config.json so AutoModel can locate the custom classes
    config_path = os.path.join(output_path, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)
    config_dict["auto_map"] = {
        "AutoConfig": "custom_mistral.MistralMoEConfig",
        "AutoModelForCausalLM": "custom_mistral.MistralMoEForCausalLM",
    }
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    # Copy custom model files into the saved directory so trust_remote_code works
    for src in ["models/custom_mistral.py", "models/moe_layer.py", "models/__init__.py"]:
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_path, os.path.basename(src)))
            logger.info("Copied %s", src)

    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True, help="Path to Mistral-7B-v0.3")
    parser.add_argument("--output", required=True, help="Output path for MoE model")
    args = parser.parse_args()
    create_moe_model(args.base_model, args.output)
