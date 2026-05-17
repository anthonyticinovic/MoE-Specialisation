"""Mistral-7B subclassed into a two-expert MoE causal LM.

These are thin subclasses of the HuggingFace Mistral classes that swap the
standard FFN for :class:`~models.moe_layer.MoELayer` in every decoder layer.

Because the architecture is custom, any script that loads a saved checkpoint
must register it with the Auto* factories *before* calling ``from_pretrained``::

    from transformers import AutoConfig, AutoModelForCausalLM

    AutoConfig.register("mistral_moe", MistralMoEConfig)
    AutoModelForCausalLM.register(MistralMoEConfig, MistralMoEForCausalLM)

``models.utils.common.register_moe_model()`` performs exactly this registration.
"""

import torch.nn as nn
from transformers import MistralConfig, MistralForCausalLM
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

from .moe_layer import MoELayer


class MistralMoEConfig(MistralConfig):
    """Mistral config tagged with a distinct ``model_type`` for Auto* registration."""

    model_type = "mistral_moe"


class MistralMoEDecoderLayer(MistralDecoderLayer):
    """A Mistral decoder layer whose FFN is replaced by a two-expert MoE layer."""

    def __init__(self, config: MistralMoEConfig, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        self.mlp = MoELayer(config=config, d_model=config.hidden_size)


class MistralMoEForCausalLM(MistralForCausalLM):
    """Mistral causal LM with every decoder layer's FFN swapped for an MoE layer."""

    config_class = MistralMoEConfig

    def __init__(self, config: MistralMoEConfig) -> None:
        super().__init__(config)
        self.model.layers = nn.ModuleList(
            [MistralMoEDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
