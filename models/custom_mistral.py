import torch.nn as nn

from transformers import MistralForCausalLM, MistralConfig
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from .moe_layer import MoELayer


class MistralMoEConfig(MistralConfig):
    model_type = "mistral_moe"


class MistralMoEDecoderLayer(MistralDecoderLayer):
    def __init__(self, config: MistralMoEConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.mlp = MoELayer(config=config, d_model=config.hidden_size)


class MistralMoEForCausalLM(MistralForCausalLM):
    config_class = MistralMoEConfig

    def __init__(self, config):
        super().__init__(config)
        self.model.layers = nn.ModuleList(
            [MistralMoEDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
