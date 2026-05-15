from transformers import MistralForCausalLM

from models import MoELayer


def replace_ffn_with_moe(model: MistralForCausalLM) -> MistralForCausalLM:
    """Surgically replaces the FFN in each Mistral layer with our MoELayer."""
    print(" surgically replacing FFNs with MoE layers...")
    for i, layer in enumerate(model.model.layers):
        original_ffn = layer.mlp
        d_model = original_ffn.gate_proj.in_features
        moe_layer = MoELayer(model.config, d_model=d_model, num_experts=2)

        # Initialize both experts with the original FFN's weights
        moe_layer.experts[0].load_state_dict(original_ffn.state_dict())
        moe_layer.experts[1].load_state_dict(original_ffn.state_dict())

        layer.mlp = moe_layer
    print("✅ All FFN layers have been replaced with MoE layers.")
    return model
