import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.mistral.modeling_mistral import MistralMLP

class MoELayer(nn.Module):
    def __init__(self, config, d_model: int, num_experts: int = 2, routing_mode: str = 'hard'):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.routing_mode = routing_mode
        # The router is not needed for this hard-coded routing logic
        self.experts = nn.ModuleList([MistralMLP(config) for _ in range(num_experts)])

        # The gate is only used for 'soft' routing, but we initialize it
        # to ensure the model structure is consistent when loading weights.
        self.gate = nn.Linear(self.d_model, self.num_experts, bias=False)

        # Attribute to store the load balancing loss for 'soft' routing
        self.load_balancing_loss = 0.0

    def forward(self, hidden_states: torch.Tensor):
        """
        Main forward pass that dispatches to the correct routing logic.
        """
        if self.routing_mode == 'hard':
            return self._hard_routing_forward(hidden_states)
        elif self.routing_mode == 'soft':
            return self._soft_routing_forward(hidden_states)
        else:
            raise ValueError(f"Invalid routing mode: {self.routing_mode}. Must be 'hard' or 'soft'.")

    def _hard_routing_forward(self, hidden_states: torch.Tensor):
        """
        Original hard-coded routing logic from Stage 2.
        Relies on `self.routing_mask` being set externally.
        """
        routing_mask = self.routing_mask
        final_output = torch.zeros_like(hidden_states)

        vision_indices = torch.where(routing_mask == 0)
        text_indices = torch.where(routing_mask == 1)

        if vision_indices[0].numel() > 0:
            vision_tokens = hidden_states[vision_indices]
            vision_output = self.experts[0](vision_tokens)
            final_output[vision_indices] = vision_output

        if text_indices[0].numel() > 0:
            text_tokens = hidden_states[text_indices]
            text_output = self.experts[1](text_tokens)
            final_output[text_indices] = text_output

        return final_output

    def _soft_routing_forward(self, hidden_states: torch.Tensor):
        """
        Trainable soft routing logic for Stage 2.5.
        Uses the internal gating network.
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

        # Get routing logits from the gate.
        router_logits = self.gate(hidden_states_reshaped)
        
        # --- Load Balancing Loss ---
        # This loss encourages the gate to use all experts roughly equally.
        tokens_per_expert = F.one_hot(router_logits.argmax(dim=-1), num_classes=self.num_experts).float()
        router_load = tokens_per_expert.sum(dim=0)
        router_prob_per_expert = router_logits.softmax(dim=-1).sum(dim=0)
        
        # We store the loss as an attribute to be collected later in the training loop.
        self.load_balancing_loss = self.num_experts * torch.sum(router_load * router_prob_per_expert) / (hidden_states_reshaped.shape[0]**2)

        # --- Top-1 Gating ---
        routing_weights, selected_experts = torch.topk(F.softmax(router_logits, dim=1), 1, dim=-1)
        selected_experts = selected_experts.squeeze(-1)

        # --- Route tokens to their selected expert ---
        final_hidden_states = torch.zeros_like(hidden_states_reshaped)
        for expert_idx in range(self.num_experts):
            token_indices = torch.where(selected_experts == expert_idx)[0]
            
            if token_indices.shape[0] > 0:
                tokens_for_expert = hidden_states_reshaped[token_indices]
                weights_for_expert = routing_weights[token_indices]
                
                expert_output = self.experts[expert_idx](tokens_for_expert)
                weighted_output = expert_output * weights_for_expert
                weighted_output = weighted_output.to(final_hidden_states.dtype)

                
                final_hidden_states.index_add_(0, token_indices, weighted_output)
                
        return final_hidden_states.view(batch_size, sequence_length, hidden_dim)
