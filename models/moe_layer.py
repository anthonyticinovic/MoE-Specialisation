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
        nn.init.normal_(self.gate.weight, std=0.02)

        # Attribute to store the load balancing loss for 'soft' routing
        self.load_balancing_loss = 0.0

    def initialize_gate(self):
        """
        Explicitly initializes or re-initializes the gate weights.
        This ensures the gate has the correct shape and a fresh start.
        """
        self.gate = nn.Linear(self.d_model, self.num_experts, bias=False)
        nn.init.normal_(self.gate.weight, std=0.02)

    def forward(self, hidden_states: torch.Tensor, temperature: float = 1.0):
        """
        Main forward pass that dispatches to the correct routing logic.
        """
        if self.routing_mode == 'hard':
            return self._hard_routing_forward(hidden_states)
        elif self.routing_mode == 'soft':
            # Pass the temperature argument to the soft routing function
            return self._soft_routing_forward(hidden_states, temperature)
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

    def _soft_routing_forward(self, hidden_states: torch.Tensor, temperature: float = 1.0):
        """
        Differentiable and sparse soft routing using Straight-Through Gumbel-Softmax.
    
        In the forward pass, this behaves like hard routing (only one expert is
        computed per token). In the backward pass, gradients flow back to all
        router logits as if it were a soft, probabilistic choice. This provides
        a rich learning signal while maintaining computational efficiency.
        """
        # Get initial logits from the gate
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states_reshaped)

        # 1. Gumbel-Softmax for stochastic, differentiable sampling
        # Gumbel noise is added for exploration during training
        gumbels = -torch.empty_like(router_logits).exponential_().log()
        y = (router_logits + gumbels) / temperature
        # router_probs contains the "soft" probabilities used for the backward pass
        router_probs = F.softmax(y, dim=-1)

        # 2. Straight-Through Estimator Trick
        # Get a "hard" one-hot vector for the forward pass
        hard_idx = router_probs.argmax(dim=-1, keepdim=True)
        hard_onehot = torch.zeros_like(router_probs).scatter_(1, hard_idx, 1.0)
        # This line is the core of the trick:
        # Forward pass uses `hard_onehot`, backward pass uses `router_probs`
        router_onehot = hard_onehot - router_probs.detach() + router_probs

        # Initialize a final output tensor of zeros
        final_hidden_states = torch.zeros_like(hidden_states_reshaped)

        # 3. Sparse Dispatch: Compute only the selected expert for each token
        for expert_idx, expert in enumerate(self.experts):
            # Find the indices of all tokens that should be routed to this expert
            token_indices = torch.where(router_onehot[:, expert_idx] == 1)[0]
            
            # If no tokens are routed to this expert, skip it
            if token_indices.numel() > 0:
                # Select the hidden states for these specific tokens
                tokens_for_expert = hidden_states_reshaped[token_indices]
                
                # Compute the expert's output ONLY for its assigned tokens
                expert_output = expert(tokens_for_expert)
                
                # Get the weights from the straight-through estimator for these tokens
                weights_for_expert = router_onehot[token_indices, expert_idx].unsqueeze(-1)
                
                # Scale the output by the weights (this carries the gradient)
                weighted_output = expert_output * weights_for_expert
                
                # Add the weighted output back to the final tensor at the correct positions
                final_hidden_states.index_add_(0, token_indices, weighted_output.to(final_hidden_states.dtype))

        # 4. Differentiable Load Balancing Loss
        tokens_per_expert = torch.mean(router_onehot, dim=0) # Fraction of tokens to each expert
        avg_prob_per_expert = torch.mean(router_probs, dim=0) # Average router probability
        # The loss is scaled by the number of experts
        self.load_balancing_loss = self.num_experts * torch.sum(tokens_per_expert * avg_prob_per_expert)

        # Reshape to the original dimensions and return
        return final_hidden_states.view(batch_size, sequence_length, hidden_dim)
