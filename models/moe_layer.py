import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.mistral.modeling_mistral import MistralMLP


class MoELayer(nn.Module):
    """
    Two-expert MoE layer replacing the standard FFN in each Mistral decoder layer.

    Supports two routing modes:
    - 'hard': deterministic routing via an externally-set `routing_mask` (0=vision, 1=text).
      Used in Stage 2 where the mask is derived from token position (visual vs. text).
    - 'soft': differentiable routing via a learned gate with Gumbel-Softmax +
      Straight-Through Estimator. Used in Stages 2.5 and 3.
    """

    def __init__(self, config, d_model: int, num_experts: int = 2, routing_mode: str = 'hard'):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.routing_mode = routing_mode
        self.experts = nn.ModuleList([MistralMLP(config) for _ in range(num_experts)])

        # Gate is only active in 'soft' mode but initialised in both to keep the
        # model structure consistent when loading checkpoints across stages.
        self.gate = nn.Linear(self.d_model, self.num_experts, bias=False)
        nn.init.normal_(self.gate.weight, std=0.05)

        self.router_dropout = nn.Dropout(0.1)

    def initialize_gate(self):
        """Re-initialise gate weights (called in Stage 2.5 to break symmetry)."""
        self.gate = nn.Linear(self.d_model, self.num_experts, bias=False)
        nn.init.normal_(self.gate.weight, std=0.05)

    def forward(self, hidden_states: torch.Tensor, temperature: float = 1.0):
        if hasattr(self, '_forward_temperature'):
            temperature = self._forward_temperature

        if self.routing_mode == 'hard':
            return self._hard_routing_forward(hidden_states)
        elif self.routing_mode == 'soft':
            return self._soft_routing_forward(hidden_states, temperature)
        else:
            raise ValueError(f"Invalid routing mode: {self.routing_mode}. Must be 'hard' or 'soft'.")

    def _hard_routing_forward(self, hidden_states: torch.Tensor):
        """
        Route tokens based on an externally-set `self.routing_mask`.
        Mask values: 0 → Expert 0 (vision), 1 → Expert 1 (text).
        """
        routing_mask = self.routing_mask
        final_output = torch.zeros_like(hidden_states)

        vision_indices = torch.where(routing_mask == 0)
        text_indices = torch.where(routing_mask == 1)

        if vision_indices[0].numel() > 0:
            final_output[vision_indices] = self.experts[0](hidden_states[vision_indices])

        if text_indices[0].numel() > 0:
            final_output[text_indices] = self.experts[1](hidden_states[text_indices])

        return final_output

    def _soft_routing_forward(self, hidden_states: torch.Tensor, temperature: float = 1.0):
        """
        Differentiable routing via Straight-Through Gumbel-Softmax.

        Forward pass: hard (one expert per token, computed sparsely).
        Backward pass: gradients flow through soft probabilities to the gate.

        FSDP note: every expert must be called on every rank to keep collective
        operations in sync. When no tokens route to an expert, a zero-weighted
        dummy forward pass is performed to satisfy this requirement.
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_flat)
        router_logits = self.router_dropout(router_logits)

        # Store for metric collection during validation
        self._last_router_logits = router_logits.view(batch_size, sequence_length, self.num_experts)

        if self.training:
            gumbels = -torch.empty_like(router_logits).exponential_().log()
            y = (router_logits + gumbels) / temperature
        else:
            y = router_logits / temperature

        router_probs = F.softmax(y, dim=-1)

        # Straight-Through Estimator: hard selection in forward, soft probs in backward
        hard_idx = router_probs.argmax(dim=-1, keepdim=True)
        hard_onehot = torch.zeros_like(router_probs).scatter_(1, hard_idx, 1.0)
        router_onehot = hard_onehot - router_probs.detach() + router_probs

        final_hidden_states = torch.zeros_like(hidden_flat)

        for expert_idx, expert in enumerate(self.experts):
            token_indices = torch.where(router_onehot[:, expert_idx] == 1)[0]

            if token_indices.numel() > 0:
                tokens_for_expert = hidden_flat[token_indices]
                expert_output = expert(tokens_for_expert)

                if hasattr(self, 'expert_dropout') and self.training:
                    expert_output = self.expert_dropout(expert_output)

                weights = router_onehot[token_indices, expert_idx].unsqueeze(-1)
                final_hidden_states.index_add_(
                    0, token_indices, (expert_output * weights).to(final_hidden_states.dtype)
                )
            else:
                # FSDP: call expert with a dummy input so all ranks trigger the same
                # all-gather collective. The zero weight ensures no contribution to output.
                dummy_out = expert(hidden_flat[:1])
                final_hidden_states[:1] += dummy_out.to(final_hidden_states.dtype) * 0.0

        return final_hidden_states.view(batch_size, sequence_length, hidden_dim)
