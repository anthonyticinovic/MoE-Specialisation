"""Getting hidden states out of the model, for the purity analyses.

The model-plumbing half of ``CrossModalityPurityAnalyzer``: build the
``inputs_embeds`` for one modality, force a token through a chosen expert, pull
the hidden state at a layer, and pool it. Separated from the metrics computed on
top of those states so neither file runs past the 800-line guideline.

A mixin rather than free functions because every method reads the loaded models
off ``self``, and two analysers in other modules subclass the composed class —
mixing in keeps their method resolution byte-for-byte identical.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class RepresentationExtractionMixin:
    def _prepare_vision_input(self, concept: str) -> torch.Tensor:
        """Generate or load an image and project it to visual tokens.

        Args:
            concept: Concept to generate image for, or file path to image

        Returns:
            Visual tokens after CLIP and vision connector (pre-transformer)
        """
        # Check if concept is a file path
        if os.path.isfile(concept):
            # Load image from disk (any size, any format - CLIP processor will resize)
            image = Image.open(concept).convert("RGB")
            concept_name = os.path.splitext(os.path.basename(concept))[0]
            logger.info(
                f"      Loaded '{concept_name}' from {concept} (original size: {image.size})"
            )
        else:
            # Generate synthetic image
            image = self.image_generator.generate_concept_image(concept)

        # Process through CLIP (automatically resizes to 224×224 and normalises)
        pixel_values = self.clip_processor(images=image, return_tensors="pt").pixel_values.to(
            self.device
        )

        with torch.no_grad():
            patch_embeddings = self.vision_encoder(pixel_values).last_hidden_state
            # Use learned vision connector
            visual_tokens = self.vision_connector(patch_embeddings)
            # Convert to bfloat16 to match model dtype
            visual_tokens = visual_tokens.to(torch.bfloat16)

        return visual_tokens

    def _prepare_text_input(self, concept: str) -> torch.Tensor:
        """Tokenize concept and convert to text embeddings.

        Args:
            concept: Either a concept name (e.g., "cat") or file path (e.g., "data/images/cat.jpg")
                    If file path, extracts concept name from filename
        """
        # If concept is a file path, extract the concept name from filename
        if os.path.isfile(concept):
            concept_name = os.path.splitext(os.path.basename(concept))[0]
            logger.info(f"      Extracted text concept '{concept_name}' from image path")
            text = f"{concept_name}"
        else:
            text = f"{concept}"

        input_ids = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).input_ids.to(self.device)

        # DEBUG: Log tokenization details (only for first occurrence per concept)
        debug_key = concept if not os.path.isfile(concept) else concept_name
        if (
            hasattr(self, "_debug_mode")
            and self._debug_mode
            and not hasattr(self, f"_logged_text_{debug_key}")
        ):
            tokens = [self.tokenizer.decode([tid]) for tid in input_ids[0]]
            logger.info(f"      Text tokenization: '{text}' → {tokens} ({len(tokens)} tokens)")
            setattr(self, f"_logged_text_{debug_key}", True)

        with torch.no_grad():
            text_embeddings = self.llm.model.embed_tokens(input_ids)
            # Ensure bfloat16 dtype
            text_embeddings = text_embeddings.to(torch.bfloat16)

        return text_embeddings

    def _force_routing_through_expert(
        self, expert_id: int, batch_size: int, num_tokens: int
    ) -> torch.Tensor:
        """
        Create routing mask to force all tokens through a specific expert.

        Args:
            expert_id: 0 for vision expert, 1 for text expert
            batch_size: Batch size
            num_tokens: Number of tokens per batch

        Returns:
            routing_mask tensor of shape [batch_size, num_tokens]
        """
        routing_mask = torch.full(
            (batch_size, num_tokens), expert_id, dtype=torch.long, device=self.device
        )
        return routing_mask

    def _extract_hidden_state_at_layer(
        self, embeddings: torch.Tensor, routing_mask: torch.Tensor, target_layer: int
    ) -> torch.Tensor:
        """
        Run forward pass and extract hidden state at specified layer.

        Args:
            embeddings: Input embeddings [batch_size, seq_len, hidden_dim]
            routing_mask: Routing decisions for each token
            target_layer: Layer index to extract from (0-31)

        Returns:
            Hidden state tensor at target layer
        """
        # Set routing mask for all layers
        for layer in self.llm.model.layers:
            layer.mlp.routing_mask = routing_mask

        # Use model's forward pass with output_hidden_states to get all layer outputs
        # This handles attention masks, position embeddings, and cache correctly
        outputs = self.llm.model(
            inputs_embeds=embeddings, output_hidden_states=True, return_dict=True
        )

        # outputs.hidden_states is a tuple: (embedding_output, layer_0_output, ..., layer_31_output)
        # So target_layer 0 is at index 1, target_layer 31 is at index 32
        hidden_state = outputs.hidden_states[target_layer + 1]

        return hidden_state

    def _extract_all_layer_states(
        self, embeddings: torch.Tensor, routing_mask: torch.Tensor | None = None
    ) -> list[torch.Tensor]:
        """Extract hidden states at ALL layers in one forward pass.

        Args:
            embeddings: Input embeddings [batch_size, seq_len, hidden_dim]
            routing_mask: Optional routing decisions for forced routing mode

        Returns:
            List of 33 hidden state tensors: [embedding_output, layer_0, ..., layer_31]
        """
        # Set routing mask if forced mode (Stage 2 style)
        if routing_mask is not None:
            for layer in self.llm.model.layers:
                layer.mlp.routing_mask = routing_mask

        # Single forward pass capturing all layer outputs
        with torch.no_grad():
            outputs = self.llm.model(
                inputs_embeds=embeddings, output_hidden_states=True, return_dict=True
            )

        # outputs.hidden_states is a tuple: (embedding_output, layer_0_output, ..., layer_31_output)
        return list(outputs.hidden_states)

    def _pool_representation(
        self, hidden_state: torch.Tensor, pooling: str, modality: str
    ) -> np.ndarray:
        """Pool hidden state to single representation vector.

        Args:
            hidden_state: Hidden state tensor [batch_size, seq_len, hidden_dim]
            pooling: "cls" or "mean"
            modality: "vision" or "text"

        Returns:
            Pooled representation as numpy array
        """
        if modality == "vision":
            if pooling == "cls":
                # CLS token (position 0)
                representation = hidden_state[:, 0, :].squeeze(0)
            else:  # mean pooling
                # Average all 257 tokens
                representation = hidden_state.mean(dim=1).squeeze(0)
        else:  # text modality
            seq_len = hidden_state.shape[1]
            # Always exclude BOS token (position 0)
            if seq_len == 2:
                # Single concept token: use position 1 only
                representation = hidden_state[:, 1, :].squeeze(0)
            elif seq_len > 2:
                # Multi-token concept: average positions 1 onwards (excluding BOS)
                concept_tokens = hidden_state[:, 1:, :]
                representation = concept_tokens.mean(dim=1).squeeze(0)
            else:
                # Edge case: only BOS token
                representation = hidden_state[:, 0, :].squeeze(0)

        return representation.cpu().float().numpy()
