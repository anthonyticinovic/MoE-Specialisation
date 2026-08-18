"""Metrics computed on the representations the extraction mixin produces.

Cosine similarity and Euclidean distance between the two modalities' views of a
concept, the per-layer alignment curve, the purity matrix, the CLIP-versus-
connector comparison, and the token-variance and position-specific analyses.

A mixin, for the reason given in ``cross_modality_extraction``.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class PurityMetricsMixin:
    def analyze_representation(
        self, concept: str, expert: str, layer: int, modality: str, pooling: str = "cls"
    ) -> np.ndarray:
        """Extract hidden state representation for a concept at a specific layer.

        Args:
            pooling: "cls" (CLS token for vision) or "mean" (mean-pooled for vision)
        """
        if expert not in ["vision", "text"]:
            raise ValueError(f"Invalid expert: {expert}. Must be 'vision' or 'text'.")
        if modality not in ["vision", "text"]:
            raise ValueError(f"Invalid modality: {modality}. Must be 'vision' or 'text'.")
        if not (-1 <= layer < 32):
            raise ValueError(f"Invalid layer: {layer}. Must be in range [-1, 31].")

        expert_id = 0 if expert == "vision" else 1

        # Prepare input based on layer
        if layer == -1:
            # Layer -1: Pre-transformer embeddings (vision connector output vs text embeddings)
            if modality == "vision":
                embeddings = self._prepare_vision_input(concept)
            else:
                embeddings = self._prepare_text_input(concept)
            hidden_state = embeddings
        else:
            # Layers 0-31: Post-transformer hidden states with expert routing
            embeddings = (
                self._prepare_vision_input(concept)
                if modality == "vision"
                else self._prepare_text_input(concept)
            )
            batch_size, num_tokens = embeddings.shape[0], embeddings.shape[1]
            routing_mask = self._force_routing_through_expert(expert_id, batch_size, num_tokens)

            with torch.no_grad():
                hidden_state = self._extract_hidden_state_at_layer(embeddings, routing_mask, layer)

        # Extract representation based on modality and pooling
        # DEBUG: Only log first occurrence per concept+layer combination
        should_debug = (
            hasattr(self, "_debug_mode")
            and self._debug_mode
            and not hasattr(self, f"_logged_extract_{concept}_{layer}_{modality}_{pooling}")
        )

        if modality == "vision":
            if pooling == "cls":
                representation = hidden_state[:, 0, :].squeeze(0)
                if should_debug:
                    logger.info(f"      Vision CLS: pos 0 of {hidden_state.shape[1]} tokens")
            else:  # mean pooling
                representation = hidden_state.mean(dim=1).squeeze(0)
                if should_debug:
                    logger.info(f"      Vision mean: averaged {hidden_state.shape[1]} tokens")
        else:  # text modality
            seq_len = hidden_state.shape[1]
            # Always exclude BOS token (position 0)
            # For single concept tokens: use position 1 only
            # For multi-token concepts: average positions 1 to -1 (excluding BOS and potential EOS)
            if seq_len == 2:
                # Single concept token after BOS: just use position 1
                representation = hidden_state[:, 1, :].squeeze(0)
                if should_debug:
                    logger.info("      Text single: pos 1 (excluding BOS at pos 0)")
            elif seq_len > 2:
                # Multi-token concept: average all tokens from position 1 onwards (excluding only BOS)
                concept_tokens = hidden_state[:, 1:, :]
                representation = concept_tokens.mean(dim=1).squeeze(0)
                if should_debug:
                    logger.info(
                        f"      Text multi: pos [1:] of {seq_len} → {concept_tokens.shape[1]} tokens averaged (excluding BOS only)"
                    )
            else:
                # Edge case: only BOS token (shouldn't happen)
                representation = hidden_state[:, 0, :].squeeze(0)
                if should_debug:
                    logger.warning("       Text edge case: only BOS token")

        if should_debug:
            setattr(self, f"_logged_extract_{concept}_{layer}_{modality}_{pooling}", True)

        return representation.cpu().float().numpy()

    def analyze_vocab(
        self,
        concept: str,
        expert: str,
        layer: int,
        modality: str,
        top_k: int = 10,
        pooling: str = "cls",
    ) -> dict[str, float]:
        """Analyse top-k vocabulary predictions for a concept."""
        hidden_state = self.analyze_representation(concept, expert, layer, modality, pooling)
        hidden_state_tensor = (
            torch.from_numpy(hidden_state).to(self.device, dtype=torch.bfloat16).unsqueeze(0)
        )
        hidden_state_tensor = self.llm.model.norm(hidden_state_tensor)

        with torch.no_grad():
            logits = self.llm.lm_head(hidden_state_tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0)

        top_probs, top_indices = torch.topk(probs, top_k)
        return {
            self.tokenizer.decode([idx.item()]).strip(): prob.item()
            for prob, idx in zip(top_probs, top_indices, strict=True)
        }

    def compute_cosine_similarity(self, concept: str, layer: int, pooling: str = "cls") -> float:
        """Compute cosine similarity between vision and text expert representations."""
        # Only debug first concept and select layers
        should_debug = (
            hasattr(self, "_debug_mode")
            and self._debug_mode
            and layer in [-1, 0, 15, 31]
            and not hasattr(self, f"_logged_cosine_{concept}_{layer}_{pooling}")
        )

        if should_debug:
            logger.info(f"\n    Layer {layer} ({pooling}): '{concept}'")

        vision_rep = self.analyze_representation(concept, "vision", layer, "vision", pooling)
        text_rep = self.analyze_representation(concept, "text", layer, "text", pooling)

        cosine_sim = float(
            np.dot(vision_rep, text_rep)
            / (np.linalg.norm(vision_rep) * np.linalg.norm(text_rep) + 1e-8)
        )

        if should_debug:
            logger.info(
                f"      Vision: norm={np.linalg.norm(vision_rep):.2f}, mean={vision_rep.mean():.4f}"
            )
            logger.info(
                f"      Text:   norm={np.linalg.norm(text_rep):.2f}, mean={text_rep.mean():.4f}"
            )
            logger.info(f"       Cosine similarity: {cosine_sim:.4f}")
            setattr(self, f"_logged_cosine_{concept}_{layer}_{pooling}", True)

        return cosine_sim

    def compute_euclidean_distance(self, concept: str, layer: int, pooling: str = "cls") -> float:
        """Compute Euclidean distance between vision and text expert representations."""
        vision_rep = self.analyze_representation(concept, "vision", layer, "vision", pooling)
        text_rep = self.analyze_representation(concept, "text", layer, "text", pooling)
        return float(np.linalg.norm(vision_rep - text_rep))

    def compute_alignment_curve(
        self, image_path: str, text: str, pooling: str = "mean", routing_mode: str = "natural"
    ) -> dict[int, float]:
        """Compute cosine similarity at each layer for a concept pair.

        This method performs a single forward pass per modality and extracts
        hidden states at all 33 layers (embedding + 32 transformer layers).

        Args:
            image_path: Path to image file
            text: Text concept (e.g., "cat")
            pooling: "mean" or "cls" pooling strategy
            routing_mode: "natural" (learned routing) or "forced" (vision→expert_0, text→expert_1)

        Returns:
            Dict mapping layer_id (-1 to 31) to cosine similarity
        """
        # Prepare inputs
        vision_embeddings = self._prepare_vision_input(image_path)
        text_embeddings = self._prepare_text_input(text)

        # Extract all layer states in one forward pass per modality
        if routing_mode == "forced":
            # Stage 2 style: force vision→expert_0, text→expert_1
            vision_routing = self._force_routing_through_expert(0, 1, vision_embeddings.shape[1])
            text_routing = self._force_routing_through_expert(1, 1, text_embeddings.shape[1])
        else:
            # Stage 3 style: let model route naturally
            vision_routing = None
            text_routing = None

        vision_states = self._extract_all_layer_states(vision_embeddings, vision_routing)
        text_states = self._extract_all_layer_states(text_embeddings, text_routing)

        # Compute similarity at each layer
        similarities = {}
        for layer_idx, (vis_state, txt_state) in enumerate(
            zip(vision_states, text_states, strict=True)
        ):
            # Pool representations
            vis_rep = self._pool_representation(vis_state, pooling, modality="vision")
            txt_rep = self._pool_representation(txt_state, pooling, modality="text")

            # Cosine similarity
            cos_sim = np.dot(vis_rep, txt_rep) / (
                np.linalg.norm(vis_rep) * np.linalg.norm(txt_rep) + 1e-8
            )

            # Map layer_idx to actual layer number: 0→-1, 1→0, 2→1, ..., 32→31
            layer_number = layer_idx - 1
            similarities[layer_number] = float(cos_sim)

        return similarities

    def compute_purity_matrix(
        self, concepts: list[str], layer: int, pooling: str = "mean"
    ) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix for all concept-modality combinations.

        Args:
            concepts: List of exactly 2 concepts to compare
            layer: Layer index to analyse
            pooling: Pooling strategy ("mean" for mean-pooled representations)

        Returns:
            NxN matrix where N = 2 * len(concepts), organised as:
            [concept1_vis, concept1_txt, concept2_vis, concept2_txt, ...]
        """
        if len(concepts) != 2:
            raise ValueError(f"Purity matrix requires exactly 2 concepts, got {len(concepts)}")

        # Extract all representations: [concept1_vis, concept1_txt, concept2_vis, concept2_txt]
        representations = []
        labels = []

        for concept in concepts:
            # Vision representation through vision expert
            vis_rep = self.analyze_representation(concept, "vision", layer, "vision", pooling)
            representations.append(vis_rep)
            labels.append(f"{concept}_vis")

            # Text representation through text expert
            txt_rep = self.analyze_representation(concept, "text", layer, "text", pooling)
            representations.append(txt_rep)
            labels.append(f"{concept}_txt")

        # Compute pairwise cosine similarities
        n = len(representations)
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    cos_sim = np.dot(representations[i], representations[j]) / (
                        np.linalg.norm(representations[i]) * np.linalg.norm(representations[j])
                        + 1e-8
                    )
                    matrix[i, j] = cos_sim

        return matrix, labels

    def _compute_clip_connector_comparison_generic(
        self, concepts: list[str], pooling: str = "mean"
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Generic helper to compare CLIP vs connector embeddings with different pooling strategies.

        Args:
            concepts: List of exactly 2 concepts to compare
            pooling: "mean" for mean-pooling or "cls" for CLS token only

        Returns:
            Tuple of (clip_matrix, connector_matrix, labels)
        """
        if len(concepts) != 2:
            raise ValueError(
                f"CLIP vs connector comparison requires exactly 2 concepts, got {len(concepts)}"
            )

        pooling_desc = "mean-pooled" if pooling == "mean" else "CLS token"
        logger.info(f"\nComparing CLIP vs connector ({pooling_desc}) for: {concepts}")

        # Helper function to load image (file path or synthetic)
        def load_image(concept):
            if os.path.isfile(concept):
                image = Image.open(concept).convert("RGB")
                label = os.path.splitext(os.path.basename(concept))[0]
                logger.info(f"      Loaded '{label}' from {concept} (size: {image.size})")
                return image, label
            else:
                image = self.image_generator.generate_concept_image(concept)
                return image, concept

        # Helper to extract embedding based on pooling strategy
        def extract_embedding(hidden_state):
            if pooling == "cls":
                # Extract CLS token (position 0)
                return hidden_state[:, 0, :].squeeze(0).cpu().float().numpy()
            else:  # mean pooling
                # Mean-pool across all tokens
                return hidden_state.mean(dim=1).squeeze(0).cpu().float().numpy()

        # Extract embeddings for both concepts
        clip_embeddings = []
        connector_embeddings = []
        labels = []

        for concept in concepts:
            image, label = load_image(concept)
            labels.append(label)

            # Process through CLIP (auto-resizes to 224×224)
            pixel_values = self.clip_processor(images=image, return_tensors="pt").pixel_values.to(
                self.device
            )

            with torch.no_grad():
                # Get CLIP output
                clip_output = self.vision_encoder(pixel_values).last_hidden_state
                clip_embedding = extract_embedding(clip_output)
                clip_embeddings.append(clip_embedding)

                # Get connector output
                connector_output = self.vision_connector(clip_output)
                connector_embedding = extract_embedding(connector_output)
                connector_embeddings.append(connector_embedding)

            logger.info(
                f"  CLIP ({pooling_desc}) for '{label}': shape={clip_embedding.shape}, norm={np.linalg.norm(clip_embedding):.2f}"
            )
            logger.info(
                f"  Connector ({pooling_desc}) for '{label}': shape={connector_embedding.shape}, norm={np.linalg.norm(connector_embedding):.2f}"
            )

        # Compute 2×2 similarity matrices
        def compute_similarity_matrix(embeddings):
            n = len(embeddings)
            matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i == j:
                        matrix[i, j] = 1.0
                    else:
                        cos_sim = np.dot(embeddings[i], embeddings[j]) / (
                            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8
                        )
                        matrix[i, j] = cos_sim
            return matrix

        clip_matrix = compute_similarity_matrix(clip_embeddings)
        connector_matrix = compute_similarity_matrix(connector_embeddings)

        logger.info(
            f"\n  CLIP ({pooling_desc}): {labels[0]} vs {labels[1]} = {clip_matrix[0, 1]:.4f}"
        )
        logger.info(
            f"  Connector ({pooling_desc}): {labels[0]} vs {labels[1]} = {connector_matrix[0, 1]:.4f}"
        )

        return clip_matrix, connector_matrix, labels

    def compute_clip_connector_comparison(
        self, concepts: list[str]
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Compare raw CLIP embeddings vs post-connector embeddings (mean-pooled).

        This diagnostic analysis helps determine if CLIP already fails to distinguish concepts,
        or if the vision connector is crushing good CLIP features into a narrow subspace.

        Args:
            concepts: List of exactly 2 concepts to compare

        Returns:
            Tuple of (clip_matrix, connector_matrix, labels)
        """
        return self._compute_clip_connector_comparison_generic(concepts, pooling="mean")

    def compute_clip_connector_comparison_cls(
        self, concepts: list[str]
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Compare raw CLIP embeddings vs post-connector embeddings (CLS token only).

        This tests whether mean-pooling is washing out discriminative information,
        or if the problem exists at the global (CLS) representation level.

        Args:
            concepts: List of exactly 2 concepts to compare

        Returns:
            Tuple of (clip_matrix, connector_matrix, labels)
        """
        return self._compute_clip_connector_comparison_generic(concepts, pooling="cls")

    def analyze_token_variance(self, concepts: list[str]) -> dict:
        """
        Level 1: Analyse internal token diversity within each image.

        This measures whether the connector is collapsing spatial structure by comparing
        the variance of pairwise similarities between tokens within a single image.

        Args:
            concepts: List of exactly 2 concepts to compare

        Returns:
            Dict with variance statistics for CLIP and connector, per concept
        """
        if len(concepts) != 2:
            raise ValueError(
                f"Token variance analysis requires exactly 2 concepts, got {len(concepts)}"
            )

        logger.info(f"\nLevel 1: Analysing token-level variance for {concepts}")

        def load_image(concept):
            if os.path.isfile(concept):
                image = Image.open(concept).convert("RGB")
                label = os.path.splitext(os.path.basename(concept))[0]
                return image, label
            else:
                image = self.image_generator.generate_concept_image(concept)
                return image, concept

        def compute_internal_variance(tokens):
            """Compute variance of pairwise cosine similarities within token sequence."""
            tokens_np = tokens.cpu().float().numpy()  # [257, dim]
            n = tokens_np.shape[0]

            # Compute all pairwise similarities
            similarities = []
            for i in range(n):
                for j in range(i + 1, n):
                    cos_sim = np.dot(tokens_np[i], tokens_np[j]) / (
                        np.linalg.norm(tokens_np[i]) * np.linalg.norm(tokens_np[j]) + 1e-8
                    )
                    similarities.append(cos_sim)

            similarities = np.array(similarities)
            return {
                "mean": float(similarities.mean()),
                "std": float(similarities.std()),
                "min": float(similarities.min()),
                "max": float(similarities.max()),
            }

        results = {}

        for concept in concepts:
            image, label = load_image(concept)

            pixel_values = self.clip_processor(images=image, return_tensors="pt").pixel_values.to(
                self.device
            )

            with torch.no_grad():
                clip_output = self.vision_encoder(pixel_values).last_hidden_state
                clip_tokens = clip_output.squeeze(0)  # [257, 1024]

                connector_output = self.vision_connector(clip_output)
                connector_tokens = connector_output.squeeze(0)  # [257, 4096]

            clip_variance = compute_internal_variance(clip_tokens)
            connector_variance = compute_internal_variance(connector_tokens)

            results[label] = {"clip": clip_variance, "connector": connector_variance}

            logger.info(f"  {label}:")
            logger.info(
                f"      CLIP variance: std={clip_variance['std']:.4f}, range=[{clip_variance['min']:.3f}, {clip_variance['max']:.3f}]"
            )
            logger.info(
                f"      Connector variance: std={connector_variance['std']:.4f}, range=[{connector_variance['min']:.3f}, {connector_variance['max']:.3f}]"
            )

        return results

    def analyze_position_specific_similarity(self, concepts: list[str]) -> dict:
        """
        Level 2: Analyse cat-car similarity at each of the 257 token positions.

        This reveals whether certain positions (e.g., CLS token) maintain better
        concept separation than others.

        Args:
            concepts: List of exactly 2 concepts to compare

        Returns:
            Dict with per-position similarities for CLIP and connector
        """
        if len(concepts) != 2:
            raise ValueError(
                f"Position-specific analysis requires exactly 2 concepts, got {len(concepts)}"
            )

        logger.info(f"\nLevel 2: Analysing position-specific similarity for {concepts}")

        def load_image(concept):
            if os.path.isfile(concept):
                image = Image.open(concept).convert("RGB")
                label = os.path.splitext(os.path.basename(concept))[0]
                return image, label
            else:
                image = self.image_generator.generate_concept_image(concept)
                return image, concept

        # Load both images
        images = []
        labels = []
        for concept in concepts:
            image, label = load_image(concept)
            images.append(image)
            labels.append(label)

        # Process both images
        concept1_clip_tokens = None
        concept1_connector_tokens = None
        concept2_clip_tokens = None
        concept2_connector_tokens = None

        for idx, (image, label) in enumerate(zip(images, labels, strict=True)):
            pixel_values = self.clip_processor(images=image, return_tensors="pt").pixel_values.to(
                self.device
            )

            with torch.no_grad():
                clip_output = self.vision_encoder(pixel_values).last_hidden_state
                clip_tokens = clip_output.squeeze(0).cpu().float().numpy()  # [257, 1024]

                connector_output = self.vision_connector(clip_output)
                connector_tokens = connector_output.squeeze(0).cpu().float().numpy()  # [257, 4096]

            if idx == 0:
                concept1_clip_tokens = clip_tokens
                concept1_connector_tokens = connector_tokens
            else:
                concept2_clip_tokens = clip_tokens
                concept2_connector_tokens = connector_tokens

        # Compute similarity at each position
        clip_similarities = []
        connector_similarities = []

        for pos in range(257):
            # CLIP similarity at this position
            clip_sim = np.dot(concept1_clip_tokens[pos], concept2_clip_tokens[pos]) / (
                np.linalg.norm(concept1_clip_tokens[pos])
                * np.linalg.norm(concept2_clip_tokens[pos])
                + 1e-8
            )
            clip_similarities.append(float(clip_sim))

            # Connector similarity at this position
            conn_sim = np.dot(concept1_connector_tokens[pos], concept2_connector_tokens[pos]) / (
                np.linalg.norm(concept1_connector_tokens[pos])
                * np.linalg.norm(concept2_connector_tokens[pos])
                + 1e-8
            )
            connector_similarities.append(float(conn_sim))

        logger.info(
            f"  Position 0 (CLS): CLIP={clip_similarities[0]:.4f}, Connector={connector_similarities[0]:.4f}"
        )
        logger.info(
            f"  Positions 1-256 (patches): CLIP_mean={np.mean(clip_similarities[1:]):.4f}, Connector_mean={np.mean(connector_similarities[1:]):.4f}"
        )

        return {
            "labels": labels,
            "clip_similarities": clip_similarities,
            "connector_similarities": connector_similarities,
        }
