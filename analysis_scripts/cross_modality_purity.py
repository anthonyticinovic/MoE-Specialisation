"""
Cross-Modality Purity Analysis for MoE Vision-Language Models

This script analyzes how "pure" expert representations are across modalities
by comparing vision and text expert activations for the same concept.

Usage:
    python analysis_scripts/cross_modality_purity.py --concepts red blue --layers 0 8 16 24 31
    python analysis_scripts/cross_modality_purity.py --concepts circle --top-k 20
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from analysis_scripts import cross_modality_purity_plots as cmp_plots
from analysis_scripts._lib import (
    SyntheticImageGenerator,
    load_stage2_models,
    load_stage3_models,
    load_training_config,
)


class CrossModalityPurityAnalyzer:
    """
    Analyzes cross-modality purity of expert representations.

    Key Methods:
        - analyze_vocab(): Top-k vocabulary predictions
        - analyze_representation(): Hidden state extraction
        - compute_cosine_similarity(): Representation similarity
        - compute_probability_ratio(): Cross-modal concept probability
    """

    def __init__(
        self,
        config_path: str = "configs/training_config.yaml",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.config = self._load_config(config_path)
        self.image_generator = SyntheticImageGenerator()

        # Model components (loaded later)
        self.llm = None
        self.vision_encoder = None
        self.vision_connector = None
        self.tokenizer = None
        self.clip_processor = None

        # Cache for storing intermediate results
        self.hidden_states_cache = {}

        print(f"🔧 Initialized analyzer on device: {self.device}")

    def _load_config(self, config_path: str) -> dict:
        """Load training configuration (validated, placeholder-checked)."""
        return load_training_config(config_path)

    def _assign(self, models) -> None:
        """Copy a _lib.LoadedModels bundle onto this analyzer's attributes."""
        self.llm = models.llm
        self.vision_encoder = models.vision_encoder
        self.vision_connector = models.vision_connector
        self.tokenizer = models.tokenizer
        self.clip_processor = models.clip_processor

    def load_models(self):
        """Load CLIP + connector + Stage-2 MoE LLM (hard routing)."""
        print("📦 Loading models...")
        self._assign(load_stage2_models(self.config, self.device))
        print("✅ All models loaded successfully")

    def load_stage3_models(self, checkpoint_path: str, temperature: float = 0.01):
        """Load Stage 3 models with learned soft routing.

        Args:
            checkpoint_path: Path to Stage 3 checkpoint (full or portable version)
            temperature: Softmax temperature for routing (default: 0.01 for near-deterministic)
        """
        print("📦 Loading Stage 3 models with learned routing...")
        self._assign(load_stage3_models(self.config, self.device, checkpoint_path, temperature))

    def _prepare_vision_input(self, concept: str) -> torch.Tensor:
        """Generate synthetic image or load from file path and convert to visual tokens via vision connector.

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
            print(f"      📸 Loaded '{concept_name}' from {concept} (original size: {image.size})")
        else:
            # Generate synthetic image
            image = self.image_generator.generate_concept_image(concept)

        # Process through CLIP (automatically resizes to 224×224 and normalizes)
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
            print(f"      💬 Extracted text concept '{concept_name}' from image path")
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
            print(f"      💬 Text tokenization: '{text}' → {tokens} ({len(tokens)} tokens)")
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
                    print(f"      🎯 Vision CLS: pos 0 of {hidden_state.shape[1]} tokens")
            else:  # mean pooling
                representation = hidden_state.mean(dim=1).squeeze(0)
                if should_debug:
                    print(f"      🎯 Vision mean: averaged {hidden_state.shape[1]} tokens")
        else:  # text modality
            seq_len = hidden_state.shape[1]
            # Always exclude BOS token (position 0)
            # For single concept tokens: use position 1 only
            # For multi-token concepts: average positions 1 to -1 (excluding BOS and potential EOS)
            if seq_len == 2:
                # Single concept token after BOS: just use position 1
                representation = hidden_state[:, 1, :].squeeze(0)
                if should_debug:
                    print("      🎯 Text single: pos 1 (excluding BOS at pos 0)")
            elif seq_len > 2:
                # Multi-token concept: average all tokens from position 1 onwards (excluding only BOS)
                concept_tokens = hidden_state[:, 1:, :]
                representation = concept_tokens.mean(dim=1).squeeze(0)
                if should_debug:
                    print(
                        f"      🎯 Text multi: pos [1:] of {seq_len} → {concept_tokens.shape[1]} tokens averaged (excluding BOS only)"
                    )
            else:
                # Edge case: only BOS token (shouldn't happen)
                representation = hidden_state[:, 0, :].squeeze(0)
                if should_debug:
                    print("      ⚠️  Text edge case: only BOS token")

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
        """Analyze top-k vocabulary predictions for a concept."""
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
            for prob, idx in zip(top_probs, top_indices)
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
            print(f"\n    🔍 Layer {layer} ({pooling}): '{concept}'")

        vision_rep = self.analyze_representation(concept, "vision", layer, "vision", pooling)
        text_rep = self.analyze_representation(concept, "text", layer, "text", pooling)

        cosine_sim = float(
            np.dot(vision_rep, text_rep)
            / (np.linalg.norm(vision_rep) * np.linalg.norm(text_rep) + 1e-8)
        )

        if should_debug:
            print(
                f"      📊 Vision: norm={np.linalg.norm(vision_rep):.2f}, mean={vision_rep.mean():.4f}"
            )
            print(
                f"      📊 Text:   norm={np.linalg.norm(text_rep):.2f}, mean={text_rep.mean():.4f}"
            )
            print(f"      ➡️  Cosine similarity: {cosine_sim:.4f}")
            setattr(self, f"_logged_cosine_{concept}_{layer}_{pooling}", True)

        return cosine_sim

    def compute_euclidean_distance(self, concept: str, layer: int, pooling: str = "cls") -> float:
        """Compute Euclidean distance between vision and text expert representations."""
        vision_rep = self.analyze_representation(concept, "vision", layer, "vision", pooling)
        text_rep = self.analyze_representation(concept, "text", layer, "text", pooling)
        return float(np.linalg.norm(vision_rep - text_rep))

    def extract_concept_samples(
        self, annotations_file: str, concepts: list[str], samples_per_concept: int, seed: int = 42
    ) -> dict[str, list[dict]]:
        """
        Extract balanced samples from COCO annotations for specified concepts.

        Args:
            annotations_file: Path to COCO annotations JSON
            concepts: List of concept keywords (e.g., ["cat", "dog", "car"])
            samples_per_concept: Target number of samples per concept
            seed: Random seed for reproducibility

        Returns:
            Dict mapping concept -> list of sample dicts with keys:
                - 'image_id': COCO image ID
                - 'caption': Image caption
                - 'image_path': Relative path to image file
        """
        print("📚 Extracting concept samples from COCO annotations...")
        print(f"   Concepts: {concepts}")
        print(f"   Target: {samples_per_concept} samples per concept")

        # Load COCO annotations
        with open(annotations_file) as f:
            coco_data = json.load(f)

        # Build image_id -> image_path mapping
        image_id_to_path = {}
        for img in coco_data["images"]:
            image_id_to_path[img["id"]] = img["file_name"]

        # Set random seed
        np.random.seed(seed)

        # Extract samples for each concept
        concept_samples = {concept: [] for concept in concepts}

        for annotation in coco_data["annotations"]:
            caption = annotation["caption"].lower()
            image_id = annotation["image_id"]

            # Check which concepts appear in this caption
            matching_concepts = [c for c in concepts if c.lower() in caption.split()]

            # Skip if multiple specified concepts appear (ambiguous)
            if len(matching_concepts) > 1:
                continue

            # Skip if no concepts match
            if len(matching_concepts) == 0:
                continue

            # Add to the matching concept's sample list
            concept = matching_concepts[0]
            if len(concept_samples[concept]) < samples_per_concept:
                concept_samples[concept].append(
                    {
                        "image_id": image_id,
                        "caption": annotation["caption"],
                        "image_path": image_id_to_path[image_id],
                        "concept": concept,
                    }
                )

        # Print statistics
        print("\n   📊 Extracted samples:")
        for concept, samples in concept_samples.items():
            print(f"      {concept}: {len(samples)} samples")

        # Warn if any concept is under-sampled
        for concept, samples in concept_samples.items():
            if len(samples) < samples_per_concept:
                print(
                    f"   ⚠️  Warning: Only found {len(samples)} samples for '{concept}' (target: {samples_per_concept})"
                )

        return concept_samples

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
        for layer_idx, (vis_state, txt_state) in enumerate(zip(vision_states, text_states)):
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
            layer: Layer index to analyze
            pooling: Pooling strategy ("mean" for mean-pooled representations)

        Returns:
            NxN matrix where N = 2 * len(concepts), organized as:
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
        print(f"\n🔬 Comparing CLIP vs connector ({pooling_desc}) for: {concepts}")

        # Helper function to load image (file path or synthetic)
        def load_image(concept):
            if os.path.isfile(concept):
                image = Image.open(concept).convert("RGB")
                label = os.path.splitext(os.path.basename(concept))[0]
                print(f"      📸 Loaded '{label}' from {concept} (size: {image.size})")
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

            print(
                f"  ✓ CLIP ({pooling_desc}) for '{label}': shape={clip_embedding.shape}, norm={np.linalg.norm(clip_embedding):.2f}"
            )
            print(
                f"  ✓ Connector ({pooling_desc}) for '{label}': shape={connector_embedding.shape}, norm={np.linalg.norm(connector_embedding):.2f}"
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

        print(f"\n  📊 CLIP ({pooling_desc}): {labels[0]} vs {labels[1]} = {clip_matrix[0, 1]:.4f}")
        print(
            f"  📊 Connector ({pooling_desc}): {labels[0]} vs {labels[1]} = {connector_matrix[0, 1]:.4f}"
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
        Level 1: Analyze internal token diversity within each image.

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

        print(f"\n🔬 Level 1: Analyzing token-level variance for {concepts}")

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

            print(f"  ✓ {label}:")
            print(
                f"      CLIP variance: std={clip_variance['std']:.4f}, range=[{clip_variance['min']:.3f}, {clip_variance['max']:.3f}]"
            )
            print(
                f"      Connector variance: std={connector_variance['std']:.4f}, range=[{connector_variance['min']:.3f}, {connector_variance['max']:.3f}]"
            )

        return results

    def analyze_position_specific_similarity(self, concepts: list[str]) -> dict:
        """
        Level 2: Analyze cat-car similarity at each of the 257 token positions.

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

        print(f"\n🔬 Level 2: Analyzing position-specific similarity for {concepts}")

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

        for idx, (image, label) in enumerate(zip(images, labels)):
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

        print(
            f"  ✓ Position 0 (CLS): CLIP={clip_similarities[0]:.4f}, Connector={connector_similarities[0]:.4f}"
        )
        print(
            f"  ✓ Positions 1-256 (patches): CLIP_mean={np.mean(clip_similarities[1:]):.4f}, Connector_mean={np.mean(connector_similarities[1:]):.4f}"
        )

        return {
            "labels": labels,
            "clip_similarities": clip_similarities,
            "connector_similarities": connector_similarities,
        }

    def run_comprehensive_analysis(
        self,
        concepts: list[str],
        layers: list[int],
        output_dir: str = "results/cross_modality_purity",
    ) -> dict:
        """
        Run comprehensive cross-modality purity analysis.

        Args:
            concepts: List of concepts to analyze
            layers: List of layer indices
            output_dir: Directory to save results

        Returns:
            Dictionary containing all analysis results
        """
        print(
            f"\n🔬 Running comprehensive analysis on {len(concepts)} concepts across {len(layers)} layers..."
        )

        os.makedirs(output_dir, exist_ok=True)

        # Setup debug logging to file if in debug mode
        debug_log_path = None
        if hasattr(self, "_debug_mode") and self._debug_mode:
            debug_log_path = os.path.join(output_dir, "debug_token_analysis.log")
            print(f"🐛 Debug output will be saved to: {debug_log_path}")

        results = {
            "concepts": concepts,
            "layers": layers,
            "cosine_similarity": {},
            "euclidean_distance": {},
            "cosine_similarity_mean_pooled": {},
            "euclidean_distance_mean_pooled": {},
            "vocab_predictions": {},
        }

        for concept_idx, concept in enumerate(concepts):
            print(f"\n📊 Analyzing concept: '{concept}'")

            # Add debug summary for first concept only
            if hasattr(self, "_debug_mode") and self._debug_mode and concept_idx == 0:
                print("   🐛 Debug info will be shown for layers: -1, 0, 15, 31")

            results["cosine_similarity"][concept] = {}
            results["euclidean_distance"][concept] = {}
            results["cosine_similarity_mean_pooled"][concept] = {}
            results["euclidean_distance_mean_pooled"][concept] = {}
            results["vocab_predictions"][concept] = {}

            for layer in layers:
                print(f"  - Layer {layer}...", end=" ")

                try:
                    # Cosine similarity (CLS and mean-pooled)
                    cosine_sim = self.compute_cosine_similarity(concept, layer, pooling="cls")
                    cosine_sim_mp = self.compute_cosine_similarity(concept, layer, pooling="mean")
                    results["cosine_similarity"][concept][f"layer_{layer}"] = cosine_sim
                    results["cosine_similarity_mean_pooled"][concept][f"layer_{layer}"] = (
                        cosine_sim_mp
                    )

                    # Euclidean distance (CLS and mean-pooled)
                    euclidean_dist = self.compute_euclidean_distance(concept, layer, pooling="cls")
                    euclidean_dist_mp = self.compute_euclidean_distance(
                        concept, layer, pooling="mean"
                    )
                    results["euclidean_distance"][concept][f"layer_{layer}"] = euclidean_dist
                    results["euclidean_distance_mean_pooled"][concept][f"layer_{layer}"] = (
                        euclidean_dist_mp
                    )

                    # Vocab predictions (CLS, mean-pooled, and text)
                    results["vocab_predictions"][concept][f"layer_{layer}"] = {
                        "vision_expert_cls": self.analyze_vocab(
                            concept, "vision", layer, "vision", top_k=10, pooling="cls"
                        ),
                        "vision_expert_mean_pooled": self.analyze_vocab(
                            concept, "vision", layer, "vision", top_k=10, pooling="mean"
                        ),
                        "text_expert": self.analyze_vocab(concept, "text", layer, "text", top_k=10),
                    }

                    print(
                        f"✓ (cos_cls={cosine_sim:.3f}, cos_mp={cosine_sim_mp:.3f}, euc_cls={euclidean_dist:.2f}, euc_mp={euclidean_dist_mp:.2f})"
                    )

                except Exception as e:
                    print(f"✗ Error: {e}")
                    continue

        # Save results
        results_path = os.path.join(output_dir, "purity_analysis_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {results_path}")

        # Generate visualizations
        self._visualize_results(results, output_dir)

        return results

    def run_stage3_alignment_analysis(
        self, config_path: str = "configs/stage3_alignment.json"
    ) -> dict:
        """Run comprehensive Stage 3 layer-by-layer alignment analysis using COCO sampling.

        Args:
            config_path: Path to Stage 3 alignment config file

        Returns:
            Dictionary containing alignment curves and metadata
        """
        print("\n" + "=" * 80)
        print("Stage 3: Layer-by-Layer Cross-Modal Alignment Analysis")
        print("=" * 80)

        # Load config
        print(f"\n📋 Loading config from {config_path}")
        with open(config_path) as f:
            config = json.load(f)["stage3_alignment_analysis"]

        checkpoint_path = config["checkpoint_path"]
        temperature = config["temperature"]
        concepts = config["concepts"]
        samples_per_concept = config["samples_per_concept"]
        annotations_file = config["annotations_file"]
        image_dir = config["image_dir"]
        pooling = config["pooling"]
        routing_mode = config["routing_mode"]
        output_dir = config["output_dir"]
        seed = config.get("seed", 42)

        os.makedirs(output_dir, exist_ok=True)

        print(f"  ✓ Checkpoint: {checkpoint_path}")
        print(f"  ✓ Temperature: {temperature}")
        print(f"  ✓ Pooling: {pooling}")
        print(f"  ✓ Routing: {routing_mode}")
        print(f"  ✓ Concepts: {concepts}")
        print(f"  ✓ Samples per concept: {samples_per_concept}")

        # Load Stage 3 models
        self.load_stage3_models(checkpoint_path, temperature)

        # Extract concept samples from COCO
        concept_samples = self.extract_concept_samples(
            annotations_file=annotations_file,
            concepts=concepts,
            samples_per_concept=samples_per_concept,
            seed=seed,
        )

        # Compute alignment curves for each concept
        print(
            f"\n🔬 Computing alignment curves (averaging {samples_per_concept} samples per concept)..."
        )
        alignment_curves = {}

        for idx, concept in enumerate(concepts, 1):
            samples = concept_samples[concept]
            if len(samples) == 0:
                print(f"  [{idx}/{len(concepts)}] ⚠️  Skipping '{concept}' (no samples found)")
                continue

            print(
                f"  [{idx}/{len(concepts)}] Processing '{concept}' ({len(samples)} samples)...",
                end=" ",
            )

            try:
                # Compute alignment curve for each sample and average
                sample_curves = []

                for sample in samples:
                    image_path = os.path.join(image_dir, sample["image_path"])
                    text = sample["caption"]

                    curve = self.compute_alignment_curve(
                        image_path=image_path, text=text, pooling=pooling, routing_mode=routing_mode
                    )
                    sample_curves.append(curve)

                # Average curves across all samples for this concept
                # All curves should have same keys (layer indices)
                avg_curve = {}
                all_layers = sample_curves[0].keys()
                for layer in all_layers:
                    avg_curve[layer] = np.mean([curve[layer] for curve in sample_curves])

                alignment_curves[concept] = avg_curve

                # Print key layer similarities
                emb_sim = avg_curve[-1]
                final_sim = avg_curve[31]
                print(f"✓ (emb={emb_sim:.3f}, L0={avg_curve[0]:.3f}, L31={final_sim:.3f})")

            except Exception as e:
                print(f"✗ Error: {e}")
                import traceback

                traceback.print_exc()
                continue

        # Save results
        results = {
            "config": config,
            "alignment_curves": alignment_curves,
            "metadata": {
                "checkpoint": checkpoint_path,
                "temperature": temperature,
                "pooling": pooling,
                "routing_mode": routing_mode,
                "num_concepts": len(alignment_curves),
                "samples_per_concept": samples_per_concept,
            },
        }

        results_path = os.path.join(output_dir, "alignment_curves.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {results_path}")

        # Generate visualization
        print("\n📈 Generating alignment curve plot...")
        cmp_plots.plot_alignment_curves(alignment_curves, output_dir, title_suffix="Stage 3")

        print("\n" + "=" * 80)
        print("✅ Stage 3 alignment analysis complete!")
        print(f"📁 Results saved to: {output_dir}")
        print("=" * 80)

        return results

    def _visualize_results(self, results: dict, output_dir: str):
        """Generate visualization plots from analysis results."""
        print("\n📈 Generating visualizations...")

        concepts = results["concepts"]
        layers = results["layers"]

        # Plot all four metrics
        cmp_plots.plot_metric(
            results,
            concepts,
            layers,
            "cosine_similarity",
            output_dir,
            "Cosine Similarity",
            "Cross-Modality Representation Similarity",
            "cosine_similarity_lineplot.png",
            ylim=(-1, 1),
        )

        cmp_plots.plot_metric(
            results,
            concepts,
            layers,
            "euclidean_distance",
            output_dir,
            "Euclidean Distance (L2 Norm)",
            "Cross-Modality Representation Distance",
            "euclidean_distance_lineplot.png",
        )

        cmp_plots.plot_metric(
            results,
            concepts,
            layers,
            "cosine_similarity_mean_pooled",
            output_dir,
            "Cosine Similarity",
            "Cross-Modality Representation Similarity (Mean-Pooled Vision)",
            "cosine_similarity_meanpooled_lineplot.png",
            ylim=(-1, 1),
        )

        cmp_plots.plot_metric(
            results,
            concepts,
            layers,
            "euclidean_distance_mean_pooled",
            output_dir,
            "Euclidean Distance (L2 Norm)",
            "Cross-Modality Representation Distance (Mean-Pooled Vision)",
            "euclidean_distance_meanpooled_lineplot.png",
        )

        # Generate purity matrix and divergence tracking if exactly 2 concepts
        if len(concepts) == 2:
            print("\n📊 Generating purity matrix and divergence analysis (2 concepts detected)...")

            # CLIP vs Connector comparison (diagnostic analysis)
            print("\n🔬 Running CLIP vs Connector diagnostic...")

            # Mean-pooled comparison
            try:
                clip_matrix, connector_matrix, labels = self.compute_clip_connector_comparison(
                    concepts
                )
                cmp_plots.plot_clip_connector_comparison(
                    clip_matrix, connector_matrix, labels, output_dir, pooling="mean"
                )
            except Exception as e:
                print(f"  ✗ Error computing mean-pooled CLIP vs connector comparison: {e}")

            # CLS token comparison
            try:
                clip_matrix_cls, connector_matrix_cls, labels_cls = (
                    self.compute_clip_connector_comparison_cls(concepts)
                )
                cmp_plots.plot_clip_connector_comparison(
                    clip_matrix_cls, connector_matrix_cls, labels_cls, output_dir, pooling="cls"
                )
            except Exception as e:
                print(f"  ✗ Error computing CLS token CLIP vs connector comparison: {e}")

            # Level 1: Token variance analysis
            print("\n🔬 Level 1: Analyzing token-level variance...")
            try:
                variance_results = self.analyze_token_variance(concepts)
                cmp_plots.plot_token_variance(variance_results, labels_cls, output_dir)
            except Exception as e:
                print(f"  ✗ Error in token variance analysis: {e}")

            # Level 2: Position-specific similarity
            print("\n🔬 Level 2: Analyzing position-specific similarity...")
            try:
                position_results = self.analyze_position_specific_similarity(concepts)
                cmp_plots.plot_position_specific_similarity(position_results, output_dir)
            except Exception as e:
                print(f"  ✗ Error in position-specific analysis: {e}")

            # Purity matrices at key layers
            target_layers = [-1, 0, 15, 31]
            matrices = {}
            for layer in target_layers:
                if layer in layers:
                    try:
                        matrix, labels = self.compute_purity_matrix(concepts, layer, pooling="mean")
                        matrices[layer] = (matrix, labels)
                    except Exception as e:
                        print(f"  ✗ Error computing purity matrix for layer {layer}: {e}")

            if matrices:
                cmp_plots.plot_purity_matrices(matrices, target_layers, output_dir)

        print("✅ Visualization complete!")


def main():
    """Main entry point for cross-modality purity analysis."""
    parser = argparse.ArgumentParser(description="Cross-Modality Purity Analysis for MoE VLM")
    parser.add_argument(
        "--concepts",
        nargs="+",
        default=["red", "blue"],
        help="Concepts to analyze (e.g., red blue circle square)",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[-1, 0, 8, 16, 24, 31],
        help="Layer indices to analyze (-1=pre-transformer embeddings, 0-31=transformer layers)",
    )
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="Analyze all layers from -1 to 31 (overrides --layers)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/cross_modality_purity",
        help="Directory to save results",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training config file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda/cpu)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed debug output for token extraction and representation analysis",
    )
    parser.add_argument(
        "--stage3-alignment",
        type=str,
        default=None,
        metavar="CONFIG_PATH",
        help="Run Stage 3 layer-by-layer alignment analysis (provide config path, e.g., configs/stage3_alignment.json)",
    )

    args = parser.parse_args()

    # Initialize analyzer
    print("=" * 80)

    # Check if running Stage 3 alignment analysis
    if args.stage3_alignment:
        print("Stage 3: Layer-by-Layer Alignment Analysis")
        print("=" * 80)

        analyzer = CrossModalityPurityAnalyzer(config_path=args.config, device=args.device)
        results = analyzer.run_stage3_alignment_analysis(config_path=args.stage3_alignment)

        return  # Exit after Stage 3 analysis

    # Otherwise run standard Stage 2 purity analysis
    print("Cross-Modality Purity Analysis")
    if args.debug:
        print("🐛 DEBUG MODE ENABLED")
    print("=" * 80)

    # Handle --all-layers flag
    if args.all_layers:
        layers = [-1] + list(range(32))
        print(
            f"📊 Using --all-layers: analyzing layers {layers[0]} to {layers[-1]} ({len(layers)} total layers)"
        )
    else:
        layers = args.layers

    analyzer = CrossModalityPurityAnalyzer(config_path=args.config, device=args.device)

    # Enable debug mode if requested
    if args.debug:
        analyzer._debug_mode = True
        print("🐛 Debug mode: Will show tokenization + detailed stats for layers [-1, 0, 15, 31]")

    # Load models
    analyzer.load_models()

    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis(
        concepts=args.concepts, layers=layers, output_dir=args.output_dir
    )

    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print(f"📁 Results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
