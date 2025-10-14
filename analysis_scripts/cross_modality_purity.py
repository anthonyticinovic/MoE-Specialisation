"""
Cross-Modality Purity Analysis for MoE Vision-Language Models

This script analyzes how "pure" expert representations are across modalities
by comparing vision and text expert activations for the same concept.

Usage:
    python analysis_scripts/cross_modality_purity.py --concepts red blue --layers 0 8 16 24 31
    python analysis_scripts/cross_modality_purity.py --concepts circle --top-k 20
"""

import torch
import torch.nn as nn
import numpy as np
import yaml
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoConfig,
    AutoModelForCausalLM,
    CLIPVisionModel,
)

from models.custom_mistral import MistralMoEConfig, MistralMoEForCausalLM
from models import VisionLanguageConnector

# Register custom architecture
AutoConfig.register("mistral_moe", MistralMoEConfig)
AutoModelForCausalLM.register(MistralMoEConfig, MistralMoEForCausalLM)


class SyntheticImageGenerator:
    """Generates simple synthetic images for concept testing."""

    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.colors = {
            "red": (255, 0, 0), "blue": (0, 0, 255), "green": (0, 255, 0),
            "yellow": (255, 255, 0), "orange": (255, 165, 0), "purple": (128, 0, 128),
            "black": (0, 0, 0), "white": (255, 255, 255),
        }

    def generate_concept_image(self, concept: str) -> Image.Image:
        """Generate image from concept: 'red', 'circle', or 'red circle'."""
        parts = concept.lower().split()
        
        # Pure color patch
        if len(parts) == 1 and parts[0] in self.colors:
            return Image.new("RGB", (self.image_size, self.image_size), self.colors[parts[0]])
        
        # Shape (with optional color)
        if len(parts) == 2:
            # Colored shape: "red circle"
            color_name = parts[0]
            shape_name = parts[1]
            if color_name not in self.colors or shape_name not in ["circle", "square", "triangle"]:
                raise ValueError(f"Unknown concept: {concept}")
            fill_color = self.colors[color_name]
            outline_color = self.colors[color_name]
        elif len(parts) == 1 and parts[0] in ["circle", "square", "triangle"]:
            # Pure shape: black outline only, white fill (no color contamination)
            shape_name = parts[0]
            fill_color = (255, 255, 255)  # white fill
            outline_color = (0, 0, 0)  # black outline
        else:
            raise ValueError(f"Unknown concept: {concept}")
        
        # Create white background and draw shape
        image = Image.new("RGB", (self.image_size, self.image_size), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        margin = self.image_size // 4
        line_width = 3  # Make outline visible
        
        if shape_name == "circle":
            draw.ellipse([margin, margin, self.image_size - margin, self.image_size - margin], 
                        fill=fill_color, outline=outline_color, width=line_width)
        elif shape_name == "square":
            draw.rectangle([margin, margin, self.image_size - margin, self.image_size - margin], 
                          fill=fill_color, outline=outline_color, width=line_width)
        elif shape_name == "triangle":
            center_x = self.image_size // 2
            draw.polygon([(center_x, margin), (margin, self.image_size - margin), 
                         (self.image_size - margin, self.image_size - margin)], 
                        fill=fill_color, outline=outline_color)
        
        return image


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

    def _load_config(self, config_path: str) -> Dict:
        """Load training configuration."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def load_models(self):
        """Load all required models and weights."""
        print("📦 Loading models...")

        paths = self.config["paths"]
        output_dir = paths["output_dir"]

        # Load tokenizer
        print(f"  - Loading tokenizer from {paths['mistral_local_path']}")
        self.tokenizer = AutoTokenizer.from_pretrained(paths["mistral_local_path"])
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load CLIP processor and vision encoder
        print(f"  - Loading CLIP from {paths['clip_local_path']}")
        self.clip_processor = AutoProcessor.from_pretrained(paths["clip_local_path"])
        self.vision_encoder = CLIPVisionModel.from_pretrained(
            paths["clip_local_path"]
        ).to(self.device)
        self.vision_encoder.eval()

        # Load base MoE model
        moe_model_path = (
            "/data/gpfs/projects/COMP90055/aticinovic/models/Mistral-7B-MoE"
        )
        print(f"  - Loading base MoE model from {moe_model_path}")
        self.llm = AutoModelForCausalLM.from_pretrained(
            moe_model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        ).to(self.device)

        # Load Stage 2 expert weights
        stage2_path = os.path.join(
            output_dir, "stage2_checkpoints", "llm_stage2_best.pth"
        )
        print(f"  - Loading Stage 2 expert weights from {stage2_path}")
        expert_weights = torch.load(stage2_path, map_location=self.device)
        self.llm.load_state_dict(expert_weights, strict=False)

        # Load Stage 2.5 router weights
        stage2_5_path = os.path.join(
            output_dir, "stage2_5_checkpoints", "llm_stage2_5_best.pth"
        )
        print(f"  - Loading Stage 2.5 router weights from {stage2_5_path}")
        router_weights = torch.load(stage2_5_path, map_location=self.device)
        self.llm.load_state_dict(router_weights, strict=False)

        self.llm.eval()

        # Force hard routing mode
        for layer in self.llm.model.layers:
            if hasattr(layer.mlp, "routing_mode"):
                layer.mlp.routing_mode = "hard"

        # Load vision connector
        print("  - Loading vision connector")
        self.vision_connector = VisionLanguageConnector().to(self.device)
        connector_path = os.path.join(
            output_dir, "vision_connector_stage1_best.pth"
        )
        self.vision_connector.load_state_dict(
            torch.load(connector_path, map_location=self.device)
        )
        self.vision_connector.eval()

        print("✅ All models loaded successfully")

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
        pixel_values = self.clip_processor(
            images=image, return_tensors="pt"
        ).pixel_values.to(self.device)

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
        if hasattr(self, '_debug_mode') and self._debug_mode and not hasattr(self, f'_logged_text_{debug_key}'):
            tokens = [self.tokenizer.decode([tid]) for tid in input_ids[0]]
            print(f"      💬 Text tokenization: '{text}' → {tokens} ({len(tokens)} tokens)")
            setattr(self, f'_logged_text_{debug_key}', True)

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
            embeddings = self._prepare_vision_input(concept) if modality == "vision" else self._prepare_text_input(concept)
            batch_size, num_tokens = embeddings.shape[0], embeddings.shape[1]
            routing_mask = self._force_routing_through_expert(expert_id, batch_size, num_tokens)
            
            with torch.no_grad():
                hidden_state = self._extract_hidden_state_at_layer(embeddings, routing_mask, layer)

        # Extract representation based on modality and pooling
        # DEBUG: Only log first occurrence per concept+layer combination
        should_debug = (hasattr(self, '_debug_mode') and self._debug_mode and 
                       not hasattr(self, f'_logged_extract_{concept}_{layer}_{modality}_{pooling}'))
        
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
                    print(f"      🎯 Text single: pos 1 (excluding BOS at pos 0)")
            elif seq_len > 2:
                # Multi-token concept: average all tokens from position 1 onwards (excluding only BOS)
                concept_tokens = hidden_state[:, 1:, :]
                representation = concept_tokens.mean(dim=1).squeeze(0)
                if should_debug:
                    print(f"      🎯 Text multi: pos [1:] of {seq_len} → {concept_tokens.shape[1]} tokens averaged (excluding BOS only)")
            else:
                # Edge case: only BOS token (shouldn't happen)
                representation = hidden_state[:, 0, :].squeeze(0)
                if should_debug:
                    print(f"      ⚠️  Text edge case: only BOS token")
        
        if should_debug:
            setattr(self, f'_logged_extract_{concept}_{layer}_{modality}_{pooling}', True)

        return representation.cpu().float().numpy()

    def analyze_vocab(
        self, concept: str, expert: str, layer: int, modality: str, top_k: int = 10, pooling: str = "cls"
    ) -> Dict[str, float]:
        """Analyze top-k vocabulary predictions for a concept."""
        hidden_state = self.analyze_representation(concept, expert, layer, modality, pooling)
        hidden_state_tensor = torch.from_numpy(hidden_state).to(self.device, dtype=torch.bfloat16).unsqueeze(0)
        hidden_state_tensor = self.llm.model.norm(hidden_state_tensor)

        with torch.no_grad():
            logits = self.llm.lm_head(hidden_state_tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0)

        top_probs, top_indices = torch.topk(probs, top_k)
        return {self.tokenizer.decode([idx.item()]).strip(): prob.item() 
                for prob, idx in zip(top_probs, top_indices)}

    def compute_cosine_similarity(self, concept: str, layer: int, pooling: str = "cls") -> float:
        """Compute cosine similarity between vision and text expert representations."""
        # Only debug first concept and select layers
        should_debug = (hasattr(self, '_debug_mode') and self._debug_mode and 
                       layer in [-1, 0, 15, 31] and 
                       not hasattr(self, f'_logged_cosine_{concept}_{layer}_{pooling}'))
        
        if should_debug:
            print(f"\n    🔍 Layer {layer} ({pooling}): '{concept}'")
        
        vision_rep = self.analyze_representation(concept, "vision", layer, "vision", pooling)
        text_rep = self.analyze_representation(concept, "text", layer, "text", pooling)
        
        cosine_sim = float(np.dot(vision_rep, text_rep) / (np.linalg.norm(vision_rep) * np.linalg.norm(text_rep) + 1e-8))
        
        if should_debug:
            print(f"      📊 Vision: norm={np.linalg.norm(vision_rep):.2f}, mean={vision_rep.mean():.4f}")
            print(f"      📊 Text:   norm={np.linalg.norm(text_rep):.2f}, mean={text_rep.mean():.4f}")
            print(f"      ➡️  Cosine similarity: {cosine_sim:.4f}")
            setattr(self, f'_logged_cosine_{concept}_{layer}_{pooling}', True)
        
        return cosine_sim

    def compute_euclidean_distance(self, concept: str, layer: int, pooling: str = "cls") -> float:
        """Compute Euclidean distance between vision and text expert representations."""
        vision_rep = self.analyze_representation(concept, "vision", layer, "vision", pooling)
        text_rep = self.analyze_representation(concept, "text", layer, "text", pooling)
        return float(np.linalg.norm(vision_rep - text_rep))

    def compute_purity_matrix(self, concepts: List[str], layer: int, pooling: str = "mean") -> np.ndarray:
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
                        np.linalg.norm(representations[i]) * np.linalg.norm(representations[j]) + 1e-8
                    )
                    matrix[i, j] = cos_sim
        
        return matrix, labels

    def compute_purity_separation_scores(self, concepts: List[str], layers: List[int], pooling: str = "mean") -> Dict:
        """
        Compute purity and separation scores across layers.
        
        Purity Score: How well vision and text representations align for the SAME concept
        Separation Score: How well the model distinguishes DIFFERENT concepts
        
        Args:
            concepts: List of exactly 2 concepts
            layers: Layer indices to analyze
            pooling: Pooling strategy
            
        Returns:
            Dict with 'purity_scores' and 'separation_scores' lists
        """
        if len(concepts) != 2:
            raise ValueError(f"Purity/separation tracking requires exactly 2 concepts, got {len(concepts)}")
        
        concept1, concept2 = concepts
        purity_scores = []
        separation_scores = []
        
        for layer in layers:
            # Get all 4 representations
            c1_vis = self.analyze_representation(concept1, "vision", layer, "vision", pooling)
            c1_txt = self.analyze_representation(concept1, "text", layer, "text", pooling)
            c2_vis = self.analyze_representation(concept2, "vision", layer, "vision", pooling)
            c2_txt = self.analyze_representation(concept2, "text", layer, "text", pooling)
            
            # Purity: average cross-modal similarity for same concepts
            cos_c1 = np.dot(c1_vis, c1_txt) / (np.linalg.norm(c1_vis) * np.linalg.norm(c1_txt) + 1e-8)
            cos_c2 = np.dot(c2_vis, c2_txt) / (np.linalg.norm(c2_vis) * np.linalg.norm(c2_txt) + 1e-8)
            purity = (cos_c1 + cos_c2) / 2.0
            
            # Separation: 1 - average within-modal similarity for different concepts
            cos_vis = np.dot(c1_vis, c2_vis) / (np.linalg.norm(c1_vis) * np.linalg.norm(c2_vis) + 1e-8)
            cos_txt = np.dot(c1_txt, c2_txt) / (np.linalg.norm(c1_txt) * np.linalg.norm(c2_txt) + 1e-8)
            separation = 1.0 - (cos_vis + cos_txt) / 2.0
            
            purity_scores.append(float(purity))
            separation_scores.append(float(separation))
        
        return {
            "purity_scores": purity_scores,
            "separation_scores": separation_scores,
            "layers": layers
        }

    def compute_clip_connector_comparison(self, concepts: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Compare raw CLIP embeddings vs post-connector embeddings for two concepts.
        
        This diagnostic analysis helps determine if CLIP already fails to distinguish concepts,
        or if the vision connector is crushing good CLIP features into a narrow subspace.
        
        Args:
            concepts: List of exactly 2 concepts to compare
            
        Returns:
            Tuple of (clip_matrix, connector_matrix, labels) where:
            - clip_matrix: 2×2 similarity matrix of raw CLIP embeddings
            - connector_matrix: 2×2 similarity matrix of post-connector embeddings
            - labels: List of concept names for axis labels
        """
        if len(concepts) != 2:
            raise ValueError(f"CLIP vs connector comparison requires exactly 2 concepts, got {len(concepts)}")
        
        print(f"\n🔬 Comparing raw CLIP vs post-connector embeddings for: {concepts}")
        
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
        
        # Extract raw CLIP embeddings (before connector)
        clip_embeddings = []
        labels = []
        for concept in concepts:
            image, label = load_image(concept)
            
            # Process through CLIP only (auto-resizes to 224×224)
            pixel_values = self.clip_processor(
                images=image, return_tensors="pt"
            ).pixel_values.to(self.device)
            
            with torch.no_grad():
                clip_output = self.vision_encoder(pixel_values).last_hidden_state
                # Mean-pool across all 257 tokens
                clip_embedding = clip_output.mean(dim=1).squeeze(0).cpu().float().numpy()
            
            clip_embeddings.append(clip_embedding)
            labels.append(label)
            print(f"  ✓ CLIP embedding for '{label}': shape={clip_embedding.shape}, norm={np.linalg.norm(clip_embedding):.2f}")
        
        # Extract post-connector embeddings (after connector)
        connector_embeddings = []
        for concept in concepts:
            image, label = load_image(concept)
            
            # Process through CLIP + connector (auto-resizes to 224×224)
            pixel_values = self.clip_processor(
                images=image, return_tensors="pt"
            ).pixel_values.to(self.device)
            
            with torch.no_grad():
                clip_output = self.vision_encoder(pixel_values).last_hidden_state
                connector_output = self.vision_connector(clip_output)
                # Mean-pool across all 257 tokens
                connector_embedding = connector_output.mean(dim=1).squeeze(0).cpu().float().numpy()
            
            connector_embeddings.append(connector_embedding)
            print(f"  ✓ Connector embedding for '{label}': shape={connector_embedding.shape}, norm={np.linalg.norm(connector_embedding):.2f}")
        
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
        
        print(f"\n  📊 CLIP similarity: {labels[0]} vs {labels[1]} = {clip_matrix[0, 1]:.4f}")
        print(f"  📊 Connector similarity: {labels[0]} vs {labels[1]} = {connector_matrix[0, 1]:.4f}")
        
        return clip_matrix, connector_matrix, labels

    def run_comprehensive_analysis(
        self,
        concepts: List[str],
        layers: List[int],
        output_dir: str = "results/cross_modality_purity",
    ) -> Dict:
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
        if hasattr(self, '_debug_mode') and self._debug_mode:
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
            if hasattr(self, '_debug_mode') and self._debug_mode and concept_idx == 0:
                print(f"   🐛 Debug info will be shown for layers: -1, 0, 15, 31")
            
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
                    results["cosine_similarity_mean_pooled"][concept][f"layer_{layer}"] = cosine_sim_mp

                    # Euclidean distance (CLS and mean-pooled)
                    euclidean_dist = self.compute_euclidean_distance(concept, layer, pooling="cls")
                    euclidean_dist_mp = self.compute_euclidean_distance(concept, layer, pooling="mean")
                    results["euclidean_distance"][concept][f"layer_{layer}"] = euclidean_dist
                    results["euclidean_distance_mean_pooled"][concept][f"layer_{layer}"] = euclidean_dist_mp

                    # Vocab predictions (CLS, mean-pooled, and text)
                    results["vocab_predictions"][concept][f"layer_{layer}"] = {
                        "vision_expert_cls": self.analyze_vocab(concept, "vision", layer, "vision", top_k=10, pooling="cls"),
                        "vision_expert_mean_pooled": self.analyze_vocab(concept, "vision", layer, "vision", top_k=10, pooling="mean"),
                        "text_expert": self.analyze_vocab(concept, "text", layer, "text", top_k=10),
                    }

                    print(f"✓ (cos_cls={cosine_sim:.3f}, cos_mp={cosine_sim_mp:.3f}, euc_cls={euclidean_dist:.2f}, euc_mp={euclidean_dist_mp:.2f})")

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

    def _plot_metric(self, results: Dict, concepts: List[str], layers: List[int], 
                     metric_key: str, output_dir: str, ylabel: str, title: str, 
                     filename: str, ylim: tuple = None):
        """Generic plotting function for metrics."""
        plt.figure(figsize=(12, 8))
        
        for concept in concepts:
            values = [results[metric_key][concept].get(f"layer_{layer}", 0.0) for layer in layers]
            plt.plot(layers, values, marker="o", linewidth=2, markersize=8, label=concept)
        
        if "Cosine" in ylabel:
            plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        
        plt.xlabel("Layer", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(f"{title}\n(Vision vs Text Expert)", fontweight="bold")
        plt.legend(loc="best", fontsize=10)
        plt.grid(True, alpha=0.3)
        if ylim:
            plt.ylim(ylim)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()
        print(f"  ✓ Saved {filename}")

    def _plot_purity_matrices(self, matrices: Dict, target_layers: List[int], output_dir: str):
        """Plot purity matrices as heatmaps for comparison across layers."""
        # Dynamically create subplots based on number of matrices actually computed
        num_matrices = len(matrices)
        fig, axes = plt.subplots(1, num_matrices, figsize=(6 * num_matrices, 5))
        
        # Handle case where only 1 matrix (axes won't be an array)
        if num_matrices == 1:
            axes = [axes]
        
        for idx, layer in enumerate(target_layers):
            if layer not in matrices:
                continue  # Skip if this layer wasn't computed
            matrix, labels = matrices[layer]
            
            # Create heatmap
            sns.heatmap(
                matrix, 
                annot=True, 
                fmt=".3f", 
                cmap="RdYlGn",
                vmin=-1, 
                vmax=1,
                xticklabels=labels,
                yticklabels=labels,
                ax=axes[idx],
                cbar_kws={'label': 'Cosine Similarity'},
                square=True
            )
            axes[idx].set_title(f"Layer {layer}", fontsize=14, fontweight="bold")
        
        plt.suptitle("Cross-Concept Purity Matrix (Mean-Pooled)", fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, "purity_matrix_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved purity_matrix_comparison.png")

    def _plot_purity_separation_scores(self, scores: Dict, output_dir: str):
        """Plot purity and separation scores across layers."""
        plt.figure(figsize=(12, 7))
        
        layers = scores["layers"]
        purity = scores["purity_scores"]
        separation = scores["separation_scores"]
        
        plt.plot(layers, purity, marker='o', linewidth=2.5, markersize=8, 
                label='Purity Score', color='#2ecc71')
        plt.plot(layers, separation, marker='s', linewidth=2.5, markersize=8, 
                label='Separation Score', color='#e74c3c')
        
        plt.xlabel("Layer", fontsize=13)
        plt.ylabel("Score", fontsize=13)
        plt.title("Layer-wise Purity and Separation Scores\n(Mean-Pooled Representations)", 
                 fontsize=15, fontweight="bold")
        plt.legend(loc="best", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, "purity_separation_scores.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"  ✓ Saved purity_separation_scores.png")

    def _plot_clip_connector_comparison(self, clip_matrix: np.ndarray, connector_matrix: np.ndarray, 
                                        labels: List[str], output_dir: str):
        """Plot side-by-side comparison of CLIP vs connector similarity matrices."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot raw CLIP embeddings
        sns.heatmap(
            clip_matrix,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=-1,
            vmax=1,
            xticklabels=labels,
            yticklabels=labels,
            ax=axes[0],
            cbar_kws={'label': 'Cosine Similarity'},
            square=True
        )
        axes[0].set_title("Raw CLIP Embeddings\n(1024-dim, mean-pooled)", fontsize=13, fontweight="bold")
        
        # Plot post-connector embeddings
        sns.heatmap(
            connector_matrix,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=-1,
            vmax=1,
            xticklabels=labels,
            yticklabels=labels,
            ax=axes[1],
            cbar_kws={'label': 'Cosine Similarity'},
            square=True
        )
        axes[1].set_title("Post-Connector Embeddings\n(4096-dim, mean-pooled)", fontsize=13, fontweight="bold")
        
        plt.suptitle("CLIP vs Vision Connector: Concept Similarity Comparison", 
                    fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, "clip_vs_connector_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved clip_vs_connector_comparison.png")

    def _visualize_results(self, results: Dict, output_dir: str):
        """Generate visualization plots from analysis results."""
        print("\n📈 Generating visualizations...")
        
        concepts = results["concepts"]
        layers = results["layers"]
        
        # Plot all four metrics
        self._plot_metric(results, concepts, layers, "cosine_similarity", output_dir,
                         "Cosine Similarity", "Cross-Modality Representation Similarity",
                         "cosine_similarity_lineplot.png", ylim=(-1, 1))
        
        self._plot_metric(results, concepts, layers, "euclidean_distance", output_dir,
                         "Euclidean Distance (L2 Norm)", "Cross-Modality Representation Distance",
                         "euclidean_distance_lineplot.png")
        
        self._plot_metric(results, concepts, layers, "cosine_similarity_mean_pooled", output_dir,
                         "Cosine Similarity", "Cross-Modality Representation Similarity (Mean-Pooled Vision)",
                         "cosine_similarity_meanpooled_lineplot.png", ylim=(-1, 1))
        
        self._plot_metric(results, concepts, layers, "euclidean_distance_mean_pooled", output_dir,
                         "Euclidean Distance (L2 Norm)", "Cross-Modality Representation Distance (Mean-Pooled Vision)",
                         "euclidean_distance_meanpooled_lineplot.png")
        
        # Generate purity matrix and divergence tracking if exactly 2 concepts
        if len(concepts) == 2:
            print("\n📊 Generating purity matrix and divergence analysis (2 concepts detected)...")
            
            # CLIP vs Connector comparison (diagnostic analysis)
            print("\n🔬 Running CLIP vs Connector diagnostic...")
            try:
                clip_matrix, connector_matrix, labels = self.compute_clip_connector_comparison(concepts)
                self._plot_clip_connector_comparison(clip_matrix, connector_matrix, labels, output_dir)
            except Exception as e:
                print(f"  ✗ Error computing CLIP vs connector comparison: {e}")
            
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
                self._plot_purity_matrices(matrices, target_layers, output_dir)
            
            # Purity/separation scores across all layers
            try:
                scores = self.compute_purity_separation_scores(concepts, layers, pooling="mean")
                self._plot_purity_separation_scores(scores, output_dir)
            except Exception as e:
                print(f"  ✗ Error computing purity/separation scores: {e}")
        
        print("✅ Visualization complete!")


def main():
    """Main entry point for cross-modality purity analysis."""
    parser = argparse.ArgumentParser(
        description="Cross-Modality Purity Analysis for MoE VLM"
    )
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

    args = parser.parse_args()

    # Handle --all-layers flag
    if args.all_layers:
        layers = [-1] + list(range(32))
        print(
            f"📊 Using --all-layers: analyzing layers {layers[0]} to {layers[-1]} ({len(layers)} total layers)"
        )
    else:
        layers = args.layers

    # Initialize analyzer
    print("=" * 80)
    print("Cross-Modality Purity Analysis")
    if args.debug:
        print("🐛 DEBUG MODE ENABLED")
    print("=" * 80)

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
