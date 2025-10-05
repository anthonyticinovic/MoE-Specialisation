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
        
        # Define color palette (RGB values)
        self.colors = {
            "red": (255, 0, 0),
            "blue": (0, 0, 255),
            "green": (0, 255, 0),
            "yellow": (255, 255, 0),
            "orange": (255, 165, 0),
            "purple": (128, 0, 128),
            "black": (0, 0, 0),
            "white": (255, 255, 255),
        }
        
    def generate_color_patch(self, color: str) -> Image.Image:
        """Generate a solid color patch."""
        if color not in self.colors:
            raise ValueError(f"Unknown color: {color}. Available: {list(self.colors.keys())}")
        
        rgb = self.colors[color]
        image = Image.new("RGB", (self.image_size, self.image_size), rgb)
        return image
    
    def generate_shape(self, shape: str, color: str = "red") -> Image.Image:
        """Generate a shape on white background."""
        if color not in self.colors:
            raise ValueError(f"Unknown color: {color}")
        if shape not in ["circle", "square", "triangle"]:
            raise ValueError(f"Unknown shape: {shape}")
        
        # Create white background
        image = Image.new("RGB", (self.image_size, self.image_size), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        # Calculate centered shape coordinates
        margin = self.image_size // 4
        rgb = self.colors[color]
        
        if shape == "circle":
            bbox = [margin, margin, self.image_size - margin, self.image_size - margin]
            draw.ellipse(bbox, fill=rgb, outline=rgb)
            
        elif shape == "square":
            bbox = [margin, margin, self.image_size - margin, self.image_size - margin]
            draw.rectangle(bbox, fill=rgb, outline=rgb)
            
        elif shape == "triangle":
            center_x = self.image_size // 2
            points = [
                (center_x, margin),  # Top
                (margin, self.image_size - margin),  # Bottom left
                (self.image_size - margin, self.image_size - margin),  # Bottom right
            ]
            draw.polygon(points, fill=rgb, outline=rgb)
        
        return image
    
    def generate_concept_image(self, concept: str) -> Image.Image:
        """
        Generate an image for a given concept.
        
        Handles:
        - Pure colors: "red", "blue", etc.
        - Shapes: "circle", "square", "triangle"
        - Colored shapes: "red circle", "blue square"
        """
        parts = concept.lower().split()
        
        if len(parts) == 1:
            # Single word - could be color or shape
            if parts[0] in self.colors:
                return self.generate_color_patch(parts[0])
            else:
                # Default to red shape
                return self.generate_shape(parts[0], color="red")
        
        elif len(parts) == 2:
            # Assume "color shape" format
            color, shape = parts[0], parts[1]
            return self.generate_shape(shape, color=color)
        
        else:
            raise ValueError(f"Cannot parse concept: {concept}")


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
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
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
        moe_model_path = "/data/gpfs/projects/COMP90055/aticinovic/models/Mistral-7B-MoE"
        print(f"  - Loading base MoE model from {moe_model_path}")
        self.llm = AutoModelForCausalLM.from_pretrained(
            moe_model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        
        # Load Stage 2 expert weights
        stage2_path = os.path.join(output_dir, "stage2_checkpoints", "llm_stage2_best.pth")
        print(f"  - Loading Stage 2 expert weights from {stage2_path}")
        expert_weights = torch.load(stage2_path, map_location=self.device)
        self.llm.load_state_dict(expert_weights, strict=False)
        
        # Load Stage 2.5 router weights
        stage2_5_path = os.path.join(
            output_dir, "stage2_5_checkpoints/archive", "llm_stage2_5_best.pth"
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
            output_dir, "archive/vision_connector_stage1_best.pth"
        )
        self.vision_connector.load_state_dict(
            torch.load(connector_path, map_location=self.device)
        )
        self.vision_connector.eval()
        
        print("✅ All models loaded successfully")
    
    def _prepare_vision_input(self, concept: str, use_connector: bool = True) -> torch.Tensor:
        """Generate synthetic image and convert to visual tokens.
        
        Args:
            concept: Concept to generate image for
            use_connector: If False, return raw CLIP embeddings with simple linear projection.
                          If True, use learned vision connector (default).
        """
        # Generate synthetic image
        image = self.image_generator.generate_concept_image(concept)
        
        # Process through CLIP
        pixel_values = self.clip_processor(
            images=image, return_tensors="pt"
        ).pixel_values.to(self.device)
        
        with torch.no_grad():
            patch_embeddings = self.vision_encoder(pixel_values).last_hidden_state
            
            if use_connector:
                # Use learned vision connector (layer -1)
                visual_tokens = self.vision_connector(patch_embeddings)
            else:
                # Simple Xavier-initialized linear projection (layer -2, baseline)
                # This gives us raw CLIP embeddings projected to 4096-dim without learned weights
                clip_dim = patch_embeddings.shape[-1]  # 1024
                llm_dim = 4096
                simple_proj = nn.Linear(clip_dim, llm_dim, bias=False).to(
                    self.device, dtype=patch_embeddings.dtype
                )
                nn.init.xavier_uniform_(simple_proj.weight)
                visual_tokens = simple_proj(patch_embeddings)
            
            # Convert to bfloat16 to match model dtype
            visual_tokens = visual_tokens.to(torch.bfloat16)
        
        return visual_tokens
    
    def _prepare_text_input(self, concept: str) -> torch.Tensor:
        """Tokenize concept and convert to text embeddings."""
        # Add prompt context for better embeddings
        text = f"The concept is {concept}."
        
        input_ids = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).input_ids.to(self.device)
        
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
        self,
        embeddings: torch.Tensor,
        routing_mask: torch.Tensor,
        target_layer: int
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
            inputs_embeds=embeddings,
            output_hidden_states=True,
            return_dict=True
        )
        
        # outputs.hidden_states is a tuple: (embedding_output, layer_0_output, ..., layer_31_output)
        # So target_layer 0 is at index 1, target_layer 31 is at index 32
        hidden_state = outputs.hidden_states[target_layer + 1]
        
        return hidden_state
    
    def analyze_representation(
        self,
        concept: str,
        expert: str,
        layer: int,
        modality: str
    ) -> np.ndarray:
        """
        Extract hidden state representation for a concept at a specific layer.
        
        Args:
            concept: Concept to analyze (e.g., "red", "circle")
            expert: "vision" or "text"
            layer: Layer index (0-31)
            modality: "vision" or "text" (input modality)
        
        Returns:
            Mean-pooled hidden state representation as numpy array
        """
        # Validate inputs
        if expert not in ["vision", "text"]:
            raise ValueError(f"Invalid expert: {expert}. Must be 'vision' or 'text'.")
        if modality not in ["vision", "text"]:
            raise ValueError(f"Invalid modality: {modality}. Must be 'vision' or 'text'.")
        if not (-2 <= layer < 32):
            raise ValueError(f"Invalid layer: {layer}. Must be in range [-2, 31].")
        
        expert_id = 0 if expert == "vision" else 1
        
        # Prepare input based on layer
        # Layer -2: Raw CLIP without learned connector (baseline)
        # Layer -1: CLIP with learned connector (post-training stage 1)
        # Layer 0-31: Transformer layers
        
        # For layers -2 and -1 (pre-transformer), use expert type to determine input
        # Vision expert gets vision embeddings, text expert gets text embeddings
        if layer in [-2, -1]:
            if expert == "vision":
                if layer == -2:
                    embeddings = self._prepare_vision_input(concept, use_connector=False)
                else:  # layer == -1
                    embeddings = self._prepare_vision_input(concept, use_connector=True)
            else:  # expert == "text"
                embeddings = self._prepare_text_input(concept)
            hidden_state = embeddings
        else:
            # For layers 0-31, use modality to determine input (same input, different experts)
            if modality == "vision":
                embeddings = self._prepare_vision_input(concept, use_connector=True)
            else:
                embeddings = self._prepare_text_input(concept)
            
            # Run through transformer with expert routing
            batch_size = embeddings.shape[0]
            num_tokens = embeddings.shape[1]
            routing_mask = self._force_routing_through_expert(expert_id, batch_size, num_tokens)
            
            # Extract hidden state
            with torch.no_grad():
                hidden_state = self._extract_hidden_state_at_layer(
                    embeddings, routing_mask, layer
                )
        
        # Mean pool across tokens
        mean_representation = hidden_state.mean(dim=1).squeeze(0)
        
        return mean_representation.cpu().float().numpy()
    
    def analyze_vocab(
        self,
        concept: str,
        expert: str,
        layer: int,
        modality: str,
        top_k: int = 10
    ) -> Dict[str, float]:
        """
        Analyze top-k vocabulary predictions for a concept.
        
        Args:
            concept: Concept to analyze
            expert: "vision" or "text"
            layer: Layer index
            modality: Input modality
            top_k: Number of top tokens to return
        
        Returns:
            Dictionary mapping tokens to probabilities
        """
        # Get hidden state at specified layer
        hidden_state = self.analyze_representation(concept, expert, layer, modality)
        hidden_state_tensor = torch.from_numpy(hidden_state).to(
            self.device, dtype=torch.bfloat16
        ).unsqueeze(0)
        
        # Apply final layer norm
        hidden_state_tensor = self.llm.model.norm(hidden_state_tensor)
        
        # Project to vocabulary space
        with torch.no_grad():
            logits = self.llm.lm_head(hidden_state_tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, top_k)
        
        # Convert to dictionary
        vocab_predictions = {}
        for prob, idx in zip(top_probs, top_indices):
            token = self.tokenizer.decode([idx.item()]).strip()
            vocab_predictions[token] = prob.item()
        
        return vocab_predictions
    
    def compute_cosine_similarity(
        self,
        concept: str,
        layer: int,
        modality: str = "vision"
    ) -> float:
        """
        Compute cosine similarity between vision and text expert representations.
        
        Args:
            concept: Concept to analyze
            layer: Layer index
            modality: Input modality (same for both experts)
        
        Returns:
            Cosine similarity score [-1, 1]
        """
        # Get representations from both experts
        vision_rep = self.analyze_representation(concept, "vision", layer, modality)
        text_rep = self.analyze_representation(concept, "text", layer, modality)
        
        # Compute cosine similarity
        cosine_sim = np.dot(vision_rep, text_rep) / (
            np.linalg.norm(vision_rep) * np.linalg.norm(text_rep) + 1e-8
        )
        
        return float(cosine_sim)
    
    def compute_probability_ratio(
        self,
        concept: str,
        layer: int,
        modality: str = "vision"
    ) -> Dict[str, float]:
        """
        Compute probability ratio for the target concept across experts.
        
        Returns:
            Dictionary with vision_prob, text_prob, and ratio
        """
        # Get vocab predictions from both experts
        vision_vocab = self.analyze_vocab(concept, "vision", layer, modality, top_k=50)
        text_vocab = self.analyze_vocab(concept, "text", layer, modality, top_k=50)
        
        # Find probability of target concept (try exact match and variations)
        concept_lower = concept.lower()
        
        def find_concept_prob(vocab_dict):
            # Try exact match first
            for token, prob in vocab_dict.items():
                if token.lower() == concept_lower:
                    return prob
            # Try partial matches
            for token, prob in vocab_dict.items():
                if concept_lower in token.lower() or token.lower() in concept_lower:
                    return prob
            return 0.0
        
        vision_prob = find_concept_prob(vision_vocab)
        text_prob = find_concept_prob(text_vocab)
        
        # Compute ratio (avoid division by zero)
        ratio = vision_prob / text_prob if text_prob > 1e-8 else float('inf')
        
        return {
            "vision_prob": vision_prob,
            "text_prob": text_prob,
            "ratio": ratio
        }
    
    def run_comprehensive_analysis(
        self,
        concepts: List[str],
        layers: List[int],
        output_dir: str = "results/cross_modality_purity"
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
        print(f"\n🔬 Running comprehensive analysis on {len(concepts)} concepts across {len(layers)} layers...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            "concepts": concepts,
            "layers": layers,
            "cosine_similarity": {},
            "probability_ratios": {},
            "vocab_predictions": {}
        }
        
        for concept in concepts:
            print(f"\n📊 Analyzing concept: '{concept}'")
            results["cosine_similarity"][concept] = {}
            results["probability_ratios"][concept] = {}
            results["vocab_predictions"][concept] = {}
            
            for layer in layers:
                print(f"  - Layer {layer}...", end=" ")
                
                try:
                    # Cosine similarity
                    cosine_sim = self.compute_cosine_similarity(
                        concept, layer, modality="vision"
                    )
                    results["cosine_similarity"][concept][f"layer_{layer}"] = cosine_sim
                    
                    # Probability ratio
                    prob_ratio = self.compute_probability_ratio(
                        concept, layer, modality="vision"
                    )
                    results["probability_ratios"][concept][f"layer_{layer}"] = prob_ratio
                    
                    # Vocab predictions (top-10)
                    vision_vocab = self.analyze_vocab(
                        concept, "vision", layer, "vision", top_k=10
                    )
                    text_vocab = self.analyze_vocab(
                        concept, "text", layer, "vision", top_k=10
                    )
                    results["vocab_predictions"][concept][f"layer_{layer}"] = {
                        "vision_expert": vision_vocab,
                        "text_expert": text_vocab
                    }
                    
                    print(f"✓ (cosine_sim={cosine_sim:.3f})")
                    
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
    
    def _visualize_results(self, results: Dict, output_dir: str):
        """Generate visualization plots from analysis results."""
        print("\n📈 Generating visualizations...")
        
        concepts = results["concepts"]
        layers = results["layers"]
        
        # 1. Cosine Similarity Heatmap
        cosine_matrix = np.zeros((len(concepts), len(layers)))
        for i, concept in enumerate(concepts):
            for j, layer in enumerate(layers):
                layer_key = f"layer_{layer}"
                cosine_matrix[i, j] = results["cosine_similarity"][concept].get(
                    layer_key, 0.0
                )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            cosine_matrix,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            xticklabels=[f"L{l}" for l in layers],
            yticklabels=concepts,
            center=0,
            vmin=-1,
            vmax=1,
            cbar_kws={"label": "Cosine Similarity"}
        )
        plt.title("Cross-Modality Representation Similarity\n(Vision vs Text Expert)", fontweight="bold")
        plt.xlabel("Layer")
        plt.ylabel("Concept")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cosine_similarity_heatmap.png"), dpi=300)
        plt.close()
        print("  ✓ Saved cosine_similarity_heatmap.png")
        
        # 2. Probability Ratio Analysis
        for concept in concepts:
            ratios = []
            for layer in layers:
                layer_key = f"layer_{layer}"
                prob_data = results["probability_ratios"][concept].get(layer_key, {})
                ratio = prob_data.get("ratio", 0)
                # Cap ratio at 10 for visualization
                ratios.append(min(ratio, 10) if ratio != float('inf') else 10)
            
            plt.figure(figsize=(10, 6))
            plt.plot(layers, ratios, marker='o', linewidth=2, markersize=8)
            plt.axhline(y=1, color='r', linestyle='--', label='Perfect Purity')
            plt.xlabel("Layer", fontsize=12)
            plt.ylabel("P(concept|vision) / P(concept|text)", fontsize=12)
            plt.title(f"Modality Purity for '{concept}'\n(Ratio closer to 0 = more purity)", fontweight="bold")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            safe_concept = concept.replace(" ", "_")
            plt.savefig(
                os.path.join(output_dir, f"prob_ratio_{safe_concept}.png"), dpi=300
            )
            plt.close()
        print(f"  ✓ Saved probability ratio plots for {len(concepts)} concepts")
        
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
        help="Concepts to analyze (e.g., red blue circle square)"
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[-2, -1, 0, 8, 16, 24, 31],
        help="Layer indices to analyze (-2=raw CLIP, -1=with connector, 0-31=transformer layers)"
    )
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="Analyze all layers from -2 to 31 (overrides --layers)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/cross_modality_purity",
        help="Directory to save results"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training config file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # Handle --all-layers flag
    if args.all_layers:
        layers = [-2, -1] + list(range(32))
        print(f"📊 Using --all-layers: analyzing layers {layers[0]} to {layers[-1]} ({len(layers)} total layers)")
    else:
        layers = args.layers
    
    # Initialize analyzer
    print("=" * 80)
    print("Cross-Modality Purity Analysis")
    print("=" * 80)
    
    analyzer = CrossModalityPurityAnalyzer(
        config_path=args.config,
        device=args.device
    )
    
    # Load models
    analyzer.load_models()
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis(
        concepts=args.concepts,
        layers=layers,
        output_dir=args.output_dir
    )
    
    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print(f"📁 Results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()