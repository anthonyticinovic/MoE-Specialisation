"""
Cross-Concept Similarity Matrix Analysis for MoE Vision-Language Models

This script computes 2N×2N similarity matrices comparing N image-text pairs at specified layers.

Stage 2 Mode: Uses expert routing to force images through vision expert and text through text expert.
Stage 3 Mode: Uses learned soft routing from end-to-end trained model (representation alignment).

Usage:
    # Stage 2 (forced routing)
    python analysis_scripts/cross_concept_similarity_matrix.py \\
        --config-file experiments/similarity_config.json \\
        --mode stage2

    # Stage 3 (learned routing + alignment analysis)
    python analysis_scripts/cross_concept_similarity_matrix.py \\
        --config-file experiments/similarity_config.json \\
        --mode stage3 \\
        --stage3-checkpoint /path/to/stage3_best_portable.pth

Config file format (JSON):
    {
      "concepts": ["cat", "dog", "car", "bus"],
      "samples_per_concept": 20,
      "annotations_file": "/path/to/coco/annotations/captions_train2017.json",
      "image_dir": "/path/to/coco/train2017",
      "layers": [0, 16, 31],
      "pooling": "mean",
      "output_dir": "results/similarity_matrix/",
      "mode": "stage2",  // or "stage3"
      "seed": 42
    }
"""

import torch
import numpy as np
import json
import os
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

from analysis_scripts.cross_modality_purity import CrossModalityPurityAnalyzer


class CrossConceptSimilarityAnalyzer:
    """
    Analyzer for computing cross-concept similarity matrices.
    
    Supports two modes:
    - Stage 2: Uses forced expert routing (vision→expert0, text→expert1)
    - Stage 3: Uses learned soft routing from end-to-end trained model
    
    Reuses core functionality from CrossModalityPurityAnalyzer for model loading,
    expert routing, and representation extraction.
    """
    
    def __init__(
        self,
        config_path: str = "configs/training_config.yaml",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mode: str = "stage2",
        stage2_checkpoint: Optional[str] = None,
        stage3_checkpoint: Optional[str] = None,
        temperature: float = 0.01,
    ):
        """
        Initialize analyzer by creating base analyzer for model access.
        
        Args:
            config_path: Path to training configuration
            device: Device to run on (cuda/cpu)
            mode: "stage2" or "stage3"
            stage2_checkpoint: Path to Stage 2 checkpoint (optional, defaults to best from training_config)
            stage3_checkpoint: Path to Stage 3 portable checkpoint (required if mode="stage3")
            temperature: Routing temperature for Stage 3 (lower = more deterministic)
        """
        print(f"🔧 Initializing Cross-Concept Similarity Analyzer")
        print(f"   Mode: {mode.upper()}")
        print(f"   Device: {device}")
        
        self.mode = mode
        self.device = device
        self.temperature = temperature
        self.stage2_checkpoint = stage2_checkpoint
        self.stage3_checkpoint = stage3_checkpoint
        
        if mode == "stage3" and stage3_checkpoint is None:
            raise ValueError("stage3_checkpoint path required when mode='stage3'")
        
        if mode == "stage2":
            # Initialize base analyzer for Stage 2 forced routing
            print("   Using Stage 2 forced expert routing")
            if stage2_checkpoint:
                print(f"   Stage 2 checkpoint: {stage2_checkpoint}")
            self.base_analyzer = CrossModalityPurityAnalyzer(
                config_path=config_path,
                device=device
            )
        elif mode == "stage3":
            # For Stage 3, we'll load models directly
            print(f"   Using Stage 3 learned soft routing (temperature={temperature})")
            print(f"   Checkpoint: {stage3_checkpoint}")
            self.config_path = config_path
            self.base_analyzer = None  # No base analyzer for Stage 3
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'stage2' or 'stage3'")
    
    
    def load_models(self):
        """Load all required models (delegates to appropriate method based on mode)."""
        if self.mode == "stage2":
            self._load_stage2_models()
        elif self.mode == "stage3":
            self._load_stage3_models()
    
    def _load_stage2_models(self):
        """Load Stage 2 models with forced expert routing.
        
        Uses base analyzer's load_models() which loads:
        1. Stage 1 vision connector (vision_connector_stage1_best.pth)
        2. Stage 2 LLM checkpoint (llm_stage2_best.pth) with trained experts
        3. Hard routing mode enabled
        
        Optionally overrides the Stage 2 checkpoint path if provided.
        """
        print("\n📦 Loading Stage 2 models...")
        
        # If custom checkpoint path provided, load models with custom checkpoint
        if self.stage2_checkpoint:
            print(f"   Using custom Stage 2 checkpoint: {self.stage2_checkpoint}")
            
            # Import required classes
            from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, CLIPVisionModel
            from models import VisionLanguageConnector
            
            # Load base models (tokenizer, CLIP, MoE architecture, vision connector)
            paths = self.base_analyzer.config["paths"]
            output_dir = paths["output_dir"]
            
            # Load tokenizer
            print(f"  - Loading tokenizer from {paths['mistral_local_path']}")
            self.base_analyzer.tokenizer = AutoTokenizer.from_pretrained(paths["mistral_local_path"])
            self.base_analyzer.tokenizer.pad_token = self.base_analyzer.tokenizer.eos_token
            
            # Load CLIP processor and vision encoder
            print(f"  - Loading CLIP from {paths['clip_local_path']}")
            self.base_analyzer.clip_processor = AutoProcessor.from_pretrained(paths["clip_local_path"])
            self.base_analyzer.vision_encoder = CLIPVisionModel.from_pretrained(
                paths["clip_local_path"]
            ).to(self.device)
            self.base_analyzer.vision_encoder.eval()
            
            # Load base MoE model
            moe_model_path = "/data/gpfs/projects/COMP90055/aticinovic/models/Mistral-7B-MoE"
            print(f"  - Loading base MoE model from {moe_model_path}")
            self.base_analyzer.llm = AutoModelForCausalLM.from_pretrained(
                moe_model_path,
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
            ).to(self.device)
            
            # Load custom Stage 2 checkpoint (load to CPU first to avoid OOM)
            print(f"  - Loading Stage 2 expert weights from {self.stage2_checkpoint}")
            expert_weights = torch.load(self.stage2_checkpoint, map_location='cpu')
            self.base_analyzer.llm.load_state_dict(expert_weights, strict=False)
            self.base_analyzer.llm.eval()
            
            # Force hard routing mode
            for layer in self.base_analyzer.llm.model.layers:
                if hasattr(layer.mlp, "routing_mode"):
                    layer.mlp.routing_mode = "hard"
            
            # Load vision connector
            print("  - Loading vision connector")
            self.base_analyzer.vision_connector = VisionLanguageConnector().to(self.device)
            connector_path = os.path.join(output_dir, "vision_connector_stage1_best.pth")
            self.base_analyzer.vision_connector.load_state_dict(
                torch.load(connector_path, map_location=self.device)
            )
            self.base_analyzer.vision_connector.eval()
            
            print("✅ All Stage 2 models loaded successfully (custom checkpoint)")
        else:
            # Use default Stage 2 checkpoint path from base analyzer
            print("   Using default Stage 2 checkpoint from training_config.yaml")
            self.base_analyzer.load_models()
    
    def _load_stage3_models(self):
        """Load Stage 3 end-to-end trained model with learned soft routing.
        
        Reuses base analyzer infrastructure for CLIP/tokenizer, only differs in:
        1. Model checkpoint (Stage 3 vs Stage 2/2.5)
        2. Routing mode (soft vs hard)
        3. Temperature setting for deterministic analysis
        """
        from models import VisionLanguageConnector
        
        print("\n📦 Loading Stage 3 models...")
        
        # Create base analyzer to reuse CLIP/tokenizer loading logic
        self.base_analyzer = CrossModalityPurityAnalyzer(
            config_path=self.config_path,
            device=self.device
        )
        
        # Load base models (CLIP, tokenizer, MoE architecture)
        # This loads Stage 2/2.5 weights initially, but we'll override with Stage 3
        self.base_analyzer.load_models()
        
        # Now override with Stage 3 checkpoint
        print(f"   Loading Stage 3 checkpoint: {self.stage3_checkpoint}")
        checkpoint = torch.load(self.stage3_checkpoint, map_location=self.device)
        
        # Check if this is a full checkpoint or portable checkpoint
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # FULL checkpoint format (with training state)
            print(f"      Detected FULL checkpoint format")
            self.base_analyzer.llm.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"      ✓ Loaded LLM weights (epoch {checkpoint.get('epoch', 'unknown')})")
            
            if 'connector_state_dict' in checkpoint:
                self.base_analyzer.vision_connector.load_state_dict(checkpoint['connector_state_dict'])
                print(f"      ✓ Loaded vision connector weights")
        else:
            # PORTABLE checkpoint format (direct state_dict)
            print(f"      Detected PORTABLE checkpoint format (state_dict only)")
            self.base_analyzer.llm.load_state_dict(checkpoint, strict=False)
            print(f"      ✓ Loaded LLM weights (portable format)")
            print(f"      ⚠️  Note: Vision connector NOT updated (using Stage 1 weights)")
        
        # Set all MoE layers to soft routing mode (CRITICAL DIFFERENCE from Stage 2)
        print("   Setting MoE layers to soft routing mode...")
        for layer in self.base_analyzer.llm.model.layers:
            if hasattr(layer.mlp, "routing_mode"):
                layer.mlp.routing_mode = 'soft'
                # Set low temperature for deterministic routing
                layer.mlp._forward_temperature = self.temperature
        
        # Ensure models are in eval mode for deterministic behavior
        self.base_analyzer.llm.eval()
        self.base_analyzer.vision_encoder.eval()
        self.base_analyzer.vision_connector.eval()
        
        # Set random seed for reproducibility (Gumbel noise)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        print("   ✅ Stage 3 models loaded successfully!")
        print(f"      Routing mode: SOFT (learned, no forcing)")
        print(f"      Temperature: {self.temperature} (lower = more deterministic routing)")
        print(f"      Random seed: 42 (for reproducibility)")
        print(f"      Validation loss: {checkpoint.get('val_loss', 'N/A')}")
    
    def extract_concept_samples(
        self,
        annotations_file: str,
        concepts: List[str],
        samples_per_concept: int,
        seed: int = 42
    ) -> Dict[str, List[Dict]]:
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
        print(f"\n📚 Extracting concept samples from COCO annotations...")
        print(f"   Concepts: {concepts}")
        print(f"   Target: {samples_per_concept} samples per concept")
        
        # Load COCO annotations
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        # Build image_id -> image_path mapping
        image_id_to_path = {}
        for img in coco_data['images']:
            image_id_to_path[img['id']] = img['file_name']
        
        # Set random seed
        np.random.seed(seed)
        
        # Extract samples for each concept
        concept_samples = {concept: [] for concept in concepts}
        
        for annotation in coco_data['annotations']:
            caption = annotation['caption'].lower()
            image_id = annotation['image_id']
            words = set(caption.split())
            
            # Check which concepts appear in this caption
            # Support both single words (e.g., "cat") and compound concepts (e.g., "red_apple")
            matching_concepts = []
            for concept in concepts:
                concept_lower = concept.lower()
                
                # Check if it's a compound concept (contains underscore)
                if '_' in concept_lower:
                    # For compound concepts like "red_apple", check if all parts are present
                    parts = concept_lower.split('_')
                    if all(part in words for part in parts):
                        matching_concepts.append(concept)
                else:
                    # For single-word concepts, check if word is in caption
                    if concept_lower in words:
                        matching_concepts.append(concept)
            
            # Skip if multiple specified concepts appear (ambiguous)
            if len(matching_concepts) > 1:
                continue
            
            # Skip if no concepts match
            if len(matching_concepts) == 0:
                continue
            
            # Add to the matching concept's sample list
            concept = matching_concepts[0]
            if len(concept_samples[concept]) < samples_per_concept:
                concept_samples[concept].append({
                    'image_id': image_id,
                    'caption': annotation['caption'],
                    'image_path': image_id_to_path[image_id],
                    'concept': concept
                })
        
        # Print statistics
        print(f"\n   📊 Extracted samples:")
        for concept, samples in concept_samples.items():
            print(f"      {concept}: {len(samples)} samples")
        
        # Warn if any concept is under-sampled
        for concept, samples in concept_samples.items():
            if len(samples) < samples_per_concept:
                print(f"   ⚠️  Warning: Only found {len(samples)} samples for '{concept}' (target: {samples_per_concept})")
        
        return concept_samples
    
    
    def _extract_representation(
        self,
        concept: str,
        expert: str,
        layer: int,
        modality: str,
        pooling: str = "mean"
    ) -> np.ndarray:
        """
        Extract hidden state representation for a concept at a specific layer.
        
        For Stage 2: Uses forced routing (wrapper around base analyzer)
        For Stage 3: Uses learned soft routing (natural model behavior)
        
        Args:
            concept: Image path (for vision) or text string (for text)
            expert: "vision" or "text" - which expert to route through (Stage 2 only, ignored in Stage 3)
            layer: Layer index to extract from (0-31)
            modality: "vision" or "text" - type of input
            pooling: "mean" for mean-pooling (default)
        
        Returns:
            Numpy array of shape [hidden_dim] representing the pooled hidden state
        """
        if self.mode == "stage2":
            # Delegate to base analyzer's representation extraction logic
            return self.base_analyzer.analyze_representation(
                concept=concept,
                expert=expert,
                layer=layer,
                modality=modality,
                pooling=pooling
            )
        elif self.mode == "stage3":
            # Extract using learned soft routing
            return self._extract_stage3_representation(
                concept=concept,
                layer=layer,
                modality=modality,
                pooling=pooling
            )
    
    def _extract_stage3_representation(
        self,
        concept: str,
        layer: int,
        modality: str,
        pooling: str = "mean"
    ) -> np.ndarray:
        """
        Extract representation from Stage 3 model using learned soft routing.
        
        REUSES base analyzer's input preparation (_prepare_vision_input, _prepare_text_input).
        DIFFERS in: no routing masks, custom forward pass with layer extraction.
        
        Args:
            concept: Image path or text string
            layer: Layer index (0-31)
            modality: "vision" or "text"
            pooling: Pooling strategy (mean recommended for cross-modal alignment)
        
        Returns:
            Pooled representation as numpy array
        """
        with torch.no_grad():
            # REUSE: Input preparation from base analyzer (CLIP + connector for vision, tokenizer for text)
            if modality == "vision":
                visual_soft_tokens = self.base_analyzer._prepare_vision_input(concept)
                inputs_embeds = visual_soft_tokens
                attention_mask = None  # Vision tokens don't need masking
                num_content_tokens = 257  # All 257 visual tokens
                
            elif modality == "text":
                text_embeddings = self.base_analyzer._prepare_text_input(concept)
                inputs_embeds = text_embeddings
                # Create attention mask for text (all ones, no padding in single-sample case)
                attention_mask = torch.ones(text_embeddings.shape[:2], device=self.device)
                # IMPORTANT: Exclude BOS token (position 0) to match Stage 2 behavior
                num_content_tokens = text_embeddings.shape[1]  # Total tokens including BOS
                
            else:
                raise ValueError(f"Invalid modality: {modality}")
            
            # DIFFERS: Extract hidden state at specified layer with learned soft routing (no masks)
            hidden_states = self._forward_with_layer_extraction(
                inputs_embeds=inputs_embeds,
                target_layer=layer,
                attention_mask=attention_mask
            )
            
            # Pool over content tokens
            if pooling == "mean":
                if modality == "vision":
                    # Mean pool over all 257 visual tokens
                    representation = hidden_states[0, :, :].mean(dim=0).float().cpu().numpy()
                else:
                    # Mean pool over text tokens EXCLUDING BOS (position 0) to match Stage 2
                    # Stage 2 does: hidden_state[:, 1:, :].mean(dim=1)
                    # We replicate: hidden_states[0, 1:, :].mean(dim=0)
                    seq_len = hidden_states.shape[1]
                    if seq_len > 1:
                        # Exclude BOS token at position 0
                        representation = hidden_states[0, 1:, :].mean(dim=0).float().cpu().numpy()
                    else:
                        # Edge case: only BOS token (shouldn't happen)
                        representation = hidden_states[0, 0, :].float().cpu().numpy()
            else:
                raise ValueError(f"Unsupported pooling: {pooling}")
        
        return representation
    
    def _forward_with_layer_extraction(
        self,
        inputs_embeds: torch.Tensor,
        target_layer: int,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through model and extract hidden states at target layer.
        
        Uses base analyzer's LLM with learned soft routing (no routing masks).
        
        Args:
            inputs_embeds: Input embeddings [1, seq_len, hidden_dim]
            target_layer: Layer to extract from (0-31)
            attention_mask: Optional attention mask
        
        Returns:
            Hidden states at target layer [1, seq_len, hidden_dim]
        """
        if attention_mask is None:
            attention_mask = torch.ones(inputs_embeds.shape[:2], device=self.device)
        
        # Use base analyzer's LLM (loaded with Stage 3 weights)
        # Forward pass WITHOUT routing masks (soft routing will use learned behavior)
        outputs = self.base_analyzer.llm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # outputs.hidden_states is a tuple: (embedding_output, layer_0_output, ..., layer_31_output)
        # target_layer 0 is at index 1, target_layer 31 is at index 32
        hidden_states = outputs.hidden_states[target_layer + 1]
        
        return hidden_states
    
    
    def _compute_cosine_similarity_matrix(
        self,
        representations: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix for a list of representations.
        
        Args:
            representations: List of N representations, each of shape [hidden_dim]
        
        Returns:
            N×N matrix where matrix[i,j] = cosine_similarity(rep_i, rep_j)
        """
        n = len(representations)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    # Cosine similarity: dot(a,b) / (norm(a) * norm(b))
                    cos_sim = np.dot(representations[i], representations[j]) / (
                        np.linalg.norm(representations[i]) * 
                        np.linalg.norm(representations[j]) + 1e-8
                    )
                    matrix[i, j] = float(cos_sim)
        
        return matrix
    
    def compute_cross_concept_matrix(
        self,
        concept_samples: Dict[str, List[Dict]],
        image_dir: str,
        layer: int = 31,
        pooling: str = "mean"
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute 2N×2N cross-concept similarity matrix from COCO samples.
        
        For N concepts with S samples each, creates a 2N×2N matrix where:
        - First N rows/cols are average image representations per concept
        - Last N rows/cols are average text representations per concept
        
        Args:
            concept_samples: Dict mapping concept -> list of sample dicts
            image_dir: Base directory for COCO images
            layer: Layer to extract representations from (default: 31)
            pooling: Pooling strategy (default: "mean")
        
        Returns:
            Tuple of:
                - matrix: 2N×2N numpy array of cosine similarities
                - labels: List of 2N labels for rows/columns
        
        Matrix structure:
            [img:cat, img:dog, ..., txt:cat, txt:dog, ...]
        """
        N = len(concept_samples)
        print(f"\n🔬 Computing {2*N}×{2*N} similarity matrix at layer {layer}")
        print(f"   Pooling strategy: {pooling}")
        print(f"   Number of concepts: {N}")
        
        representations = []
        labels = []
        
        # Extract image representations (first N entries) - averaged per concept
        print(f"\n📸 Extracting image representations through vision expert...")
        for concept, samples in concept_samples.items():
            print(f"   Concept: {concept} ({len(samples)} samples)")
            
            concept_img_reps = []
            
            for idx, sample in enumerate(samples):
                image_path = os.path.join(image_dir, sample['image_path'])
                
                if idx % 10 == 0 and idx > 0:
                    print(f"      Progress: {idx}/{len(samples)} samples")
                
                try:
                    img_rep = self._extract_representation(
                        concept=image_path,
                        expert="vision",
                        layer=layer,
                        modality="vision",
                        pooling=pooling
                    )
                    concept_img_reps.append(img_rep)
                except Exception as e:
                    print(f"      ⚠️  Error processing {image_path}: {e}")
                    continue
            
            # Average representations across samples
            if len(concept_img_reps) > 0:
                avg_img_rep = np.mean(np.stack(concept_img_reps), axis=0)
                representations.append(avg_img_rep)
                labels.append(f"img:{concept}")
                print(f"       ✓ Averaged {len(concept_img_reps)} samples: norm={np.linalg.norm(avg_img_rep):.2f}")
            else:
                print(f"       ⚠️  No valid samples for {concept}, skipping")
        
        # Extract text representations (next N entries) - averaged per concept
        print(f"\n💬 Extracting text representations through text expert...")
        for concept, samples in concept_samples.items():
            print(f"   Concept: {concept} ({len(samples)} samples)")
            
            concept_txt_reps = []
            
            for idx, sample in enumerate(samples):
                text = sample['caption']
                
                if idx % 10 == 0 and idx > 0:
                    print(f"      Progress: {idx}/{len(samples)} samples")
                
                try:
                    txt_rep = self._extract_representation(
                        concept=text,
                        expert="text",
                        layer=layer,
                        modality="text",
                        pooling=pooling
                    )
                    concept_txt_reps.append(txt_rep)
                except Exception as e:
                    print(f"      ⚠️  Error processing text '{text}': {e}")
                    continue
            
            # Average representations across samples
            if len(concept_txt_reps) > 0:
                avg_txt_rep = np.mean(np.stack(concept_txt_reps), axis=0)
                representations.append(avg_txt_rep)
                labels.append(f"txt:{concept}")
                print(f"       ✓ Averaged {len(concept_txt_reps)} samples: norm={np.linalg.norm(avg_txt_rep):.2f}")
            else:
                print(f"       ⚠️  No valid samples for {concept}, skipping")
        
        # Compute pairwise similarity matrix
        print(f"\n📊 Computing {2*N}×{2*N} cosine similarity matrix...")
        matrix = self._compute_cosine_similarity_matrix(representations)
        
        print(f"   ✓ Matrix computed: shape={matrix.shape}")
        print(f"   ✓ Similarity range: [{matrix.min():.3f}, {matrix.max():.3f}]")
        
        return matrix, labels
    
    def compute_color_coherence_score(
        self,
        matrix: np.ndarray,
        labels: List[str]
    ) -> dict:
        """
        Compute Color Coherence Score (CCS) to measure color-object binding.
        
        CCS = mean(same_color_diff_object) / mean(diff_color_same_object)
        
        CCS > 1.0: Strong color binding (color is disentangled from object)
        CCS ≈ 1.0: No color binding (color and object are entangled)
        CCS < 1.0: Anti-binding (objects dominate over color)
        
        Args:
            matrix: 2N×2N similarity matrix
            labels: List of row/column labels (format: "modality:concept")
        
        Returns:
            Dictionary containing CCS scores and component similarities
        """
        # Parse labels to extract concepts
        # Format: "img:red_apple", "txt:blue_car", etc.
        concepts = []
        for label in labels:
            if ':' in label:
                modality, concept = label.split(':', 1)
                concepts.append({'modality': modality, 'concept': concept})
            else:
                concepts.append({'modality': 'unknown', 'concept': label})
        
        # Extract color and object from concept names (assumes format: color_object or just object)
        def parse_concept(concept_str):
            """Parse 'red_apple' -> ('red', 'apple'), 'banana' -> (None, 'banana')"""
            parts = concept_str.split('_', 1)
            if len(parts) == 2:
                color, obj = parts
                # Check if first part is actually a color (heuristic)
                color_words = ['red', 'green', 'blue', 'yellow', 'white', 'black', 'pink', 'orange', 'purple']
                if color.lower() in color_words:
                    return (color, obj)
            return (None, concept_str)
        
        parsed_concepts = [parse_concept(c['concept']) for c in concepts]
        
        # Check if we have colored concepts
        has_colors = any(color is not None for color, _ in parsed_concepts)
        
        if not has_colors:
            print("\n   ⚠️  No colored concepts detected (format should be 'color_object')")
            return {
                'color_coherence_score': None,
                'same_color_diff_object_mean': None,
                'diff_color_same_object_mean': None,
                'note': 'No colored concepts found'
            }
        
        # Extract cross-modal similarities (txt rows × img columns)
        n = len(labels)
        n_concepts = n // 2
        
        # Indices: first half are images, second half are text
        img_indices = list(range(n_concepts))
        txt_indices = list(range(n_concepts, n))
        
        same_color_diff_object = []
        diff_color_same_object = []
        
        # Compare all txt-img pairs
        for txt_idx in txt_indices:
            txt_color, txt_obj = parsed_concepts[txt_idx]
            if txt_color is None:
                continue
                
            for img_idx in img_indices:
                img_color, img_obj = parsed_concepts[img_idx]
                if img_color is None:
                    continue
                
                similarity = matrix[txt_idx, img_idx]
                
                # Same color, different object (e.g., red_apple text vs red_car image)
                if txt_color == img_color and txt_obj != img_obj:
                    same_color_diff_object.append(similarity)
                
                # Different color, same object (e.g., red_apple text vs green_apple image)
                elif txt_color != img_color and txt_obj == img_obj:
                    diff_color_same_object.append(similarity)
        
        # Compute metrics
        if len(same_color_diff_object) == 0 or len(diff_color_same_object) == 0:
            print("\n   ⚠️  Insufficient color-object pairs for CCS computation")
            return {
                'color_coherence_score': None,
                'same_color_diff_object_mean': None,
                'diff_color_same_object_mean': None,
                'note': 'Insufficient pairs'
            }
        
        same_color_mean = np.mean(same_color_diff_object)
        diff_color_mean = np.mean(diff_color_same_object)
        ccs = same_color_mean / diff_color_mean if diff_color_mean != 0 else None
        
        return {
            'color_coherence_score': ccs,
            'same_color_diff_object_mean': same_color_mean,
            'diff_color_same_object_mean': diff_color_mean,
            'same_color_diff_object_count': len(same_color_diff_object),
            'diff_color_same_object_count': len(diff_color_same_object)
        }
    
    def save_results(
        self,
        matrix: np.ndarray,
        labels: List[str],
        output_dir: str,
        layer: int
    ):
        """
        Save similarity matrix and labels to JSON files.
        
        Args:
            matrix: 2N×2N similarity matrix
            labels: List of row/column labels
            output_dir: Directory to save results
            layer: Layer number (for filename)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Compute color coherence score
        print(f"\n   🎨 Computing Color Coherence Score (CCS)...")
        ccs_results = self.compute_color_coherence_score(matrix, labels)
        
        if ccs_results['color_coherence_score'] is not None:
            print(f"   📊 Color Coherence Score: {ccs_results['color_coherence_score']:.3f}")
            print(f"      • Same color, diff object: {ccs_results['same_color_diff_object_mean']:.3f} "
                  f"(n={ccs_results['same_color_diff_object_count']} pairs)")
            print(f"      • Diff color, same object: {ccs_results['diff_color_same_object_mean']:.3f} "
                  f"(n={ccs_results['diff_color_same_object_count']} pairs)")
            
            # Interpret the score
            ccs = ccs_results['color_coherence_score']
            if ccs > 1.05:
                interpretation = "✅ Strong color binding (color disentangled from object)"
            elif ccs > 0.95:
                interpretation = "⚠️  Weak/no color binding (color-object entangled)"
            else:
                interpretation = "❌ Category dominance (object identity dominates over color)"
            print(f"      • Interpretation: {interpretation}")
        
        # Save matrix as JSON (convert to list for JSON serialization)
        matrix_path = os.path.join(output_dir, f"similarity_matrix_layer{layer}.json")
        with open(matrix_path, "w") as f:
            json.dump({
                "matrix": matrix.tolist(),
                "layer": layer,
                "shape": list(matrix.shape),
                "color_coherence_score": ccs_results
            }, f, indent=2)
        print(f"   ✓ Saved matrix to: {matrix_path}")
        
        # Save labels as JSON
        labels_path = os.path.join(output_dir, f"labels_layer{layer}.json")
        with open(labels_path, "w") as f:
            json.dump({
                "labels": labels,
                "layer": layer,
                "num_pairs": len(labels) // 2
            }, f, indent=2)
        print(f"   ✓ Saved labels to: {labels_path}")
    
    def visualize_matrix(
        self,
        matrix: np.ndarray,
        labels: List[str],
        output_dir: str,
        layer: int
    ):
        """
        Create and save heatmap visualizations of similarity matrix.
        
        Generates two plots:
        1. Standard: Single color scale for all values
        2. Dual-scale: Separate color scales for within-modality vs cross-modality
        
        Args:
            matrix: 2N×2N similarity matrix
            labels: List of row/column labels
            output_dir: Directory to save plot
            layer: Layer number (for title and filename)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine figure size based on matrix size
        n = matrix.shape[0]
        figsize = max(10, n * 0.6)
        mode_str = "Forced Routing" if self.mode == "stage2" else f"Learned Soft Routing (T={self.temperature})"
        
        # ============================================================
        # PLOT 1: Standard single color scale (original)
        # ============================================================
        fig, ax = plt.subplots(figsize=(figsize, figsize))
        
        # Create mask for upper triangle (exclude diagonal)
        mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
        
        # Create heatmap
        sns.heatmap(
            matrix,
            mask=mask,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=-1,
            vmax=1,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar_kws={'label': 'Cosine Similarity'},
            square=True,
            linewidths=0.5,
            linecolor='lightgray'
        )
        
        # Customize labels
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(labels, rotation=0, fontsize=9)
        
        # Add title
        ax.set_title(
            f'Cross-Concept Similarity Matrix (Layer {layer})\n'
            f'{n//2} Concepts | Stage 3 | Soft Routing',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f"similarity_matrix_layer{layer}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved standard heatmap to: {plot_path}")
        
        # ============================================================
        # PLOT 2: Cross-modal only (txt vs img)
        # ============================================================
        self._visualize_cross_modal_only(matrix, labels, output_dir, layer, mode_str)
    
    def _visualize_cross_modal_only(
        self,
        matrix: np.ndarray,
        labels: List[str],
        output_dir: str,
        layer: int,
        mode_str: str
    ):
        """
        Create focused heatmap showing only cross-modal (txt↔img) similarities.
        
        Extracts the bottom-left quadrant (txt rows × img columns) and displays
        it with an optimized color scale for maximum sensitivity.
        
        Args:
            matrix: 2N×2N similarity matrix
            labels: List of row/column labels
            output_dir: Directory to save plot
            layer: Layer number
            mode_str: Mode description string for title
        """
        n = matrix.shape[0]
        half_n = n // 2  # Split point between img and txt
        
        # Extract cross-modal submatrix (txt rows × img columns)
        cross_modal_matrix = matrix[half_n:, :half_n]
        
        # Extract corresponding labels
        img_labels = [l.replace('img:', '') for l in labels[:half_n]]
        txt_labels = [l.replace('txt:', '') for l in labels[half_n:]]
        
        # Compute statistics
        print(f"\n   📊 Cross-modal only statistics:")
        print(f"      Shape: {cross_modal_matrix.shape} (txt × img)")
        print(f"      Range: [{cross_modal_matrix.min():.3f}, {cross_modal_matrix.max():.3f}]")
        print(f"      Mean: {cross_modal_matrix.mean():.3f}")
        print(f"      Std: {cross_modal_matrix.std():.3f}")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create heatmap with optimized color scale
        sns.heatmap(
            cross_modal_matrix,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=cross_modal_matrix.min() - 0.005,  # Add small margin
            vmax=cross_modal_matrix.max() + 0.005,
            xticklabels=img_labels,
            yticklabels=txt_labels,
            ax=ax,
            cbar_kws={'label': 'Cosine Similarity (Cross-Modal)'},
            square=True,
            linewidths=0.5,
            linecolor='lightgray'
        )
        
        # Customize labels
        ax.set_xlabel('Image Concepts', fontsize=12, fontweight='bold')
        ax.set_ylabel('Text Concepts', fontsize=12, fontweight='bold')
        ax.set_xticklabels(img_labels, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(txt_labels, rotation=0, fontsize=10)
        
        # Add title
        ax.set_title(
            f'Cross-Modal Similarity Matrix (Layer {layer})\n'
            f'Text ↔ Image Alignment | {half_n} Concepts | {mode_str}',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f"similarity_matrix_layer{layer}_cross_modal.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved cross-modal heatmap to: {plot_path}")
    
    def run_analysis(
        self,
        concepts: List[str],
        samples_per_concept: int,
        annotations_file: str,
        image_dir: str,
        layers: List[int] = [31],
        pooling: str = "mean",
        output_dir: str = "results/similarity_matrix/",
        seed: int = 42
    ) -> Dict:
        """
        Run complete similarity matrix analysis for specified layers using COCO concept sampling.
        
        Args:
            concepts: List of concept keywords (e.g., ["cat", "dog", "car"])
            samples_per_concept: Number of samples to extract per concept
            annotations_file: Path to COCO annotations JSON
            image_dir: Base directory for COCO images
            layers: List of layer indices to analyze
            pooling: Pooling strategy
            output_dir: Output directory for results
            seed: Random seed for reproducibility
        
        Returns:
            Dictionary containing results for each layer
        """
        print("=" * 80)
        print("Cross-Concept Similarity Matrix Analysis")
        print("=" * 80)
        print(f"Concepts: {concepts}")
        print(f"Samples per concept: {samples_per_concept}")
        print(f"Layers to analyze: {layers}")
        print(f"Total layers: {len(layers)}")
        print("=" * 80)
        
        # Extract concept samples once (shared across all layers)
        concept_samples = self.extract_concept_samples(
            annotations_file=annotations_file,
            concepts=concepts,
            samples_per_concept=samples_per_concept,
            seed=seed
        )
        
        results = {}
        
        # Process each layer
        print(f"\n🔍 DEBUG: About to iterate through {len(layers)} layers: {layers}")
        for layer_idx, layer in enumerate(layers):
            print(f"\n{'='*80}")
            print(f"🔍 DEBUG: Starting iteration {layer_idx + 1}/{len(layers)}")
            print(f"LAYER {layer} ({layer_idx + 1}/{len(layers)})")
            print(f"{'='*80}")
            
            try:
                # Compute similarity matrix
                print(f"\n🔄 Starting matrix computation for layer {layer}...")
                print(f"🔍 DEBUG: Calling compute_cross_concept_matrix with layer={layer}")
                matrix, labels = self.compute_cross_concept_matrix(
                    concept_samples=concept_samples,
                    image_dir=image_dir,
                    layer=layer,
                    pooling=pooling
                )
                print(f"✓ Matrix computation complete for layer {layer}")
                print(f"🔍 DEBUG: Matrix shape: {matrix.shape}, Labels count: {len(labels)}")
                
                # Save results
                print(f"\n💾 Saving results for layer {layer}...")
                print(f"🔍 DEBUG: About to call save_results")
                self.save_results(matrix, labels, output_dir, layer)
                print(f"✓ Results saved for layer {layer}")
                
                # Visualize
                print(f"\n📈 Generating visualization for layer {layer}...")
                print(f"🔍 DEBUG: About to call visualize_matrix")
                self.visualize_matrix(matrix, labels, output_dir, layer)
                print(f"✓ Visualization complete for layer {layer}")
                
                results[f"layer_{layer}"] = {
                    "matrix": matrix,
                    "labels": labels
                }
                
                print(f"\n✅ Layer {layer} processing complete!")
                print(f"🔍 DEBUG: Finished iteration {layer_idx + 1}/{len(layers)}, moving to next layer")
                
            except Exception as e:
                print(f"\n❌ ERROR processing layer {layer}: {e}")
                print(f"   Continuing to next layer...")
                import traceback
                traceback.print_exc()
                print(f"🔍 DEBUG: Exception caught, continuing to next layer")
                continue
        
        print(f"\n🔍 DEBUG: Completed all {len(layers)} layer iterations")
        
        print(f"\n{'='*80}")
        print(f"✅ Analysis complete! Processed {len(results)}/{len(layers)} layers successfully")
        print(f"📁 Results saved to: {output_dir}")
        print("=" * 80)
        
        return results


def load_config(config_file: str) -> Dict:
    """
    Load configuration from JSON file.
    
    Args:
        config_file: Path to JSON config file
    
    Returns:
        Dictionary with configuration parameters
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Validate required fields for new COCO-based format
    required_fields = ["concepts", "samples_per_concept", "annotations_file", "image_dir"]
    for field in required_fields:
        if field not in config:
            raise ValueError(
                f"Config file missing required field: {field}\n"
                f"Required fields: {required_fields}\n"
                f"See SIMILARITY_MATRIX_UPDATES.md for new config format."
            )
    
    # Set defaults for optional fields
    config.setdefault("layers", [31])
    config.setdefault("pooling", "mean")
    config.setdefault("seed", 42)  # Random seed for reproducibility
    # Note: mode, output_dir, temperature, checkpoints now come from CLI args
    
    return config


def main():
    """Main entry point for cross-concept similarity matrix analysis."""
    parser = argparse.ArgumentParser(
        description="Cross-Concept Similarity Matrix Analysis for MoE VLM"
    )
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Path to JSON config file with image-text pairs and parameters"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["stage2", "stage3"],
        help="Analysis mode: 'stage2' (forced routing) or 'stage3' (learned routing). Overrides config file."
    )
    parser.add_argument(
        "--stage2-checkpoint",
        type=str,
        help="Path to Stage 2 checkpoint (optional, defaults to llm_stage2_best.pth from training_config)"
    )
    parser.add_argument(
        "--stage3-checkpoint",
        type=str,
        help="Path to Stage 3 portable checkpoint (required if mode=stage3)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Routing temperature for Stage 3 (lower = more deterministic, default=0.01)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (overrides config file)"
    )
    parser.add_argument(
        "--training-config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training config file for model paths"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # Load config file
    print(f"📄 Loading configuration from: {args.config_file}")
    config = load_config(args.config_file)
    
    # Override with command-line arguments if provided
    if args.mode:
        config["mode"] = args.mode
        print(f"   ⚠️  Overriding mode from command line: {args.mode}")
    
    if args.stage2_checkpoint:
        config["stage2_checkpoint"] = args.stage2_checkpoint
        print(f"   ⚠️  Overriding Stage 2 checkpoint from command line")
    
    if args.stage3_checkpoint:
        config["stage3_checkpoint"] = args.stage3_checkpoint
        print(f"   ⚠️  Overriding Stage 3 checkpoint from command line")
    
    if args.temperature:
        config["temperature"] = args.temperature
        print(f"   ⚠️  Overriding temperature from command line: {args.temperature}")
    
    if args.output_dir:
        config["output_dir"] = args.output_dir
        print(f"   ⚠️  Overriding output directory from command line: {args.output_dir}")
    
    # Validate Stage 3 requirements
    if config.get("mode", "stage2") == "stage3" and "stage3_checkpoint" not in config:
        raise ValueError("--stage3-checkpoint required when mode='stage3'")
    
    # Print configuration
    print(f"\n📋 Configuration:")
    print(f"   Mode: {config.get('mode', 'stage2').upper()}")
    print(f"   Concepts: {config['concepts']}")
    print(f"   Samples per concept: {config['samples_per_concept']}")
    print(f"   Layers: {config['layers']}")
    print(f"   Pooling: {config['pooling']}")
    print(f"   Output directory: {config.get('output_dir', 'results/similarity_matrix/')}")
    print(f"   Annotations file: {config['annotations_file']}")
    print(f"   Image directory: {config['image_dir']}")
    if config.get("mode", "stage2") == "stage2":
        print(f"   Stage 2 checkpoint: {config.get('stage2_checkpoint', 'default (from training_config.yaml)')}")
    elif config.get("mode", "stage2") == "stage3":
        print(f"   Stage 3 checkpoint: {config.get('stage3_checkpoint', 'N/A')}")
        print(f"   Temperature: {config.get('temperature', 0.01)}")
    
    # Initialize analyzer
    analyzer = CrossConceptSimilarityAnalyzer(
        config_path=args.training_config,
        device=args.device,
        mode=config.get("mode", "stage2"),
        stage2_checkpoint=config.get("stage2_checkpoint"),
        stage3_checkpoint=config.get("stage3_checkpoint"),
        temperature=config.get("temperature", 0.01),
    )
    
    # Load models
    analyzer.load_models()
    
    # Run analysis
    results = analyzer.run_analysis(
        concepts=config["concepts"],
        samples_per_concept=config["samples_per_concept"],
        annotations_file=config["annotations_file"],
        image_dir=config["image_dir"],
        layers=config["layers"],
        pooling=config["pooling"],
        output_dir=config.get("output_dir", "results/similarity_matrix/"),
        seed=config.get("seed", 42)
    )


if __name__ == "__main__":
    main()
