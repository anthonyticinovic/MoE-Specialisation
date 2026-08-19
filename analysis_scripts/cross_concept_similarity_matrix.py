"""
Cross-Concept Similarity Matrix Analysis for MoE Vision-Language Models

This script computes 2N×2N similarity matrices comparing N image-text pairs at specified layers.

Stage 2 Mode: Uses expert routing to force images through the vision expert and
text through the text expert.
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

import argparse
import logging
import os

import numpy as np
import torch

from analysis_scripts._lib import (
    compute_cosine_similarity_matrix,
    extract_concept_samples,
    load_analysis_config,
    load_stage2_models,
    load_stage3_models,
)
from analysis_scripts.cross_concept_similarity_plots import SimilarityMatrixPlotsMixin
from analysis_scripts.cross_modality_purity import CrossModalityPurityAnalyzer
from models.utils.common import get_device, setup_logging

logger = logging.getLogger(__name__)


class CrossConceptSimilarityAnalyzer(SimilarityMatrixPlotsMixin):
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
        config_path: str | None = None,
        device: str | None = None,
        mode: str = "stage2",
        stage2_checkpoint: str | None = None,
        stage3_checkpoint: str | None = None,
        temperature: float = 0.01,
    ):
        """
        Initialize analyzer by creating base analyzer for model access.

        Args:
            config_path: Path to training configuration
            device: Device to run on (cuda/cpu)
            mode: "stage2" or "stage3"
            stage2_checkpoint: Path to Stage 2 checkpoint (optional; defaults to
                the best from training_config)
            stage3_checkpoint: Path to Stage 3 portable checkpoint (required if mode="stage3")
            temperature: Routing temperature for Stage 3 (lower = more deterministic)
        """
        logger.info("Initialising Cross-Concept Similarity Analyzer")
        logger.info(f"   Mode: {mode.upper()}")
        logger.info(f"   Device: {device}")

        self.mode = mode
        self.device = device or get_device()
        self.temperature = temperature
        self.stage2_checkpoint = stage2_checkpoint
        self.stage3_checkpoint = stage3_checkpoint

        if mode == "stage3" and stage3_checkpoint is None:
            raise ValueError("stage3_checkpoint path required when mode='stage3'")

        if mode == "stage2":
            # Initialize base analyzer for Stage 2 forced routing
            logger.info("   Using Stage 2 forced expert routing")
            if stage2_checkpoint:
                logger.info(f"   Stage 2 checkpoint: {stage2_checkpoint}")
            self.base_analyzer = CrossModalityPurityAnalyzer(config_path=config_path, device=device)
        elif mode == "stage3":
            # For Stage 3, we'll load models directly
            logger.info(f"   Using Stage 3 learned soft routing (temperature={temperature})")
            logger.info(f"   Checkpoint: {stage3_checkpoint}")
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
        """Load Stage 2 models with forced expert routing via the shared _lib loader."""
        logger.info("\nLoading Stage 2 models...")
        if self.stage2_checkpoint:
            logger.info(f"   Using custom Stage 2 checkpoint: {self.stage2_checkpoint}")
            self.base_analyzer._assign(
                load_stage2_models(
                    self.base_analyzer.config,
                    self.device,
                    stage2_checkpoint=self.stage2_checkpoint,
                )
            )
        else:
            logger.info("   Using default Stage 2 checkpoint from training_config.yaml")
            self.base_analyzer.load_models()

    def _load_stage3_models(self):
        """Load Stage 3 end-to-end model with learned soft routing via _lib."""
        logger.info("\nLoading Stage 3 models...")
        self.base_analyzer = CrossModalityPurityAnalyzer(
            config_path=self.config_path, device=self.device
        )
        self.base_analyzer._assign(
            load_stage3_models(
                self.base_analyzer.config,
                self.device,
                self.stage3_checkpoint,
                self.temperature,
            )
        )
        logger.info(f"   Stage 3 models loaded (soft routing, temperature={self.temperature})")

    def extract_concept_samples(
        self, annotations_file: str, concepts: list[str], samples_per_concept: int, seed: int = 42
    ) -> dict[str, list[dict]]:
        """Balanced concept sampling — see ``_lib.coco_samples``.

        Kept as a method because subclasses and callers reach it through
        ``self``; the implementation is shared so the three copies that used to
        exist cannot drift apart again.
        """
        return extract_concept_samples(annotations_file, concepts, samples_per_concept, seed)

    def _extract_representation(
        self, concept: str, expert: str, layer: int, modality: str, pooling: str = "mean"
    ) -> np.ndarray:
        """
        Extract hidden state representation for a concept at a specific layer.

        For Stage 2: Uses forced routing (wrapper around base analyzer)
        For Stage 3: Uses learned soft routing (natural model behaviour)

        Args:
            concept: Image path (for vision) or text string (for text)
            expert: "vision" or "text" — which expert to route through
                (Stage 2 only; ignored in Stage 3)
            layer: Layer index to extract from (0-31)
            modality: "vision" or "text" - type of input
            pooling: "mean" for mean-pooling (default)

        Returns:
            Numpy array of shape [hidden_dim] representing the pooled hidden state
        """
        if self.mode == "stage2":
            # Delegate to base analyzer's representation extraction logic
            return self.base_analyzer.analyze_representation(
                concept=concept, expert=expert, layer=layer, modality=modality, pooling=pooling
            )
        elif self.mode == "stage3":
            # Extract using learned soft routing
            return self._extract_stage3_representation(
                concept=concept, layer=layer, modality=modality, pooling=pooling
            )

    def _extract_stage3_representation(
        self, concept: str, layer: int, modality: str, pooling: str = "mean"
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
            # REUSE: Input preparation from base analyzer (CLIP + connector for
            # vision, tokenizer for text)
            if modality == "vision":
                visual_soft_tokens = self.base_analyzer._prepare_vision_input(concept)
                inputs_embeds = visual_soft_tokens
                attention_mask = None  # Vision tokens don't need masking

            elif modality == "text":
                text_embeddings = self.base_analyzer._prepare_text_input(concept)
                inputs_embeds = text_embeddings
                # Create attention mask for text (all ones, no padding in single-sample case)
                attention_mask = torch.ones(text_embeddings.shape[:2], device=self.device)
                # BOS is excluded at the pooling step below, not here.

            else:
                raise ValueError(f"Invalid modality: {modality}")

            # DIFFERS: Extract hidden state at specified layer with learned soft routing (no masks)
            hidden_states = self._forward_with_layer_extraction(
                inputs_embeds=inputs_embeds, target_layer=layer, attention_mask=attention_mask
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
        attention_mask: torch.Tensor | None = None,
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
        # Forward pass WITHOUT routing masks (soft routing will use learned behaviour)
        outputs = self.base_analyzer.llm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # outputs.hidden_states is a tuple: (embedding_output, layer_0_output, ..., layer_31_output)
        # target_layer 0 is at index 1, target_layer 31 is at index 32
        hidden_states = outputs.hidden_states[target_layer + 1]

        return hidden_states

    def _compute_cosine_similarity_matrix(self, representations: list[np.ndarray]) -> np.ndarray:
        """Pairwise cosine-similarity matrix (delegates to shared _lib helper)."""
        return compute_cosine_similarity_matrix(representations)

    def compute_cross_concept_matrix(
        self,
        concept_samples: dict[str, list[dict]],
        image_dir: str,
        layer: int = 31,
        pooling: str = "mean",
    ) -> tuple[np.ndarray, list[str]]:
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
        logger.info(f"\nComputing {2 * N}×{2 * N} similarity matrix at layer {layer}")
        logger.info(f"   Pooling strategy: {pooling}")
        logger.info(f"   Number of concepts: {N}")

        representations = []
        labels = []

        # Extract image representations (first N entries) - averaged per concept
        logger.info("\nExtracting image representations through vision expert...")
        for concept, samples in concept_samples.items():
            logger.info(f"   Concept: {concept} ({len(samples)} samples)")

            concept_img_reps = []

            for idx, sample in enumerate(samples):
                image_path = os.path.join(image_dir, sample["image_path"])

                if idx % 10 == 0 and idx > 0:
                    logger.info(f"      Progress: {idx}/{len(samples)} samples")

                try:
                    img_rep = self._extract_representation(
                        concept=image_path,
                        expert="vision",
                        layer=layer,
                        modality="vision",
                        pooling=pooling,
                    )
                    concept_img_reps.append(img_rep)
                except Exception as e:
                    logger.warning(f"       Error processing {image_path}: {e}")
                    continue

            # Average representations across samples
            if len(concept_img_reps) > 0:
                avg_img_rep = np.mean(np.stack(concept_img_reps), axis=0)
                representations.append(avg_img_rep)
                labels.append(f"img:{concept}")
                logger.info(
                    f"       Averaged {len(concept_img_reps)} samples: "
                    f"norm={np.linalg.norm(avg_img_rep):.2f}"
                )
            else:
                logger.warning(f"        No valid samples for {concept}, skipping")

        # Extract text representations (next N entries) - averaged per concept
        logger.info("\nExtracting text representations through text expert...")
        for concept, samples in concept_samples.items():
            logger.info(f"   Concept: {concept} ({len(samples)} samples)")

            concept_txt_reps = []

            for idx, sample in enumerate(samples):
                text = sample["caption"]

                if idx % 10 == 0 and idx > 0:
                    logger.info(f"      Progress: {idx}/{len(samples)} samples")

                try:
                    txt_rep = self._extract_representation(
                        concept=text, expert="text", layer=layer, modality="text", pooling=pooling
                    )
                    concept_txt_reps.append(txt_rep)
                except Exception as e:
                    logger.warning(f"       Error processing text '{text}': {e}")
                    continue

            # Average representations across samples
            if len(concept_txt_reps) > 0:
                avg_txt_rep = np.mean(np.stack(concept_txt_reps), axis=0)
                representations.append(avg_txt_rep)
                labels.append(f"txt:{concept}")
                logger.info(
                    f"       Averaged {len(concept_txt_reps)} samples: "
                    f"norm={np.linalg.norm(avg_txt_rep):.2f}"
                )
            else:
                logger.warning(f"        No valid samples for {concept}, skipping")

        # Compute pairwise similarity matrix
        logger.info(f"\nComputing {2 * N}×{2 * N} cosine similarity matrix...")
        matrix = self._compute_cosine_similarity_matrix(representations)

        logger.info(f"   Matrix computed: shape={matrix.shape}")
        logger.info(f"   Similarity range: [{matrix.min():.3f}, {matrix.max():.3f}]")

        return matrix, labels

    def run_analysis(
        self,
        concepts: list[str],
        samples_per_concept: int,
        annotations_file: str,
        image_dir: str,
        layers: list[int] | None = None,
        pooling: str = "mean",
        output_dir: str = "results/similarity_matrix/",
        seed: int = 42,
    ) -> dict:
        """
        Run complete similarity matrix analysis for specified layers using COCO concept sampling.

        Args:
            concepts: List of concept keywords (e.g., ["cat", "dog", "car"])
            samples_per_concept: Number of samples to extract per concept
            annotations_file: Path to COCO annotations JSON
            image_dir: Base directory for COCO images
            layers: List of layer indices to analyse
            pooling: Pooling strategy
            output_dir: Output directory for results
            seed: Random seed for reproducibility

        Returns:
            Dictionary containing results for each layer
        """
        # A mutable default is shared by every call to this method; a list is
        # only safe as a default while nobody mutates it, and nothing enforces
        # that. The last layer of the 7B model is the historical default.
        if layers is None:
            layers = [31]

        logger.info("=" * 80)
        logger.info("Cross-Concept Similarity Matrix Analysis")
        logger.info("=" * 80)
        logger.info(f"Concepts: {concepts}")
        logger.info(f"Samples per concept: {samples_per_concept}")
        logger.info(f"Layers to analyse: {layers}")
        logger.info(f"Total layers: {len(layers)}")
        logger.info("=" * 80)

        # Extract concept samples once (shared across all layers)
        concept_samples = self.extract_concept_samples(
            annotations_file=annotations_file,
            concepts=concepts,
            samples_per_concept=samples_per_concept,
            seed=seed,
        )

        results = {}
        failed_layers = []

        for layer_idx, layer in enumerate(layers):
            logger.info(f"\n{'=' * 80}")
            logger.info(f"LAYER {layer} ({layer_idx + 1}/{len(layers)})")
            logger.info(f"{'=' * 80}")

            try:
                logger.info(f"Computing the similarity matrix for layer {layer}...")
                matrix, labels = self.compute_cross_concept_matrix(
                    concept_samples=concept_samples,
                    image_dir=image_dir,
                    layer=layer,
                    pooling=pooling,
                )
                logger.info(f"  Matrix {matrix.shape} over {len(labels)} labels")

                self.save_results(matrix, labels, output_dir, layer)
                self.visualize_matrix(matrix, labels, output_dir, layer)
                logger.info(f"  Results and figure written for layer {layer}")

                results[f"layer_{layer}"] = {"matrix": matrix, "labels": labels}

            except Exception:
                # One bad layer must not lose the others, but the run has to say
                # so: `logger.exception` records the traceback through logging
                # rather than printing it to stderr, and the failures are
                # counted so a partial result set is visibly partial.
                logger.exception("Layer %s failed; continuing to the next layer", layer)
                failed_layers.append(layer)
                continue

        if failed_layers:
            logger.error(
                "%d of %d layers failed and are absent from the results: %s",
                len(failed_layers),
                len(layers),
                failed_layers,
            )
        results["failed_layers"] = failed_layers

        logger.info(f"\n{'=' * 80}")
        logger.info(
            f"Analysis complete! Processed {len(results)}/{len(layers)} layers successfully"
        )
        logger.info(f"Results saved to: {output_dir}")
        logger.info("=" * 80)

        return results


def load_config(config_file: str) -> dict:
    """Load the JSON analysis config (COCO-based similarity-matrix format)."""
    return load_analysis_config(
        config_file,
        required_fields=["concepts", "samples_per_concept", "annotations_file", "image_dir"],
        defaults={"layers": [31], "pooling": "mean", "seed": 42},
    )


def main():
    """Main entry point for cross-concept similarity matrix analysis."""
    parser = argparse.ArgumentParser(
        description="Cross-Concept Similarity Matrix Analysis for MoE VLM"
    )
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Path to JSON config file with image-text pairs and parameters",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["stage2", "stage3"],
        help="Analysis mode: 'stage2' (forced routing) or 'stage3' (learned "
        "routing). Overrides the config file.",
    )
    parser.add_argument(
        "--stage2-checkpoint",
        type=str,
        help="Path to Stage 2 checkpoint (optional; defaults to "
        "llm_stage2_best.pth from training_config)",
    )
    parser.add_argument(
        "--stage3-checkpoint",
        type=str,
        help="Path to Stage 3 portable checkpoint (required if mode=stage3)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Routing temperature for Stage 3 (lower = more deterministic, default=0.01)",
    )
    parser.add_argument("--output-dir", type=str, help="Output directory (overrides config file)")
    parser.add_argument(
        "--training-config",
        type=str,
        default=None,
        help="Path to training config (default: $MOE_CONFIG, else configs/training_config.yaml)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (cuda/cpu)",
    )

    args = parser.parse_args()

    # Load config file
    logger.info(f"Loading configuration from: {args.config_file}")
    config = load_config(args.config_file)

    # Override with command-line arguments if provided
    if args.mode:
        config["mode"] = args.mode
        logger.warning(f"    Overriding mode from command line: {args.mode}")

    if args.stage2_checkpoint:
        config["stage2_checkpoint"] = args.stage2_checkpoint
        logger.warning("    Overriding Stage 2 checkpoint from command line")

    if args.stage3_checkpoint:
        config["stage3_checkpoint"] = args.stage3_checkpoint
        logger.warning("    Overriding Stage 3 checkpoint from command line")

    if args.temperature:
        config["temperature"] = args.temperature
        logger.warning(f"    Overriding temperature from command line: {args.temperature}")

    if args.output_dir:
        config["output_dir"] = args.output_dir
        logger.warning(f"    Overriding output directory from command line: {args.output_dir}")

    # Validate Stage 3 requirements
    if config.get("mode", "stage2") == "stage3" and "stage3_checkpoint" not in config:
        raise ValueError("--stage3-checkpoint required when mode='stage3'")

    # Print configuration
    logger.info("\nConfiguration:")
    logger.info(f"   Mode: {config.get('mode', 'stage2').upper()}")
    logger.info(f"   Concepts: {config['concepts']}")
    logger.info(f"   Samples per concept: {config['samples_per_concept']}")
    logger.info(f"   Layers: {config['layers']}")
    logger.info(f"   Pooling: {config['pooling']}")
    logger.info(f"   Output directory: {config.get('output_dir', 'results/similarity_matrix/')}")
    logger.info(f"   Annotations file: {config['annotations_file']}")
    logger.info(f"   Image directory: {config['image_dir']}")
    if config.get("mode", "stage2") == "stage2":
        logger.info(
            "   Stage 2 checkpoint: "
            f"{config.get('stage2_checkpoint', 'default (from training_config.yaml)')}"
        )
    elif config.get("mode", "stage2") == "stage3":
        logger.info(f"   Stage 3 checkpoint: {config.get('stage3_checkpoint', 'N/A')}")
        logger.info(f"   Temperature: {config.get('temperature', 0.01)}")

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
        seed=config.get("seed", 42),
    )

    # A run that lost layers must not report success. Exiting non-zero is what
    # makes a partial result set visible to whatever invoked the script.
    return 1 if results["failed_layers"] else 0


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
