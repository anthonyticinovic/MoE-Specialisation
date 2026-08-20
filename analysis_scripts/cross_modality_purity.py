"""
Cross-Modality Purity Analysis for MoE Vision-Language Models

This script analyses how "pure" expert representations are across modalities
by comparing vision and text expert activations for the same concept.

Usage:
    python analysis_scripts/cross_modality_purity.py --concepts red blue --layers 0 8 16 24 31
    python analysis_scripts/cross_modality_purity.py --concepts circle --top-k 20
"""

import argparse
import json
import logging
import os

import numpy as np

from analysis_scripts import cross_modality_purity_plots as cmp_plots
from analysis_scripts._lib import (
    SyntheticImageGenerator,
    extract_concept_samples,
    load_stage2_models,
    load_stage3_models,
    load_training_config,
)
from analysis_scripts.cross_modality_extraction import RepresentationExtractionMixin
from analysis_scripts.cross_modality_metrics import PurityMetricsMixin, debug_layers
from models.utils.common import get_device, setup_logging

logger = logging.getLogger(__name__)


class CrossModalityPurityAnalyzer(RepresentationExtractionMixin, PurityMetricsMixin):
    """
    Analyses cross-modality purity of expert representations.

    Key Methods:
        - analyze_vocab(): Top-k vocabulary predictions
        - analyze_representation(): Hidden state extraction
        - compute_cosine_similarity(): Representation similarity
        - compute_probability_ratio(): Cross-modal concept probability
    """

    def __init__(
        self,
        config_path: str | None = None,
        device: str | None = None,
    ):
        self.device = device or get_device()
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

        logger.info(f"Initialised analyzer on device: {self.device}")

    def _load_config(self, config_path: str | None) -> dict:
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
        logger.info("Loading models...")
        self._assign(load_stage2_models(self.config, self.device))
        logger.info("All models loaded successfully")

    def load_stage3_models(self, checkpoint_path: str, temperature: float = 0.01):
        """Load Stage 3 models with learned soft routing.

        Args:
            checkpoint_path: Path to Stage 3 checkpoint (full or portable version)
            temperature: Softmax temperature for routing (default: 0.01 for near-deterministic)
        """
        logger.info("Loading Stage 3 models with learned routing...")
        self._assign(load_stage3_models(self.config, self.device, checkpoint_path, temperature))

    def extract_concept_samples(
        self, annotations_file: str, concepts: list[str], samples_per_concept: int, seed: int = 42
    ) -> dict[str, list[dict]]:
        """Balanced concept sampling — see ``_lib.coco_samples``.

        Kept as a method because subclasses and callers reach it through
        ``self``; the implementation is shared so the three copies that used to
        exist cannot drift apart again.
        """
        return extract_concept_samples(annotations_file, concepts, samples_per_concept, seed)

    def run_comprehensive_analysis(
        self,
        concepts: list[str],
        layers: list[int],
        output_dir: str = "results/cross_modality_purity",
    ) -> dict:
        """
        Run comprehensive cross-modality purity analysis.

        Args:
            concepts: List of concepts to analyse
            layers: List of layer indices
            output_dir: Directory to save results

        Returns:
            Dictionary containing all analysis results
        """
        logger.info(
            f"\nRunning comprehensive analysis on {len(concepts)} concepts across {len(layers)} "
            f"layers..."
        )

        os.makedirs(output_dir, exist_ok=True)

        # Setup debug logging to file if in debug mode
        debug_log_path = None
        self._debug_layers = debug_layers(layers)
        if hasattr(self, "_debug_mode") and self._debug_mode:
            debug_log_path = os.path.join(output_dir, "debug_token_analysis.log")
            logger.info(f"Debug output will be saved to: {debug_log_path}")

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
            logger.info(f"\nAnalyzing concept: '{concept}'")

            # Add debug summary for first concept only
            if hasattr(self, "_debug_mode") and self._debug_mode and concept_idx == 0:
                logger.info(f"   Debug info will be shown for layers: {self._debug_layers}")

            results["cosine_similarity"][concept] = {}
            results["euclidean_distance"][concept] = {}
            results["cosine_similarity_mean_pooled"][concept] = {}
            results["euclidean_distance_mean_pooled"][concept] = {}
            results["vocab_predictions"][concept] = {}

            for layer in layers:
                logger.info(f"  - Layer {layer}...")

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

                    logger.info(
                        f"(cos_cls={cosine_sim:.3f}, cos_mp={cosine_sim_mp:.3f}, "
                        f"euc_cls={euclidean_dist:.2f}, euc_mp={euclidean_dist_mp:.2f})"
                    )

                except Exception as e:
                    logger.error(f"Error: {e}")
                    continue

        # Save results
        results_path = os.path.join(output_dir, "purity_analysis_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to {results_path}")

        # Generate visualisations
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
        logger.info("\n" + "=" * 80)
        logger.info("Stage 3: Layer-by-Layer Cross-Modal Alignment Analysis")
        logger.info("=" * 80)

        # Load config
        logger.info(f"\nLoading config from {config_path}")
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

        logger.info(f"  Checkpoint: {checkpoint_path}")
        logger.info(f"  Temperature: {temperature}")
        logger.info(f"  Pooling: {pooling}")
        logger.info(f"  Routing: {routing_mode}")
        logger.info(f"  Concepts: {concepts}")
        logger.info(f"  Samples per concept: {samples_per_concept}")

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
        logger.info(
            f"\nComputing alignment curves (averaging {samples_per_concept} samples per concept)..."
        )
        alignment_curves = {}
        failed_concepts: list[str] = []

        for idx, concept in enumerate(concepts, 1):
            samples = concept_samples[concept]
            if len(samples) == 0:
                logger.warning(
                    f"  [{idx}/{len(concepts)}]  Skipping '{concept}' (no samples found)"
                )
                continue

            logger.info(
                f"  [{idx}/{len(concepts)}] Processing '{concept}' ({len(samples)} samples)..."
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

                # Report the embedding, first and last layer. The last index is
                # taken from the curve rather than hardcoded to 31: a model with
                # any other depth used to raise IndexError here.
                last_layer = max(layer for layer in avg_curve if layer >= 0)
                logger.info(
                    "  %s: emb=%.3f, L0=%.3f, L%d=%.3f",
                    concept,
                    avg_curve[-1],
                    avg_curve[0],
                    last_layer,
                    avg_curve[last_layer],
                )

            except Exception:
                # logger.exception records the traceback through logging rather
                # than printing it to stderr, so it lands in the same stream as
                # everything else the run reports.
                logger.exception("Concept %r failed; continuing to the next", concept)
                failed_concepts.append(concept)
                continue

        # A concept that failed is absent from alignment_curves, so the result
        # file has to say how many — otherwise a half-finished run and a
        # deliberately-narrow one produce indistinguishable output.
        if failed_concepts:
            logger.error(
                "%d of %d concepts failed and are absent from the results: %s",
                len(failed_concepts),
                len(concepts),
                failed_concepts,
            )

        results = {
            "config": config,
            "alignment_curves": alignment_curves,
            "failed_concepts": failed_concepts,
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
        logger.info(f"\nResults saved to {results_path}")

        # Generate visualisation
        logger.info("\nGenerating alignment curve plot...")
        cmp_plots.plot_alignment_curves(alignment_curves, output_dir, title_suffix="Stage 3")

        logger.info("\n" + "=" * 80)
        logger.info("Stage 3 alignment analysis complete!")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("=" * 80)

        return results

    def _visualize_results(self, results: dict, output_dir: str):
        """Generate visualisation plots from analysis results."""
        logger.info("\nGenerating visualizations...")

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
            logger.info(
                "\nGenerating purity matrix and divergence analysis (2 concepts detected)..."
            )

            # Each diagnostic below is optional — one failing must not lose the
            # others — but a failure is an error, not a status update. These
            # used to be logged with logger.info, so a run in which every
            # diagnostic failed still printed "complete" and exited zero.
            failures: list[str] = []

            def attempt(name: str, work):
                """Run one optional diagnostic; record it if it raises."""
                logger.info("Running %s...", name)
                try:
                    work()
                except Exception:
                    logger.exception("%s failed", name)
                    failures.append(name)

            labels_cls = None

            def clip_connector_mean():
                clip_matrix, connector_matrix, labels = self.compute_clip_connector_comparison(
                    concepts
                )
                cmp_plots.plot_clip_connector_comparison(
                    clip_matrix, connector_matrix, labels, output_dir, pooling="mean"
                )

            def clip_connector_cls():
                nonlocal labels_cls
                clip_matrix_cls, connector_matrix_cls, labels_cls = (
                    self.compute_clip_connector_comparison_cls(concepts)
                )
                cmp_plots.plot_clip_connector_comparison(
                    clip_matrix_cls, connector_matrix_cls, labels_cls, output_dir, pooling="cls"
                )

            def token_variance():
                # labels_cls comes from the CLS diagnostic above. When that one
                # failed, this used to raise NameError and be swallowed as a
                # second, unrelated-looking failure.
                if labels_cls is None:
                    raise RuntimeError(
                        "the CLS CLIP-vs-connector diagnostic did not run, so its "
                        "labels are unavailable"
                    )
                cmp_plots.plot_token_variance(
                    self.analyze_token_variance(concepts), labels_cls, output_dir
                )

            def position_specific():
                cmp_plots.plot_position_specific_similarity(
                    self.analyze_position_specific_similarity(concepts), output_dir
                )

            attempt("CLIP vs connector (mean-pooled)", clip_connector_mean)
            attempt("CLIP vs connector (CLS token)", clip_connector_cls)
            attempt("token-level variance", token_variance)
            attempt("position-specific similarity", position_specific)

            # Purity matrices at the layers the caller asked for. The set used
            # to be hardcoded to the 7B model's [-1, 0, 15, 31].
            requested = [layer for layer in layers if layer >= -1]
            target_layers = [requested[0], requested[len(requested) // 2], requested[-1]]
            target_layers = sorted(set(target_layers))
            matrices = {}
            for layer in target_layers:

                def purity(layer=layer):
                    matrices[layer] = self.compute_purity_matrix(concepts, layer, pooling="mean")

                attempt(f"purity matrix at layer {layer}", purity)

            if matrices:
                cmp_plots.plot_purity_matrices(matrices, target_layers, output_dir)

            if failures:
                logger.error(
                    "%d of the optional diagnostics failed and produced no output: %s",
                    len(failures),
                    failures,
                )

        logger.info("Visualisation complete!")


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
        help="Analyse the embeddings plus every transformer layer the loaded "
        "model has (overrides --layers)",
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
        default=None,
        help="Path to training config (default: $MOE_CONFIG, else configs/training_config.yaml)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
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
        help="Run Stage 3 layer-by-layer alignment analysis (provide a config path, "
        "e.g. configs/stage3_alignment.json)",
    )

    args = parser.parse_args()

    # Initialize analyzer
    logger.info("=" * 80)

    # Check if running Stage 3 alignment analysis
    if args.stage3_alignment:
        logger.info("Stage 3: Layer-by-Layer Alignment Analysis")
        logger.info("=" * 80)

        analyzer = CrossModalityPurityAnalyzer(config_path=args.config, device=args.device)
        analyzer.run_stage3_alignment_analysis(config_path=args.stage3_alignment)

        return  # Exit after Stage 3 analysis

    # Otherwise run standard Stage 2 purity analysis
    logger.info("Cross-Modality Purity Analysis")
    if args.debug:
        logger.info("DEBUG MODE ENABLED")
    logger.info("=" * 80)

    analyzer = CrossModalityPurityAnalyzer(config_path=args.config, device=args.device)

    if args.debug:
        analyzer._debug_mode = True

    # Load models
    analyzer.load_models()

    # --all-layers is resolved here, after the model is loaded, because it is a
    # claim about the model rather than a default: it used to expand to
    # range(32) and so requested layers that do not exist on anything but the
    # 7B. --layers keeps its 7B-shaped default, which is a default and stays
    # one.
    if args.all_layers:
        depth = analyzer.llm.config.num_hidden_layers
        layers = [-1] + list(range(depth))
        logger.info(
            f"Using --all-layers: analysing the embeddings plus all {depth} transformer "
            f"layers of the loaded model ({len(layers)} in total)"
        )
    else:
        layers = args.layers

    # Run comprehensive analysis
    analyzer.run_comprehensive_analysis(
        concepts=args.concepts, layers=layers, output_dir=args.output_dir
    )

    logger.info("\n" + "=" * 80)
    logger.info("Analysis complete!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    setup_logging()
    main()
