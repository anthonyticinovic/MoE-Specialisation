# Analysis Scripts

Interpretability and evaluation tooling for the MoE vision-language model. This
directory is the navigation hub; each evaluation sub-pipeline has its own
detailed README (linked below).

## Prerequisites

Run everything **from the repository root** with the repo on `PYTHONPATH`:

```bash
export PYTHONPATH="${PWD}:${PYTHONPATH}"
```

Fill in every `YOUR_PATH_HERE` placeholder in `configs/training_config.yaml`
first — all scripts read model/data paths from there via the shared loader,
which fails fast with a clear message if a placeholder is left unfilled.

To run against a different config, set `MOE_CONFIG` rather than editing
anything; the resolution order is the explicit `--training-config` argument,
then `$MOE_CONFIG`, then the repo default. This is the same mechanism the CPU
demo uses to drive the training scripts, and the reason
`routing_ablation_experiment.py` is now the demo's final step. Device follows
the same pattern: leave `--device` unset and `get_device()` picks CUDA when it
is available and CPU otherwise, and the model dtype follows the device
(bfloat16 on GPU, float32 on CPU).

MoE model registration with HuggingFace `AutoModel` is handled automatically by
the shared model loader (`_lib.model_loading`) — you no longer need to register
it per script.

## Shared library: `_lib/`

Common code that used to be copy-pasted across scripts now lives in one place:

| Module | Purpose |
|--------|---------|
| `_lib/config.py` | `load_training_config()` / `get_paths()` (validated YAML) and `load_analysis_config()` (JSON configs with required-field + default handling) |
| `_lib/model_loading.py` | `load_stage2_models()` / `load_stage3_models()` — the single, behaviour-preserving Stage-2 (hard) and Stage-3 (soft routing) checkpoint loaders |
| `_lib/representations.py` | `compute_cosine_similarity_matrix()`, `majority_vote_expert()` |
| `_lib/viz.py` | `set_publication_rcparams()`, `similarity_heatmap()` |
| `_lib/io.py` | image preprocessing, mean pooling, JSON I/O, banners |
| `_lib/synthetic_images.py` | `SyntheticImageGenerator` for concept stimuli |

Import via `from analysis_scripts._lib import ...`.

## Top-level scripts

Large analysers are split across sibling modules, composed back together by
inheritance so the class each script exposes is unchanged:

| Analyser | Siblings |
|---|---|
| `cross_modality_purity.py` | `cross_modality_extraction.py` (hidden states out of the model), `cross_modality_metrics.py` (metrics over them), `cross_modality_purity_plots.py` |
| `cross_concept_similarity_matrix.py` | `cross_concept_similarity_plots.py` (heatmaps, coherence score, JSON writer) |
| `layer_clustering_analysis.py` | `layer_clustering_plots.py` |
| `attention_routing_analysis.py` | `attention_routing_plots.py` |
| `plot_expert_metrics.py` | `expert_metrics_plots.py` (per-layer), `expert_metrics_evolution_plots.py` (across-epoch + report) |

**Add a method to the sibling that owns the concern, not to the analyser** —
`layer_clustering_analysis` and `attention_routing_analysis` subclass
`CrossModalityPurityAnalyzer`, so the composition order is load-bearing.

| Script | What it measures | Example |
|--------|------------------|---------|
| `cross_modality_purity.py` | Vision/text expert representation purity per layer (base analyzer; subclassed by others) | `python analysis_scripts/cross_modality_purity.py --concepts red blue --layers 0 8 16 24 31` |
| `cross_concept_similarity_matrix.py` | 2N×2N cosine-similarity of [image, text] across layers (Stage 2 / Stage 3) | `python analysis_scripts/cross_concept_similarity_matrix.py --config-file configs/similarity_matrix.json --mode stage2` |
| `cross_modality_purity.py` (stage3 mode) | Layer-by-layer cross-modal alignment curves | see `--help` |
| `compositional_case_study.py` | Stage 2 vs Stage 3 representations for compositional stimuli | `python analysis_scripts/compositional_case_study.py --config-file configs/compositional_case_study.json` |
| `attention_routing_analysis.py` | How attention + expert routing co-evolve across layers | `python analysis_scripts/attention_routing_analysis.py --config <json>` |
| `layer_clustering_analysis.py` | Clustering of per-layer activations (silhouette / Davies-Bouldin, t-SNE/PaCMAP) | `python analysis_scripts/layer_clustering_analysis.py --config configs/clustering_analysis.json` |
| `routing_ablation_experiment.py` | Normal vs flipped expert routing loss (Stage 2) | `python analysis_scripts/routing_ablation_experiment.py --checkpoint <pth>` |
| `plot_expert_metrics.py` | Plots expert utilisation metrics from Stage 3 training | `python analysis_scripts/plot_expert_metrics.py --metrics_dir <dir>` |
| `create_stage_comparison.py` | Side-by-side Stage 2 vs Stage 3 similarity heatmaps | `python analysis_scripts/create_stage_comparison.py --stage2-dir <d> --stage3-dir <d> --output-dir <d>` |

Sibling modules are never run directly — they are imported by the analyser that
owns them.

## Evaluation pipelines

Each is a self-contained, ordered pipeline with its own README:

- **[karpathy_evaluation/](karpathy_evaluation/README.md)** — 6-step Karpathy
  COCO retrieval + captioning pipeline (preprocess → embeddings → retrieval →
  captions → score → visualise). Shared helpers in `karpathy_utils.py` (now a
  thin layer that re-exports `_lib`).
- **[pope_evaluation/](pope_evaluation/README.md)** — POPE object-hallucination
  benchmark (random / popular / adversarial). Optional Stage-3 priming via
  `--use-priming` on `02_generate_pope_answers.py`. Shared
  extractors/metrics in `pope_utils.py`.
- **[llava_evaluation/](llava_evaluation/README.md)** — LLaVA-Wild open-ended
  conversational evaluation.

## Notes

- **Logging**: these scripts use `logging`, like the rest of the repo — every
  entry point calls `setup_logging()` and every module holds a
  `logging.getLogger(__name__)`. Two message styles are in use, and the split is
  deliberate:
  - **f-strings for reporting** (491 calls). The tables here rely on alignment
    and percentage format specs (`{value:<15.1f}`, `{share:.1%}`) that have no
    `%`-formatting equivalent, and the lazy-formatting argument — avoid work for
    a message that is never emitted — does not apply to a single-run script
    logging at INFO.
  - **Lazy `%s` on error paths** (16 calls), matching the core. These are
    `logger.error` and `logger.exception` calls, where the message may carry a
    large object and where handing the arguments to `logging` keeps the
    traceback and the message in one record.
- **File size**: nothing here exceeds 800 lines, and
  `tests/test_analysis_lib.py::test_no_source_file_is_oversized` fails anything
  that does. The largest is `cross_concept_similarity_matrix.py` at 742.
- **Karpathy COCO path**: `01_preprocess_karpathy.py`, `02_extract_embeddings.py`
  and `04_generate_captions.py` take the COCO/Karpathy path as a **required**
  CLI argument (no placeholder default); see the Karpathy README. Everything
  else derives paths from `configs/training_config.yaml`.

## Known limitations (kept honest)

Real and current, checked against the code on 19 Aug 2026. Two entries that used
to be here — `torch.cuda._total_entropy` in `train_stage_2.5.py` and the
inconsistent FSDP `device_id` across stages — are fixed and have been removed;
a third, about three files exceeding the 800-line guideline, is obsolete since
those files were split and a test now enforces the limit.

- **Two analysis entry points of twenty are executed by anything.** Every module
  here imports, and `_lib/model_loading.py` has behavioural tests, but only
  `routing_ablation_experiment.py` and `plot_expert_metrics.py` are actually run
  — by the CPU demo, on every push. The other eighteen are checked for shape (no
  `print`, no hardcoded config path, no CUDA default, no bare `strict=False`, no
  failure reported at INFO) and not for behaviour. Treat their outputs as
  un-regression-tested.
- **Duplication across the analysers.** `extract_concept_samples` is
  implemented three times (in `cross_concept_similarity_matrix.py`,
  `cross_modality_purity.py` and `layer_clustering_analysis.py`) and
  `compute_metrics` twice (`pope_utils.py` and, separately,
  `compare_priming_strategies.py`). Consolidating them is safe only once the
  point above is fixed.
- **Long functions.** Nineteen functions here exceed 120 lines, the largest
  being `generate_captions` (273) and `generate_pope_answers` (243). The
  training scripts have a test enforcing that limit; these do not, for the same
  reason.
- **Duplication across `train_stage_*.py`** — the five stage scripts total
  ~2,500 lines against ~1,000 in `_lib`. The shared runtime, FSDP wrapper,
  backbones, dataloaders, checkpoint guard and loss are extracted; the forward
  passes deliberately are not, because the `no_grad`/`autocast` boundaries
  differ per stage in ways a CPU run cannot verify. See
  [`docs/design.md`](../docs/design.md) §5 for the full argument.
