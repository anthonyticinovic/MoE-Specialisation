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

Large analyzers keep their analysis logic in the named file and their plotting
in a sibling `*_plots.py` / `*_metrics_plots.py` module (imported automatically).

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

`*_plots.py` siblings are not run directly — they are imported by their owning
script.

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

- **File size**: `cross_modality_purity.py`, `cross_concept_similarity_matrix.py`
  and `expert_metrics_plots.py` remain above the 800-line guideline. They are
  cohesive analyzers / a pure plotting module; further splitting was judged to
  risk silently changing un-runnable research outputs for little benefit. Model
  loading, similarity and plotting were extracted where it was safe to do so.
- **Karpathy COCO path**: `01_preprocess_karpathy.py`, `02_extract_embeddings.py`
  and `04_generate_captions.py` take the COCO/Karpathy path as a **required**
  CLI argument (no placeholder default); see the Karpathy README. Everything
  else derives paths from `configs/training_config.yaml`.

## Known limitations (kept honest)

These are real and known. They were left unchanged on purpose: the
training scripts reproduce the published paper's numerics and are only
covered by the Stage-1 dry-run oracle, so behavioural edits to Stages 2–3
were out of scope for the presentation cleanup.

- **`torch.cuda._total_entropy` in `train_stage_2.5.py`** — the entropy bonus
  is accumulated by attaching state to the `torch.cuda` module. It works for a
  single-process run but is an anti-pattern; a local accumulator would be
  cleaner.
- **Inconsistent FSDP `device_id`** — `train_stage_2.5.py` passes
  `f"cuda:{rank}"`, `train_stage_3.py` an `int`, `train_dense.py`
  `torch.cuda.current_device()`. All work on the cluster they were run on but
  the inconsistency is untidy.
- **Duplication across `train_stage_*.py`** — ~60–70% of each stage script is
  shared boilerplate (FSDP setup, checkpoint sync, loops). This was a
  deliberate trade-off: keeping each stage a standalone, independently
  reproducible script was prioritised over DRY for a research codebase.
