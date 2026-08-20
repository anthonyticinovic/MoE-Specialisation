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
  `logging.getLogger(__name__)`. No `print`, and a test enforces that.
  Two message styles are in use, split by what the record carries:
  - **f-strings for reporting**, which is the large majority. These scripts
    exist to print tables, and the tables rely on alignment and percentage
    format specs (`{value:<15.1f}`, `{share:.1%}`) that have no `%`-formatting
    equivalent. The usual argument for lazy formatting — do not build a string
    for a message that is never emitted — carries little weight in a single-run
    script logging at INFO.
  - **Lazy `%` where the argument should not be rendered unless the record is.**
    Two cases: `logger.error`/`logger.exception` handing an exception object to
    `logging` so the message and the traceback stay in one record, and the
    `logger.debug` calls in `layer_clustering_analysis` that dump router tensor
    shapes, which a normal INFO-level run must not pay for. `_lib` also uses
    `%` throughout, as the core does.

  Do not read this as a rule the linter enforces — it does not, and a handful of
  `_lib` reporting calls use `%` simply because that is the core's house style.
- **File size**: nothing here exceeds 800 lines, and
  `tests/test_analysis_lib.py::test_no_source_file_is_oversized` fails anything
  that does.
- **Karpathy COCO path**: `01_preprocess_karpathy.py`, `02_extract_embeddings.py`
  and `04_generate_captions.py` take the COCO/Karpathy path as a **required**
  CLI argument (no placeholder default); see the Karpathy README. Everything
  else derives paths from `configs/training_config.yaml`.

## Known limitations (kept honest)

Real and current. Each one is either enforced by a test or checkable against the
code in a minute; entries come off this list when the code changes, not when the
list is rewritten.

- **Two analysis entry points of twenty are executed by anything.** Every module
  here imports, and `_lib/model_loading.py` has behavioural tests, but only
  `routing_ablation_experiment.py` and `plot_expert_metrics.py` are actually run
  — by the CPU demo, on every push. The other eighteen are checked for shape (no
  `print`, no hardcoded config path, no CUDA default, no bare `strict=False`, no
  failure reported at INFO) and not for behaviour. Treat their outputs as
  un-regression-tested.
- **Long functions.** A number of functions here exceed the repo's 120-line
  limit — the worst are the generation loops in `04_generate_captions.py` and
  `02_generate_pope_answers.py`, at over 200 lines each. Every one is in code
  the demo cannot reach, so a refactor of it cannot be verified by anything.
  The exact set is
  `tests/test_analysis_lib.py::KNOWN_OVERSIZED_FUNCTIONS` — read it there
  rather than trusting a count in prose. A ratchet enforces it in both
  directions: a new offender fails immediately, and splitting one *requires*
  removing it from the list, so the backlog can never look larger or smaller
  than it is. The functions that *were* demo-reachable have been split, each
  verified by byte-comparing the demo's output before and after.
- **A known scoring defect in `pope_utils.extract_yes_no_answer`.** The
  affirmative phrase list is scanned before the descriptive-pattern list, so a
  caption of the form "The image features a …" scores **yes** even when the
  queried object is absent. Stage 3 collapsed into producing exactly those
  captions, so this plausibly inflates its POPE yes-rate. Pinned by a test and
  left uncorrected: changing it would change published numbers.
- **Duplication across `train_stage_*.py`** — the five stage scripts total
  ~2,500 lines against ~1,000 in `_lib`. The shared runtime, FSDP wrapper,
  backbones, dataloaders, checkpoint guard and loss are extracted; the forward
  passes deliberately are not, because the `no_grad`/`autocast` boundaries
  differ per stage in ways a CPU run cannot verify. See
  [`docs/design.md`](../docs/design.md) §5 for the full argument.
