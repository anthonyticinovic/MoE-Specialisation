# Research Code: Expert Collapse and Compositional Failure in Simple Multimodal MoE

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![CI](https://github.com/anthonyticinovic/MoE-Specialisation/actions/workflows/ci.yml/badge.svg)](https://github.com/anthonyticinovic/MoE-Specialisation/actions/workflows/ci.yml)
[![Paper](https://img.shields.io/badge/paper-AAAI%202026%20Workshop-b31b1b.svg)](https://proceedings.mlr.press/v332/)

Research code for [*Expert Collapse and Compositional Failure in Simple Multimodal MoE*](https://raw.githubusercontent.com/mlresearch/v332/main/assets/ticinovic26a/ticinovic26a.pdf)
(Ticinovic, 2026), published at the **AAAI 2026 Workshop on Bias in Multimodal
AI: Representation, Risk, and Repair** ([PMLR v332](https://proceedings.mlr.press/v332/)).
See the paper for the full method and results.

## Run it in 30 seconds (no GPU, no data, no downloads)

The full pipeline — Stage 0 → 1 → 2 → 2.5 → 3, the dense control, and one
analysis script — runs on a laptop CPU against generated fixtures:

```bash
uv sync --group dev
make demo
```

About 20 seconds. It writes 12 checkpoints, routing metrics, four figures, the
routing-ablation results and a one-page `demo_output/demo_report.md`.

The report leads with **14 executable invariants** — properties that must hold
for the pipeline to be correct, not just for it to exit zero. Among them: the
two experts must start bit-identical (Stage 0 copies the base FFN into both) and
must have diverged after Stage 2; hard routing must dispatch each token to
exactly the masked expert; Stage 2 must leave every non-expert weight untouched;
Stage 2.5 must move the gates and *only* the gates; routing entropy must stay
within `[0, ln N]`; and swapping the two experts at inference must raise the
loss on **every** sample, which is the paper's specialisation claim reduced to
something a machine can check. A failing invariant fails the run, so `make
check` catches a refactor that runs cleanly but behaves wrongly.

This runs the **real training scripts**, not a reimplementation: the same
`train_stage_*.py` files used on the H100 cluster, pointed at a miniature config
via `MOE_CONFIG`. What differs is only scale and device — a 2-layer/64-hidden
Mistral, a 4-patch CLIP tower, 24 synthetic images, and CPU fallbacks for FSDP,
8-bit loading and FlashAttention. Nothing is stubbed. The final step drives
`analysis_scripts/routing_ablation_experiment.py` against the Stage 2
checkpoint the run just produced, so the analysis code is covered by the same
mechanism.

It is a smoke test, not a result: a randomly-initialised 2-layer model cannot
caption anything, and the routing metrics sit near an even split. The point is
that the pipeline, the checkpoint formats and the routing instrumentation are
verifiably intact — and that a reader can confirm that themselves without a
cluster.

`make check` runs the lint, the tests and the demo together; so does CI, on
every push.

For *why* the code is shaped this way — two experts and not eight, a fixed mask
before a learned gate, what FSDP forced, and what I would do differently — see
[`docs/design.md`](docs/design.md).

## Overview

This project investigates whether intentional modal (vision/text) expert specialisation can emerge in a Mixture-of-Experts (MoE) language model. This creates a testbed that allows us to investigate the process of expert specialisation, particularly across modalities. We replace every FFN layer in Mistral-7B with two experts, one for visual tokens, one for text tokens, and train the model to caption images.

**Key findings:**
- Explicit modality-based routing (hard routing) successfully specialises experts, but routing collapses without enforcement.
- Cross-modal concept representations are more jointly structured in the specialised expert latent space than expected. Concepts from different modalities share geometric neighbourhood, suggesting the experts do not produce fully disjoint representations.
- A learned soft router (Stage 2.5) can recover meaningful routing after expert specialisation, but only at select layers.

### Captioning performance (COCO Karpathy test split)

Captioning quality is a *diagnostic* here, not the objective. This is an
interpretability study of routing, not a push for SOTA captioning. The numbers
are reported in full precisely because they make the routing story concrete:

| Model | Data | B-4 | METEOR | ROUGE-L | CIDEr |
|---|---|--:|--:|--:|--:|
| LLaVA-v1.5-7B (full FT, reference) | COCO | **38.2** | 23.5 | **57.3** | **111.4** |
| Stage 2 — hard routing (ours) | COCO | 31.9 | **33.3** | 55.4 | 76.2 |
| Stage 3 — soft routing (ours) | COCO → LLaVA-Ins | 4.2 | 12.2 | 29.9 | 8.1 |

Stage 2 is competitive with a fully fine-tuned LLaVA reference (and higher
METEOR) despite only training two FFN experts under a fixed routing mask. The
sharp Stage 3 drop is **the studied phenomenon, not an unexplained failure**.
Soft routing collapses after stage 3 training, which is what the interpretability analysis
in the paper dissects. Baseline from Bucciarelli et al. (2024).

## Architecture

```
Image ──► CLIP ViT-L/14 ──► VisionLanguageConnector ──► visual soft tokens ───┐
                              (2-layer MLP, 1024→4096)                        │
                                                                              ├──► [visual | text] embeddings
Text  ──────────────────────────────────────────────── text embeddings ───────┘
                                                                              │
                                                                              ▼
                                                                   Mistral-7B + MoE layers
                                                                              │
                                                              ┌───────────────┴──────────────────┐
                                                              │                                  │
                                                        Expert 0                          Expert 1
                                                      (vision tokens)                  (text tokens)
                                                              │                                  │
                                                              └───────────────┬──────────────────┘
                                                                              │
                                                                         next-token logits
```

The custom `MoELayer` (`models/moe_layer.py`) supports two routing modes:

- **Hard routing** (Stage 2): a binary mask derived from token position forces visual tokens to Expert 0 and text tokens to Expert 1. No gate is needed since the modality is known.
- **Soft routing** (Stages 2.5 & 3): a learned linear gate produces per-token routing probabilities. Training uses Gumbel-Softmax with a Straight-Through Estimator so the gate receives gradients while dispatch remains sparse.

## Training Pipeline

All stages read paths from `configs/training_config.yaml`. Fill in the `YOUR_PATH_HERE` placeholders before running.

| Stage | Script | What trains | Notes |
|-------|--------|-------------|-------|
| **0** | `models/utils/create_moe_model.py` | — | Creates the MoE model from Mistral-7B |
| **1** | `training_scripts/train_stage_1.py` | VisionLanguageConnector only | CLIP + LLM frozen; 1 GPU |
| **2** | `training_scripts/train_stage_2.py` | MoE experts (hard routing) | Router frozen; 4× H100 via FSDP |
| **2.5** | `training_scripts/train_stage_2.5.py` | Router/gate only | Experts frozen; introduces soft routing |
| **3** | `training_scripts/train_stage_3.py` | Self-attn + router + experts | End-to-end; LLaVA-Instruct data |
| Dense | `training_scripts/train_dense.py` | Standard Mistral FFN | Control baseline |

**Why a "Stage 2.5"?** Stage 2 specialises the experts using a fixed,
position-derived hard routing mask — there is no learned router. Stage 3 needs
a *learned* soft router. Jumping straight from a fixed mask to end-to-end soft
routing collapses routing onto one expert, so Stage 2.5 is a dedicated bridging
stage that trains only the gate (experts frozen) until soft routing is stable.

### Requirements & data

This is refactored research code, not a product. Before anything will run:

- **No trained checkpoints are shipped.** Reproducing any result means running
  the full pipeline (Stages 0→3) yourself. The analysis and evaluation scripts
  all require a checkpoint you have trained.
- **No datasets or sample images are shipped.** You must download COCO 2017
  (train/val + caption & instance annotations) and LLaVA-Instruct-150K, plus
  local copies of Mistral-7B-v0.3 and CLIP ViT-L/14, then point
  `configs/training_config.yaml` at them.
- **Hardware.** Stage 1 trains on a single GPU; Stages 2–3 use FSDP and need
  ≥4× A100/H100-class GPUs. The SLURM scripts in [`hpc/`](hpc/README.md) target
  an H100 cluster: paths and modules are set in `hpc/cluster_env.sh` (or
  overridden by environment variable), while the `#SBATCH` headers must be
  edited per cluster.
- **No published metrics yet.** The Stage 3 metric files behind the paper's
  routing figures are not in the repository. The regeneration path is in place
  — see [`paper_metrics/`](paper_metrics/README.md) — but until the JSON is
  added, `make figures` has nothing to plot.

### Setup

Using [uv](https://docs.astral.sh/uv/) (recommended for local development):

```bash
git clone https://github.com/anthonyticinovic/MoE-Specialisation.git
cd MoE-Specialisation
uv sync                       # creates .venv from the pinned uv.lock
# prefix commands with `uv run`, e.g. uv run python -m models.utils.create_moe_model ...
```

On HPC/SLURM (uv may be unavailable on compute nodes):

```bash
pip install -r requirements.txt   # generated export of uv.lock
```

Then edit all `YOUR_PATH_HERE` placeholders in `configs/training_config.yaml`.
`load_config()` validates these on startup and fails fast with a clear message
if any are left unfilled.

### Running the stages

| Step | Command | Produces |
|------|---------|----------|
| 0 | `python -m models.utils.create_moe_model --base-model <Mistral-7B> --output <MoE>` | MoE model dir (`trust_remote_code`) |
| 1 | `python training_scripts/train_stage_1.py` | `vision_connector_stage1_best.pth` |
| 2 | `torchrun --nproc_per_node=4 training_scripts/train_stage_2.py` | `stage2_checkpoints/llm_stage2_best.pth` |
| 2.5 | `torchrun --nproc_per_node=4 training_scripts/train_stage_2.5.py` | `stage2_5_checkpoints/` (learned router) |
| 3 | `torchrun --nproc_per_node=4 training_scripts/train_stage_3.py` | `stage3_checkpoints/` + `outputs/expert_metrics/` |
| dense | `torchrun --nproc_per_node=4 training_scripts/train_dense.py` | `dense_checkpoints/` (control baseline) |

`export PYTHONPATH="${PWD}:${PYTHONPATH}"` first, so `trust_remote_code` can
find the custom model classes. Each stage has a SLURM wrapper —
`sbatch hpc/training_scripts/train_stage_<n>.sbatch`.

> Stages 2–3 require at least 4× A100/H100 GPUs. Before submitting, set your
> checkout and virtualenv paths in `hpc/cluster_env.sh` (or export
> `MOE_PROJECT_DIR` / `MOE_VENV`) and adapt the `#SBATCH` headers. See
> [`hpc/README.md`](hpc/README.md).

Stage 3's `expert_metrics/` is what [`paper_metrics/`](paper_metrics/README.md)
expects, so a completed run makes `make figures` work without a GPU thereafter.
Then run the analyses — see [`docs/running-the-analyses.md`](docs/running-the-analyses.md).

## Analysis Scripts

All analysis scripts live in `analysis_scripts/` and need a checkpoint you have
trained. They resolve paths through `MOE_CONFIG`, falling back to
`configs/training_config.yaml`, and pick CUDA or CPU automatically — the same
mechanism the CPU demo uses to drive them.

| Script | Question it answers |
|---|---|
| `routing_ablation_experiment.py` | Does swapping the two experts cost loss? (the specialisation check `make demo` runs) |
| `plot_expert_metrics.py` | How did expert load, entropy and confidence move across Stage 3 epochs? |
| `cross_concept_similarity_matrix.py` | How similar are image and text representations of the same concept, layer by layer? |
| `cross_modality_purity.py` | How separable are the two experts' representations for one concept? |
| `layer_clustering_analysis.py` | Do per-layer activations cluster by modality or by concept? |
| `compositional_case_study.py` | Colour–object binding: does the model compose attributes? |
| `pope_evaluation/` | Object hallucination (random / popular / adversarial) |
| `karpathy_evaluation/` | COCO Karpathy retrieval (R@k) and captioning (CIDEr, BLEU, METEOR, ROUGE) |
| `llava_evaluation/` | LLaVA-Wild open-ended instruction following |

**The commands for all of these are in
[`docs/running-the-analyses.md`](docs/running-the-analyses.md)**; each evaluation
sub-pipeline additionally has its own README with the full loop and expected
outputs. Output goes to `results/`, which is git-ignored.

## Repository Structure

```
MoE-Specialisation/
├── models/
│   ├── moe_layer.py          # MoELayer: hard and soft routing
│   ├── custom_mistral.py     # MistralMoEConfig, MistralMoEForCausalLM
│   ├── vl_connector.py       # VisionLanguageConnector (CLIP→LLM projection)
│   └── utils/
│       ├── create_moe_model.py   # Build MoE model from Mistral-7B
│       ├── checkpoints.py        # Cross-stage checkpoint reading, guarded
│       └── common.py             # Shared helpers (config, device, seed, logging)
├── data/
│   ├── COCO_loader.py        # COCO captions dataset
│   └── LLaVA_loader.py       # LLaVA-Instruct-150K dataset
├── training_scripts/         # One script per training stage
│   └── _lib/                 # Shared runtime, pipeline and metric code + Stage 3 setup
├── analysis_scripts/         # Expert analysis and benchmark evaluation
│   ├── _lib/                 # Shared model loading, I/O and plotting helpers
│   ├── karpathy_evaluation/  # COCO Karpathy split pipeline
│   ├── pope_evaluation/      # POPE hallucination benchmark
│   └── llava_evaluation/     # LLaVA-Wild evaluation
├── demo/                     # CPU end-to-end demo: fixtures, runner, invariants
├── docs/
│   ├── design.md             # Why the architecture and pipeline are shaped this way
│   └── running-the-analyses.md  # Every analysis/benchmark command
├── tests/                    # CPU-only pytest suite + behavioural oracle
├── paper_metrics/            # Committed Stage 3 metrics for `make figures`
├── configs/
│   ├── training_config.yaml  # All paths + hyperparameters (edit this first)
│   └── *.json                # Per-analysis configs
└── hpc/
    ├── cluster_env.sh        # Cluster paths and modules, sourced by every job
    ├── training_scripts/     # SLURM job scripts for training
    └── model_scripts/        # SLURM job script for model creation
```

## Dependencies

`uv.lock` pins the exact dependency set the results were produced with.
`transformers` is floored at 4.56 (where `from_pretrained(dtype=...)` was
introduced) and capped below 5, which changes Mistral internals.

## Development

```bash
uv sync --group dev          # runtime + dev tooling
uv run pre-commit install

make lint     # ruff (correctness repo-wide, style on the core) + mypy
make format   # apply ruff formatting
make test     # CPU-only pytest suite (~9s)
make demo     # the whole pipeline on CPU against synthetic fixtures
make check    # lint + test + demo
```

CI runs exactly this list on every push: the lint, the type check, the test
suite, and the demo. The demo is included deliberately — the unit suite can stay
green while the pipeline itself is broken.

ruff and mypy are strict on the maintained core (`models/`, `data/`, `tests/`);
the research scripts (`training_scripts/`, `analysis_scripts/`) are held to
formatting plus the correctness rules that catch undefined names
(`F821`/`F811`/`F822`), unstrict `zip()` (`B905`) and bare `except:` (`E722`).
A missing import once left two training scripts unrunnable on `main` for months,
and the narrower lint scope was why nothing noticed. The other two are on the
list for the same reason: an unstrict `zip()` truncates to the shorter sequence
in silence, and a bare `except:` swallows `KeyboardInterrupt` alongside whatever
it meant to catch.

The suite covers four levels, deliberately:

| Level | Where | What it protects |
|---|---|---|
| Numerics | `tests/test_training_dry_run.py` | A tiny synthetic model must produce bit-identical loss and grad-norm against a recorded baseline. If those move, a refactor changed training numerics. |
| Behaviour | `tests/test_training_steps.py`, `tests/test_analysis_model_loading.py` | Each stage runs a real epoch and must change **exactly** the parameters it claims to train, with gradients reaching all of them, and must resume from its own checkpoint. The analysis loaders are then driven against the checkpoints those stages produced. |
| Structure | `tests/test_training_scripts_structure.py`, `tests/test_analysis_lib.py` | Every script stays inert on import, reuses `_lib` rather than re-implementing FSDP, and never loads weights with a bare `strict=False`. No analysis script may hardcode the config path or default to CUDA — both would put it back out of the demo's reach. No source file may exceed 800 lines. Also checks the SLURM scripts point at files that exist. |
| End-to-end | `make demo` | 14 executable invariants over a full CPU pipeline run, including the routing ablation. A partial run (`--stages 0 1`) skips the checks whose stages did not run rather than failing them. |

## Citation

```bibtex
@inproceedings{ticinovic2026expert,
  title     = {Expert Collapse and Compositional Failure in Simple Multimodal MoE},
  author    = {Ticinovic, Anthony and Han, Caren},
  booktitle = {Proceedings of the AAAI 2026 Workshop on Bias in Multimodal AI: Representation, Risk, and Repair},
  series    = {Proceedings of Machine Learning Research},
  volume    = {332},
  year      = {2026},
  month     = jan,
  address   = {Singapore},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v332/}
}
```
