# Research Code: Expert Collapse and Compositional Failure in Simple Multimodal MoE

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Paper](https://img.shields.io/badge/paper-AAAI%202026%20Workshop-b31b1b.svg)](https://proceedings.mlr.press/v332/)

Research code for [*Expert Collapse and Compositional Failure in Simple Multimodal MoE*](https://raw.githubusercontent.com/mlresearch/v332/main/assets/ticinovic26a/ticinovic26a.pdf)
(Ticinovic, 2026), published at the **AAAI 2026 Workshop on Bias in Multimodal
AI: Representation, Risk, and Repair** ([PMLR v332](https://proceedings.mlr.press/v332/)).
See the paper for the full method and results.

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
  ≥4× A100/H100-class GPUs. The SLURM scripts in `hpc/` target an H100 cluster
  and will need their `--gres`/module lines adapted.

### Setup

Using [uv](https://docs.astral.sh/uv/) (recommended for local development):

```bash
git clone https://github.com/apticinovic/MoE-Specialisation.git
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

### Stage 0 - Create the MoE model

```bash
python -m models.utils.create_moe_model \
    --base-model /path/to/Mistral-7B-v0.3 \
    --output     /path/to/Mistral-7B-MoE
```

### Stage 1 - Train the vision connector (1 GPU)

```bash
# Locally or via SLURM:
export PYTHONPATH="${PWD}:${PYTHONPATH}"
python training_scripts/train_stage_1.py
# SLURM: sbatch hpc/training_scripts/train_stage_1.sbatch
```

### Stages 2, 2.5, 3 - Multi-GPU training (FSDP)

```bash
# 4-GPU example:
export PYTHONPATH="${PWD}:${PYTHONPATH}"
torchrun --nproc_per_node=4 training_scripts/train_stage_2.py
torchrun --nproc_per_node=4 training_scripts/train_stage_2.5.py
torchrun --nproc_per_node=4 training_scripts/train_stage_3.py

# SLURM wrappers:
sbatch hpc/training_scripts/train_stage_2.sbatch
sbatch hpc/training_scripts/train_stage_2.5.sbatch
sbatch hpc/training_scripts/train_stage_3.sbatch
```

> Stages 2–3 require at least 4× A100/H100 GPUs. The SLURM scripts target a Slurm cluster with H100 nodes; adapt `--gres` and module loads for your environment.

## Analysis Scripts

All analysis scripts are in `analysis_scripts/`. They require a trained checkpoint and the paths configured in `configs/training_config.yaml` (or the relevant JSON config in `configs/`).

### Expert routing & specialisation

```bash
# Routing ablation: compare normal vs. flipped routing to verify specialisation
python analysis_scripts/routing_ablation_experiment.py \
    --checkpoint /path/to/stage2_best.pth \
    --data       /path/to/coco

# Expert utilisation metrics across epochs (reads JSON files from training)
python analysis_scripts/plot_expert_metrics.py \
    --metrics_dir /path/to/outputs/expert_metrics
```

### Concept-level analysis

```bash
# Cross-concept similarity matrix (2N×2N image-text similarity at each layer)
python analysis_scripts/cross_concept_similarity_matrix.py \
    --config-file configs/similarity_matrix.json \
    --mode stage2   # or stage3

# Cross-modality purity (how separable are expert representations per concept?)
python analysis_scripts/cross_modality_purity.py \
    --concepts dog cat car bus \
    --layers 0 8 16 24 31

# Layer-wise clustering of expert activations
python analysis_scripts/layer_clustering_analysis.py \
    --config configs/clustering_analysis.json

# Compositional case study (colour-object binding)
python analysis_scripts/compositional_case_study.py \
    --config-file configs/compositional_case_study.json

# Stage 2 vs Stage 3 similarity matrix comparison plot
python analysis_scripts/create_stage_comparison.py \
    --stage2-dir results/similarity_matrix/stage2 \
    --stage3-dir results/similarity_matrix/stage3
```

### Benchmark evaluation

#### POPE (object hallucination)

```bash
# Generates pope_{random,popular,adversarial}.json into the output dir.
python analysis_scripts/pope_evaluation/01_generate_pope_questions.py \
    --annotations_file /path/to/coco/annotations/instances_val2017.json \
    --output_dir       results/pope_evaluation

# Run once per difficulty (see pope_evaluation/README.md for the full loop).
python analysis_scripts/pope_evaluation/02_generate_pope_answers.py \
    --questions_file  results/pope_evaluation/pope_random.json \
    --image_dir       /path/to/coco/val2017 \
    --checkpoint_path /path/to/checkpoint.pth \
    --output_dir      results/pope_evaluation

python analysis_scripts/pope_evaluation/03_evaluate_pope.py \
    --stage2_dir results/pope_evaluation \
    --output_dir results/pope_evaluation
```

#### Karpathy COCO split (retrieval + captioning)

```bash
# Preprocess Karpathy split JSON
python analysis_scripts/karpathy_evaluation/01_preprocess_karpathy.py \
    --karpathy_json /path/to/dataset_coco.json

# Extract embeddings for retrieval
python analysis_scripts/karpathy_evaluation/02_extract_embeddings.py \
    --image_base_dir /path/to/coco \
    --checkpoint_path /path/to/checkpoint.pth

# Evaluate retrieval (R@1, R@5, R@10)
python analysis_scripts/karpathy_evaluation/03_evaluate_retrieval.py

# Generate captions
python analysis_scripts/karpathy_evaluation/04_generate_captions.py \
    --image_base_dir /path/to/coco \
    --checkpoint_path /path/to/checkpoint.pth

# Score captions (CIDEr, BLEU, METEOR, ROUGE)
python analysis_scripts/karpathy_evaluation/05_evaluate_captioning.py

# Visualise results
python analysis_scripts/karpathy_evaluation/06_visualize_results.py
```

#### LLaVA-Wild (instruction following)

```bash
python analysis_scripts/llava_evaluation/01_llava_wild_eval.py \
    --checkpoint /path/to/checkpoint.pth

python analysis_scripts/llava_evaluation/02_compare_results.py \
    --stage2 results/llava_wild/stage2 \
    --stage3 results/llava_wild/stage3
```

## Repository Structure

```
MoE-Specialisation/
├── models/
│   ├── moe_layer.py          # MoELayer: hard and soft routing
│   ├── custom_mistral.py     # MistralMoEConfig, MistralMoEForCausalLM
│   ├── vl_connector.py       # VisionLanguageConnector (CLIP→LLM projection)
│   └── utils/
│       ├── create_moe_model.py   # Build MoE model from Mistral-7B
│       ├── common.py             # Shared helpers (config, seed, logging, registration)
│       └── generation.py         # CaptionGenerator inference helper
├── data/
│   ├── COCO_loader.py        # COCO captions dataset
│   └── LLaVA_loader.py       # LLaVA-Instruct-150K dataset
├── training_scripts/         # One script per training stage
├── analysis_scripts/         # Expert analysis and benchmark evaluation
│   ├── karpathy_evaluation/  # COCO Karpathy split pipeline
│   ├── pope_evaluation/      # POPE hallucination benchmark
│   └── llava_evaluation/     # LLaVA-Wild evaluation
├── tests/                    # CPU-only pytest suite + behavioural oracle
├── configs/
│   ├── training_config.yaml  # All paths + hyperparameters (edit this first)
│   └── *.json                # Per-analysis configs
└── hpc/
    ├── training_scripts/     # SLURM job scripts for training
    └── model_scripts/        # SLURM job script for model creation
```

## Reproduce the paper

The full pipeline, in order, with the artifacts each stage produces:

| Step | Command | Produces |
|------|---------|----------|
| 0 | `python -m models.utils.create_moe_model --base-model <Mistral-7B> --output <MoE>` | MoE model dir (`trust_remote_code`) |
| 1 | `python training_scripts/train_stage_1.py` | `vision_connector_stage1_best.pth` |
| 2 | `torchrun --nproc_per_node=4 training_scripts/train_stage_2.py` | `stage2_checkpoints/llm_stage2_best.pth` |
| 2.5 | `torchrun --nproc_per_node=4 training_scripts/train_stage_2.5.py` | `stage2_5_checkpoints/` (learned router) |
| 3 | `torchrun --nproc_per_node=4 training_scripts/train_stage_3.py` | `stage3_checkpoints/` + `outputs/expert_metrics/` |

Then run the analysis scripts (see below) against the resulting checkpoints.
The `uv.lock` pins the exact dependency set; exact paper figures used the
`transformers` 4.x line (the dependency is capped `<5`).

## Development

```bash
uv sync --group dev          # runtime + dev tooling
uv run pre-commit install

uv run ruff check models/ data/ tests/   # lint the maintained core
uv run ruff format --check .             # formatting (whole repo)
uv run mypy                              # type-check models/ + data/
uv run pytest -q                         # CPU-only suite (~1s)
```

ruff and mypy are strict on the maintained core (`models/`, `data/`, `tests/`);
the research scripts (`training_scripts/`, `analysis_scripts/`) are held to
formatting only, since they reproduce the published paper and are changed
conservatively. `tests/test_training_dry_run.py` is a behavioural oracle: it
asserts a tiny synthetic model produces bit-identical loss/grad-norm against a
recorded baseline — if those numbers move, a refactor changed training numerics.

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
