# MoE Vision-Language Model

Research code for *Emergent Expert Specialisation in Mixture-of-Experts Vision-Language Models* (Ticinovic, 2026). The paper is included as `ticinovic26a.pdf`.

## Overview

This project investigates whether expert specialisation can emerge in a Mixture-of-Experts (MoE) language model when it is trained as a vision-language model (VLM). The core idea: replace every FFN layer in Mistral-7B with two experts — one for visual tokens, one for text tokens — and train the model to caption images.

**Key findings:**
- Explicit modality-based routing (hard routing) successfully specialises experts, but routing collapses without enforcement.
- Cross-modal concept representations are more jointly structured in the specialised expert latent space than expected — concepts from different modalities share geometric neighbourhood, suggesting the experts do not produce fully disjoint representations.
- A learned soft router (Stage 2.5) can recover meaningful routing after expert specialisation, but is sensitive to initialisation.

## Architecture

```
Image ──► CLIP ViT-L/14 ──► VisionLanguageConnector ──► visual soft tokens ─┐
                              (2-layer MLP, 1024→4096)                        │
                                                                              ├──► [visual | text] embeddings
Text  ──────────────────────────────────────────────── text embeddings ───────┘
                                                                              │
                                                                              ▼
                                                                   Mistral-7B + MoE layers
                                                                   (2 experts per FFN layer)
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

- **Hard routing** (Stage 2): a binary mask derived from token position forces visual tokens to Expert 0 and text tokens to Expert 1. No gate is needed — the modality is known.
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

### Setup

```bash
git clone <repo>
cd MoE-Specialisation
pip install -r requirements.txt

# Edit all YOUR_PATH_HERE placeholders:
vim configs/training_config.yaml
```

### Stage 0 — Create the MoE model

```bash
python -m models.utils.create_moe_model \
    --base-model /path/to/Mistral-7B-v0.3 \
    --output     /path/to/Mistral-7B-MoE
```

### Stage 1 — Train the vision connector (1 GPU)

```bash
# Locally or via SLURM:
export PYTHONPATH="${PWD}:${PYTHONPATH}"
python training_scripts/train_stage_1.py
# SLURM: sbatch hpc/training_scripts/train_stage_1.sbatch
```

### Stages 2, 2.5, 3 — Multi-GPU training (FSDP)

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
    --data-path  /path/to/coco

# Expert utilisation metrics across epochs (reads JSON files from training)
python analysis_scripts/plot_expert_metrics.py \
    --metrics-dir /path/to/outputs/expert_metrics
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
    --config-file configs/clustering_analysis.json

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
python analysis_scripts/pope_evaluation/01_generate_pope_questions.py \
    --annotations /path/to/coco/annotations/instances_val2017.json \
    --output-dir  results/pope

python analysis_scripts/pope_evaluation/02_generate_pope_answers.py \
    --questions-dir results/pope \
    --checkpoint    /path/to/checkpoint.pth

python analysis_scripts/pope_evaluation/03_evaluate_pope.py \
    --results-dir results/pope
```

#### Karpathy COCO split (retrieval + captioning)

```bash
# Preprocess Karpathy split JSON
python analysis_scripts/karpathy_evaluation/01_preprocess_karpathy.py \
    --karpathy-json /path/to/dataset_coco.json

# Extract embeddings for retrieval
python analysis_scripts/karpathy_evaluation/02_extract_embeddings.py \
    --coco-dir /path/to/coco

# Evaluate retrieval (R@1, R@5, R@10)
python analysis_scripts/karpathy_evaluation/03_evaluate_retrieval.py

# Generate captions
python analysis_scripts/karpathy_evaluation/04_generate_captions.py \
    --coco-dir /path/to/coco

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
    --stage2-results results/llava_wild/stage2 \
    --stage3-results results/llava_wild/stage3
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
│       ├── create_n_experts.py   # FFN→MoE surgical replacement utility
│       └── generation.py         # CaptionGenerator inference helper
├── data/
│   ├── COCO_loader.py        # COCO captions dataset
│   └── LLaVA_loader.py       # LLaVA-Instruct-150K dataset
├── training_scripts/         # One script per training stage
├── analysis_scripts/         # Expert analysis and benchmark evaluation
│   ├── karpathy_evaluation/  # COCO Karpathy split pipeline
│   ├── pope_evaluation/      # POPE hallucination benchmark
│   └── llava_evaluation/     # LLaVA-Wild evaluation
├── configs/
│   ├── training_config.yaml  # All paths + hyperparameters (edit this first)
│   └── *.json                # Per-analysis configs
└── hpc/
    ├── training_scripts/     # SLURM job scripts for training
    └── model_scripts/        # SLURM job script for model creation
```

## Citation

```bibtex
@inproceedings{ticinovic2026moe,
  title     = {Emergent Expert Specialisation in Mixture-of-Experts Vision-Language Models},
  author    = {Ticinovic, Anthony},
  year      = {2026}
}
```
