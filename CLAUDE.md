# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research code for a Mixture-of-Experts vision-language model (VLM). Custom MoE Mistral-7B with two experts per FFN layer — one for visual tokens, one for text tokens — trained on COCO captions and LLaVA-Instruct data.

## Key Commands

### Setup

```bash
pip install -r requirements.txt
# Edit all YOUR_PATH_HERE placeholders in configs/training_config.yaml before running anything
```

### Create the MoE model (run once)

```bash
python -m models.utils.create_moe_model \
    --base-model /path/to/Mistral-7B-v0.3 \
    --output     /path/to/Mistral-7B-MoE
```

### Training (single GPU for Stage 1, multi-GPU FSDP for stages 2–3)

```bash
export PYTHONPATH="${PWD}:${PYTHONPATH}"
python training_scripts/train_stage_1.py
torchrun --nproc_per_node=4 training_scripts/train_stage_2.py
torchrun --nproc_per_node=4 training_scripts/train_stage_2.5.py
torchrun --nproc_per_node=4 training_scripts/train_stage_3.py
# SLURM: sbatch hpc/training_scripts/train_stage_{1,2,2.5,3}.sbatch
```

### Analysis

```bash
python analysis_scripts/routing_ablation_experiment.py --checkpoint /path/to/stage2_best.pth
python analysis_scripts/cross_concept_similarity_matrix.py --config-file configs/similarity_matrix.json --mode stage2
python analysis_scripts/layer_clustering_analysis.py --config-file configs/clustering_analysis.json
```

## Architecture & Code Map

### Models (`models/`)

- **`moe_layer.py` — `MoELayer`**: Core module. Two routing modes:
  - `'hard'`: reads `self.routing_mask` (set externally per batch); `0=vision expert, 1=text expert`. No gradient through router.
  - `'soft'`: learned `self.gate` (linear) → Gumbel-Softmax + Straight-Through Estimator. Sparse dispatch (one expert per token) but full gradient to gate. Both experts always called on every rank (FSDP requirement); zero-weight dummy pass when a rank has no tokens for an expert.
  - `self._last_router_logits` stored after each forward for metric collection.

- **`custom_mistral.py`**: `MistralMoEConfig` + `MistralMoEForCausalLM` — subclasses of the HuggingFace Mistral classes, swapping FFN with `MoELayer` in every decoder layer. Must be registered with `AutoConfig`/`AutoModelForCausalLM` before loading a saved checkpoint.

- **`vl_connector.py` — `VisionLanguageConnector`**: 2-layer MLP (CLIP 1024-dim → Mistral 4096-dim) with GELU and dropout.

- **`utils/create_moe_model.py`**: Loads Mistral-7B, creates `MistralMoEForCausalLM`, copies FFN weights into both experts in each layer, patches `config.json` with `auto_map`, copies model source files to output dir for `trust_remote_code`.

- **`utils/generation.py` — `CaptionGenerator`**: Inference helper. Prepares vision tokens, tokenises text, concatenates embeddings, calls `model.generate()`. Handles the LLaVA `<image>` prompt format.

### Training Stages

| Stage | Script | What is trainable | Data |
|-------|--------|-------------------|------|
| 1 | `train_stage_1.py` | `VisionLanguageConnector` | COCO captions |
| 2 | `train_stage_2.py` | MoE experts (hard routing) | COCO captions |
| 2.5 | `train_stage_2.5.py` | Gate/router only (soft routing) | COCO captions |
| 3 | `train_stage_3.py` | Self-attn + gate + experts (soft routing) | LLaVA-Instruct-150K |
| Dense | `train_dense.py` | Standard Mistral FFN | COCO captions (baseline) |

**Stage 2 routing mask**: set at runtime, not stored in model weights. The training loop builds:
```python
routing_mask = torch.cat([
    torch.zeros(num_visual_tokens, dtype=torch.long),  # 0 = vision expert
    torch.ones(num_text_tokens, dtype=torch.long),     # 1 = text expert
], dim=1)
for layer in llm.model.layers:
    layer.mlp.routing_mask = routing_mask
```

**Stage 2.5**: `MistralMoEForCausalLM` must be registered before loading. Gates are re-initialised after loading the Stage 2 checkpoint to prevent routing collapse (`std=0.05`). Loads balancing loss + entropy bonus in the training objective.

**Stage 3**: Uses FSDP with `use_orig_params=True`. Caches `embed_tokens_layer = llm.model.embed_tokens` before FSDP wrapping to avoid accessing FSDP internals in the loop. `ExpertUsageTracker` collects 4 metrics (load distribution, routing entropy, confidence, visual vs text split) per validation epoch and saves to `outputs/expert_metrics/`.

### Data (`data/`)

- **`COCO_loader.py`**: Loads COCO captions. Returns `(pixel_values, input_ids, attention_mask)`.
- **`LLaVA_loader.py`**: LLaVA-Instruct-150K. Returns `(pixel_values, input_ids, attention_mask, labels)` where `labels` has question tokens masked to `-100` — only answer tokens contribute to loss.

### Analysis (`analysis_scripts/`)

All analysis scripts load model paths from `configs/training_config.yaml`. JSON configs in `configs/` control per-analysis parameters (layers, concepts, checkpoint paths, output dirs).

Key scripts:
- **`routing_ablation_experiment.py`**: Compares normal vs flipped routing. Expects Stage 2 hard-routing model.
- **`cross_concept_similarity_matrix.py`**: 2N×2N cosine similarity of [image, text] embeddings across layers. Supports `--mode stage2` (hard routing) and `--mode stage3` (soft routing).
- **`cross_modality_purity.py`**: Measures how separable vision/text expert activations are for the same concept across layers.
- **`layer_clustering_analysis.py`**: Clusters per-layer activations, reports silhouette / Davies-Bouldin scores.
- **`karpathy_evaluation/`**: 6-step pipeline (preprocess → embeddings → retrieval → captions → score → visualise). Shared utilities in `karpathy_utils.py`, which reads paths from `training_config.yaml`.
- **`pope_evaluation/`**: POPE hallucination benchmark (3 difficulty levels: random, popular, adversarial).
- **`llava_evaluation/`**: LLaVA-Wild open-ended evaluation.

## Configuration

All model/data paths live in `configs/training_config.yaml` under `paths:`. Every analysis script and training script reads from this file. The only exception is argparse `--default` values in some analysis scripts, which fall back to `YOUR_PATH_HERE` if not overridden via CLI.

## Important Gotchas

- **PYTHONPATH**: Must include repo root (`export PYTHONPATH="${PWD}:${PYTHONPATH}"`) for `trust_remote_code` to find custom model classes.
- **Model registration**: Any script loading a saved MoE checkpoint must call `AutoConfig.register("mistral_moe", MistralMoEConfig)` and `AutoModelForCausalLM.register(MistralMoEConfig, MistralMoEForCausalLM)` first.
- **FSDP + embed_tokens**: The embedding layer is added to `ignored_modules` to avoid FSDP sharding it. Cache the reference before wrapping — never access `llm.module.model.embed_tokens` inside the training loop.
- **Hard routing in Stage 2**: `routing_mask` must be set on every `layer.mlp` before each forward pass. It is not stored in the model and not reset automatically.
- **Soft routing temperature**: Stage 2.5 uses temperature annealing (starts at 2.0, decays to 1.0). Set via `layer.mlp._forward_temperature` before the forward pass.
- **bfloat16 + GradScaler**: Stage 3 uses bfloat16 and explicitly disables `GradScaler` (`enabled=False`) — bfloat16 has better numerical stability than float16 and doesn't need loss scaling.
