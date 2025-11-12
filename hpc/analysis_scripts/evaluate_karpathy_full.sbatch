#!/bin/bash
#SBATCH --job-name=karpathy_caption
#SBATCH --output=/home/aticinovic/out_slurm/karpathy_caption_%j.out
#SBATCH --error=/home/aticinovic/err_slurm/karpathy_caption_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8

# Karpathy COCO Captioning Evaluation Pipeline
# Runs caption generation evaluation for Stage 2 and Stage 3 models
# Includes: preprocessing, caption generation, and evaluation

set -e  # Exit on error

# Load modules
module load GCCcore/11.3.0
module load Python/3.11.3
module load CUDA/11.7.0

# Activate virtual environment
source ~/pytorch_latest_venv/bin/activate

# Set paths
SCRIPT_DIR="/home/aticinovic/MoE-Specialisation/analysis_scripts/karpathy_evaluation"
RESULTS_DIR="/home/aticinovic/MoE-Specialisation/results/karpathy_evaluation"
DATA_DIR="/data/gpfs/projects/COMP90055/aticinovic/datasets/coco"

# Model checkpoints
STAGE2_CKPT="/data/gpfs/projects/COMP90055/aticinovic/outputs/stage2_checkpoints/llm_stage2_best.pth"
STAGE3_CKPT="/data/gpfs/projects/COMP90055/aticinovic/outputs/stage3_checkpoints/llm_stage3_best_portable.pth"

# Karpathy split JSON
KARPATHY_JSON="${DATA_DIR}/karpathy_split/dataset_coco.json"

echo "=============================================================================="
echo "KARPATHY COCO CAPTIONING EVALUATION PIPELINE"
echo "=============================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=============================================================================="

# Create results directories
mkdir -p ${RESULTS_DIR}/captioning

# =============================================================================
# STEP 1: Preprocess Karpathy Split
# =============================================================================
echo ""
echo "=============================================================================="
echo "STEP 1: Preprocessing Karpathy Split"
echo "=============================================================================="
echo "Start: $(date)"

python ${SCRIPT_DIR}/01_preprocess_karpathy.py \
    --karpathy_json ${KARPATHY_JSON} \
    --output_dir ${RESULTS_DIR}

echo "Completed: $(date)"

# =============================================================================
# STEP 2: Generate Captions (Stage 2)
# =============================================================================
echo ""
echo "=============================================================================="
echo "STEP 2: Generating Captions - Stage 2"
echo "=============================================================================="
echo "Start: $(date)"

python ${SCRIPT_DIR}/04_generate_captions.py \
    --checkpoint_path ${STAGE2_CKPT} \
    --stage_name stage2 \
    --images_json ${RESULTS_DIR}/karpathy_test_images.json \
    --image_base_dir ${DATA_DIR} \
    --output_dir ${RESULTS_DIR}/captioning \
    --num_beams 5 \
    --max_length 20 \
    --batch_size 16 \
    --device cuda

echo "Completed: $(date)"

# =============================================================================
# STEP 3: Generate Captions (Stage 3)
# =============================================================================
echo ""
echo "=============================================================================="
echo "STEP 3: Generating Captions - Stage 3"
echo "=============================================================================="
echo "Start: $(date)"

python ${SCRIPT_DIR}/04_generate_captions.py \
    --checkpoint_path ${STAGE3_CKPT} \
    --stage_name stage3 \
    --images_json ${RESULTS_DIR}/karpathy_test_images.json \
    --image_base_dir ${DATA_DIR} \
    --output_dir ${RESULTS_DIR}/captioning \
    --num_beams 5 \
    --max_length 20 \
    --batch_size 16 \
    --device cuda

echo "Completed: $(date)"

# =============================================================================
# STEP 4: Evaluate Captioning Metrics
# =============================================================================
echo ""
echo "=============================================================================="
echo "STEP 4: Evaluating Captioning Metrics"
echo "=============================================================================="
echo "Start: $(date)"

python ${SCRIPT_DIR}/05_evaluate_captioning.py \
    --references_json ${RESULTS_DIR}/karpathy_test_references.json \
    --stage2_captions ${RESULTS_DIR}/captioning/stage2_captions.json \
    --stage3_captions ${RESULTS_DIR}/captioning/stage3_captions.json \
    --output_dir ${RESULTS_DIR}/captioning

echo "Completed: $(date)"

# =============================================================================
# STEP 5: Generate Visualizations
# =============================================================================
echo ""
echo "=============================================================================="
echo "STEP 5: Generating Captioning Visualizations"
echo "=============================================================================="
echo "Start: $(date)"

# Create a simple visualization script for captioning only
python -c "
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load metrics
with open('${RESULTS_DIR}/captioning/captioning_metrics.json', 'r') as f:
    metrics = json.load(f)

s2 = metrics['stage2']
s3 = metrics['stage3']

# Create bar chart
fig, ax = plt.subplots(figsize=(12, 6))
metric_names = ['Bleu_1', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE']
s2_vals = [s2.get(m, 0) for m in metric_names]
s3_vals = [s3.get(m, 0) for m in metric_names]

x = np.arange(len(metric_names))
width = 0.35

ax.bar(x - width/2, s2_vals, width, label='Stage 2', alpha=0.8, color='#3498db')
ax.bar(x + width/2, s3_vals, width, label='Stage 3', alpha=0.8, color='#e74c3c')

ax.set_xlabel('Metric', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Captioning Metrics Comparison: Stage 2 vs Stage 3', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metric_names, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('${RESULTS_DIR}/captioning_comparison.png', dpi=300, bbox_inches='tight')
print('✅ Saved: ${RESULTS_DIR}/captioning_comparison.png')
"

echo "Completed: $(date)"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================================================="
echo "PIPELINE COMPLETE"
echo "=============================================================================="
echo "End time: $(date)"
echo ""
echo "Results saved to: ${RESULTS_DIR}/"
echo ""
echo "Key files:"
echo "  • Stage 2 captions:       ${RESULTS_DIR}/captioning/stage2_captions.json"
echo "  • Stage 3 captions:       ${RESULTS_DIR}/captioning/stage3_captions.json"
echo "  • Captioning metrics:     ${RESULTS_DIR}/captioning/captioning_metrics.json"
echo "  • Comparison table:       ${RESULTS_DIR}/captioning/captioning_comparison.txt"
echo "  • Comparison plot:        ${RESULTS_DIR}/captioning_comparison.png"
echo "=============================================================================="
