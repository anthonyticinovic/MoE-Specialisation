#!/bin/bash
#SBATCH --job-name=stage3_alignment
#SBATCH --partition=gpu-a100-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=00:10:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Stage 3: Layer-by-Layer Cross-Modal Alignment Analysis
# This script computes cosine similarity between vision and text representations
# at each of the 33 layers (embedding + 32 transformer layers) using learned routing.

echo "========================================"
echo "Stage 3: Layer-by-Layer Alignment"
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "========================================"

# Load modules
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
module load CUDA

# Activate virtual environment
source ~/pytorch_latest_venv/bin/activate

# Navigate to project directory
cd ~/MoE-Specialisation

# Add project to PYTHONPATH
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Print environment info
echo ""
echo "Environment Info:"
echo "  - Python: $(which python)"
echo "  - CUDA: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
echo "  - GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1)"
echo ""

# Run Stage 3 alignment analysis
python analysis_scripts/cross_modality_purity.py \
    --stage3-alignment configs/stage3_alignment.json \
    --device cuda

echo ""
echo "========================================"
echo "Job completed at: $(date)"
echo "========================================"
