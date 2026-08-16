#!/bin/bash
# Stage 1 — vision connector, single GPU.
#
# Adapt the #SBATCH header to your cluster; see hpc/README.md. Submit from the
# repository root so the log paths resolve:
#   mkdir -p logs/slurm && sbatch hpc/training_scripts/train_stage_1.sbatch
#SBATCH --job-name=moe_stage1
#SBATCH -p gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=54:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

source "${MOE_PROJECT_DIR:-$HOME/MoE-Specialisation}/hpc/cluster_env.sh"
moe_setup_environment

echo "Starting Stage 1 (vision connector)..."
python -u training_scripts/train_stage_1.py

echo "Job finished."
