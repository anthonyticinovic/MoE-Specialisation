#!/bin/bash
# Dense control — standard Mistral FFN, no MoE. The baseline the paper compares
# against, so it must run the same data and schedule as Stage 2.
#
# Adapt the #SBATCH header to your cluster; see hpc/README.md. Submit from the
# repository root so the log paths resolve:
#   mkdir -p logs/slurm && sbatch hpc/training_scripts/train_dense.sbatch
#SBATCH --job-name=moe_dense
#SBATCH -p gpu-l40s
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

source "${MOE_PROJECT_DIR:-$HOME/MoE-Specialisation}/hpc/cluster_env.sh"
moe_setup_environment

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export TORCH_NCCL_BLOCKING_WAIT=1  # NCCL_TIMEOUT is ignored from PyTorch 2.5

echo "Starting dense control baseline..."
srun torchrun --nproc_per_node=4 training_scripts/train_dense.py

echo "Job finished."
