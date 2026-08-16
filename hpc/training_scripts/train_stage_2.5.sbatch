#!/bin/bash
# Stage 2.5 — learned router only, soft routing, 2 GPUs with FSDP.
#
# Adapt the #SBATCH header to your cluster; see hpc/README.md. Submit from the
# repository root so the log paths resolve:
#   mkdir -p logs/slurm && sbatch hpc/training_scripts/train_stage_2.5.sbatch
#SBATCH --job-name=moe_stage2_5
#SBATCH -p gpu-h100
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=23:30:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

source "${MOE_PROJECT_DIR:-$HOME/MoE-Specialisation}/hpc/cluster_env.sh"
moe_setup_environment

export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8
export TORCH_NCCL_BLOCKING_WAIT=1  # NCCL_TIMEOUT is ignored from PyTorch 2.5

echo "Starting Stage 2.5 (router training)..."
srun torchrun --nproc_per_node=2 training_scripts/train_stage_2.5.py

echo "Job finished."
