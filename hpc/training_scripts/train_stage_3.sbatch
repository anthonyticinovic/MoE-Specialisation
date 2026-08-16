#!/bin/bash
# Stage 3 — end-to-end soft routing on LLaVA-Instruct, 4 GPUs with FSDP.
#
# Adapt the #SBATCH header to your cluster; see hpc/README.md. Submit from the
# repository root so the log paths resolve:
#   mkdir -p logs/slurm && sbatch hpc/training_scripts/train_stage_3.sbatch
#SBATCH --job-name=moe_stage3
#SBATCH -p gpu-l40s
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=95:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

source "${MOE_PROJECT_DIR:-$HOME/MoE-Specialisation}/hpc/cluster_env.sh"
moe_setup_environment

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# PyTorch 2.5+ reads TORCH_NCCL_*; the older NCCL_TIMEOUT is ignored.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600  # checkpoint gathers are slow

# Reduce allocator fragmentation on the long LLaVA sequences.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# A fixed rendezvous port collides with other jobs on the same node.
RANDOM_PORT=$((29500 + RANDOM % 35500))
echo "Using rendezvous port ${RANDOM_PORT}"

echo "Starting Stage 3 (end-to-end, soft routing)..."
srun torchrun --nproc_per_node=4 --master_port="${RANDOM_PORT}" \
    training_scripts/train_stage_3.py

echo "Job finished."
