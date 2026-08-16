#!/bin/bash
# Stage 0 — build the MoE model from a Mistral-7B checkpoint. CPU-only, run once.
#
# Adapt the #SBATCH header to your cluster; see hpc/README.md. Submit from the
# repository root so the log paths resolve:
#   mkdir -p logs/slurm && sbatch hpc/model_scripts/create_moe_model.sbatch
#SBATCH --job-name=moe_create_model
#SBATCH -p bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# CPU-only: skip the CUDA module. Set before sourcing, which leaves the default
# in cluster_env.sh alone.
MOE_MODULES="GCCcore/11.3.0 Python/3.11.3"

source "${MOE_PROJECT_DIR:-$HOME/MoE-Specialisation}/hpc/cluster_env.sh"
moe_setup_environment

# Both paths are required by the CLI. Override either when submitting:
#   MOE_BASE_MODEL=/data/Mistral-7B-v0.3 sbatch hpc/model_scripts/create_moe_model.sbatch
: "${MOE_BASE_MODEL:=$HOME/models/Mistral-7B-v0.3}"
: "${MOE_OUTPUT_MODEL:=$HOME/models/Mistral-7B-MoE}"

echo "Building MoE model: ${MOE_BASE_MODEL} -> ${MOE_OUTPUT_MODEL}"
python -m models.utils.create_moe_model \
    --base-model "${MOE_BASE_MODEL}" \
    --output "${MOE_OUTPUT_MODEL}"

echo "Job finished."
