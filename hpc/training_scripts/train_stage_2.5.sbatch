#!/bin/bash
#SBATCH --job-name=vlm_stage2_5_router_training
#SBATCH -p gpu-h100
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=../out_slurm/train_stage_2_5/train_stage_2_5_%j.out
#SBATCH --error=../err_slurm/train_stage_2_5/train_stage_2_5_%j.err

# Load modules
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
module load CUDA

source ~/pytorch_latest_venv/bin/activate

echo "=== VERIFICATION ==="
which python
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

cd $HOME/MoE-Specialisation
export PYTHONPATH="${PWD}:${PYTHONPATH}"


export NCCL_TIMEOUT=3600
export OMP_NUM_THREADS=8

# enable blocking wait for better error reporting
export TORCH_NCCL_BLOCKING_WAIT=1  # ✅ Correct
export NCCL_DEBUG=WARN

echo "Starting Stage 2.5 Router Training script..."
srun torchrun --nproc_per_node=2 training_scripts/train_stage_2.5.py

echo "Job finished."
