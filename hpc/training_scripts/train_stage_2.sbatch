#!/bin/bash
#SBATCH --job-name=vlm_stage2_training
#SBATCH -p gpu-l40s
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=21:00:00
#SBATCH --output=../out_slurm/train_stage_2/train_stage_2_%j.out
#SBATCH --error=../err_slurm/train_stage_2/train_stage_2_%j.err

# Load modules
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
module load CUDA  # Load the default CUDA

source ~/pytorch_latest_venv/bin/activate # Activate the new venv

echo "=== VERIFICATION ==="
# Verify that python is coming from the correct environment
which python
# Verify PyTorch can see the GPU
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Rest of your script...
cd $HOME/MoE-Specialisation
export PYTHONPATH="${PWD}:${PYTHONPATH}"

echo "Starting training script..."
export NCCL_DEBUG=INFO # Optional: for debugging network communication
export PYTHONFAULTHANDLER=1 # Optional: for better tracebacks on segfaults
export NCCL_TIMEOUT=900    

srun torchrun --nproc_per_node=4 training_scripts/train_stage_2.py

echo "Job finished."
