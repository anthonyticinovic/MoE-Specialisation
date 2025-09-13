#!/bin/bash
#SBATCH --job-name=vlm_stage1_training
#SBATCH -p gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=../out_slurm/train_stage_1/train_stage_1_%j.out
#SBATCH --error=../err_slurm/train_stage_1/train_stage_1_%j.err

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
python training_scripts/train_stage_1.py

echo "Job finished."
