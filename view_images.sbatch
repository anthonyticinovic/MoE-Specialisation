#!/bin/bash
#SBATCH --job-name=view_images
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=00:05:00
#SBATCH --output=../out_slurm/view_images/view_images_%j.out
#SBATCH --error=../err_slurm/view_images/view_images_%j.err

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
python -u view_synthetic_images.py
echo "Job finished."
