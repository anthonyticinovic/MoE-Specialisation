#!/bin/bash
#SBATCH --job-name=create_moe_model
#SBATCH -p bigmem                 # Use a CPU partition with high memory
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G                 # Keep the high memory request
#SBATCH --time=00:20:00           # 20 minutes should be more than enough
#SBATCH --output=../out_slurm/create_moe/create_moe_%j.out
#SBATCH --error=../err_slurm/create_moe/create_moe_%j.err

# Load modules
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
# No need to load CUDA for this CPU-only task

# Activate your virtual environment
source ~/pytorch_latest_venv/bin/activate

# --- Verification ---
echo "--- Environment Verification ---"
which python
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
echo "--------------------------"

# --- Add this line to fix the import error ---
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# --- Run the Script ---
cd $HOME/MoE-Specialisation
echo "Starting model creation script..."
python models/utils/create_moe_model.py
echo "Script finished."
