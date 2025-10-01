#!/bin/bash
#SBATCH --job-name=expert_analysis
#SBATCH -p gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16      
#SBATCH --mem=128G             
#SBATCH --time=01:00:00        
#SBATCH --output=../out_slurm/analyse_experts/analyse_experts_%j.out
#SBATCH --error=../err_slurm/analyse_experts/analyse_experts_%j.err

# Load modules
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
module load CUDA

# Activate your virtual environment
source ~/pytorch_latest_venv/bin/activate

# --- Verification ---
echo "=== VERIFICATION ==="
which python
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
echo "--------------------------"

# --- Run the Script ---
cd $HOME/MoE-Specialisation
export PYTHONPATH="${PWD}:${PYTHONPATH}"

echo "Starting expert analysis script..."
# Run the analysis script with your desired parameters.
# You can easily change the --num_samples and --layer_to_analyze flags.
#
# --num_samples: How many images to process for the analysis.
# --layer_to_analyze: Can be "average", "all", or a specific layer number like "16".
#
python analysis_scripts/token_activations.py \
    --num_samples 500 \
    --top_k 10 \
    --layer_to_analyze "average"

echo "Job finished."
