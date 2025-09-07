#!/bin/bash
#SBATCH --job-name=vlm_stage2_5_router_training
#SBATCH -p gpu-l40s
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=10:00:00      # Reduced from 16 to 10 hours, which is a safe buffer
#SBATCH --output=../out_slurm/train_stage_2_5/train_stage_2_5_%j.out
#SBATCH --error=../err_slurm/train_stage_2_5/train_stage_2_5_%j.err
#SBATCH --gpus-per-task=4

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

echo "Starting Stage 2.5 Router Training script..."
srun torchrun --nproc_per_node=4 training_scripts/train_stage_2.5.py

echo "Job finished."
