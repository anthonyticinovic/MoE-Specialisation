#!/bin/bash
#SBATCH --job-name=vlm_stage3_training
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=01:30:00
#SBATCH --output=../out_slurm/train_stage_3/train_stage_3_%j.out
#SBATCH --error=../err_slurm/train_stage_3/train_stage_3_%j.err

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

# CRITICAL: PyTorch 2.5+ uses TORCH_NCCL_* environment variables
# NCCL_TIMEOUT is deprecated and ignored!
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # Enable async error handling
export TORCH_NCCL_BLOCKING_WAIT=1          # Wait for collectives to complete
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600  # 60 minutes for checkpoint loading
export TORCH_DISTRIBUTED_DEBUG=DETAIL      # Detailed distributed debugging

# CUDA memory management: Reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# CRITICAL: Use a random port to avoid conflicts with other jobs
# Port range: 29500-65000 to avoid conflicts
RANDOM_PORT=$((29500 + RANDOM % 35500))
echo "Using random port: $RANDOM_PORT"

srun torchrun --nproc_per_node=2 --master_port=$RANDOM_PORT training_scripts/train_stage_3.py

echo "Job finished."
