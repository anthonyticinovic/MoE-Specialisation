#!/bin/bash
#SBATCH --job-name=attention_routing
#SBATCH --output=../out_slurm/attention_routing/attention_routing_%j.out
#SBATCH --error=../err_slurm/attention_routing/attention_routing_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

# Script: Attention-Routing Analysis
# Purpose: Analyze attention and routing patterns across all MoE layers
# Usage: sbatch hpc/analysis_scripts/attention_routing_analysis.sbatch [optional_config_path]

echo "=========================================="
echo "MoE Attention-Routing Analysis"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

# Load modules
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
module load CUDA

source ~/pytorch_latest_venv/bin/activate

echo "=== VERIFICATION ==="
which python
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import matplotlib; import seaborn; print('Visualization packages ready')"

# Check for required packages and install if missing
echo ""
echo "=== CHECKING DEPENDENCIES ==="
python -c "import pandas" 2>/dev/null || {
    echo "pandas not found. Installing..."
    pip install pandas
}
python -c "import numpy" 2>/dev/null || {
    echo "numpy not found. Installing..."
    pip install numpy
}
echo "All dependencies ready."
echo ""

cd $HOME/MoE-Specialisation
export PYTHONPATH="${PWD}:${PYTHONPATH}"
echo "Working directory: $(pwd)"

# Verify GPU availability
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# Config file (can be overridden by command line argument)
CONFIG_FILE="${1:-configs/attention_routing_analysis.json}"
echo "Using config file: $CONFIG_FILE"

# Additional CLI arguments (optional)
EXTRA_ARGS="${@:2}"

echo ""
echo "=========================================="
echo "Starting Attention-Routing Analysis"
echo "=========================================="

# Run analysis
python -u analysis_scripts/attention_routing_analysis.py \
    --config "$CONFIG_FILE" \
    --device cuda \
    $EXTRA_ARGS

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
