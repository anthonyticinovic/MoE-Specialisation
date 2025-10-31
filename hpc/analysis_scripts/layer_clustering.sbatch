#!/bin/bash
#SBATCH --job-name=layer_clustering
#SBATCH --output=../out_slurm/clustering_analysis/layer_clustering_%j.out
#SBATCH --error=../err_slurm/clustering_analysis/layer_clustering_%j.err
#SBATCH --time=00:05:00
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

# Script: Layer Clustering Analysis
# Purpose: Analyze representation clustering at MoE layers to visualize modality/concept specialization
# Usage: sbatch hpc/analysis_scripts/layer_clustering.sbatch [optional_config_path]

echo "=========================================="
echo "MoE Layer Clustering Analysis"
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
python -c "import sklearn" 2>/dev/null || {
    echo "scikit-learn not found. Installing..."
    pip install scikit-learn
}
python -c "import pacmap" 2>/dev/null || {
    echo "pacmap not found. Installing..."
    pip install pacmap
}
python -c "import pandas" 2>/dev/null || {
    echo "pandas not found. Installing..."
    pip install pandas
}
python -c "import tabulate" 2>/dev/null || {
    echo "tabulate not found. Installing..."
    pip install tabulate
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
CONFIG_FILE="${1:-configs/clustering_analysis.json}"
echo "Using config file: $CONFIG_FILE"

# Additional CLI arguments (optional)
EXTRA_ARGS="${@:2}"

echo ""
echo "=========================================="
echo "Starting Layer Clustering Analysis"
echo "=========================================="

# Run analysis
python -u analysis_scripts/layer_clustering_analysis.py \
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
