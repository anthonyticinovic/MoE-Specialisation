#!/bin/bash
#SBATCH --job-name=purity_analysis
#SBATCH -p gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:05:00
#SBATCH --output=../out_slurm/purity_analysis/purity_%j.out
#SBATCH --error=../err_slurm/purity_analysis/purity_%j.err

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

cd $HOME/MoE-Specialisation
export PYTHONPATH="${PWD}:${PYTHONPATH}"

echo "Starting cross-modality purity analysis..."

# Run with default concepts and layers + DEBUG MODE
python -u analysis_scripts/cross_modality_purity.py \
    --concepts red blue circle triangle\
    --all-layers \
    --output-dir results/cross_modality \
    --debug
    
echo "Analysis finished."
