#!/bin/bash
#SBATCH --job-name=compositional_study
#SBATCH -p gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:10:00
#SBATCH --output=../out_slurm/compositional_case_study/compositional_%j.out
#SBATCH --error=../err_slurm/compositional_case_study/compositional_%j.err

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

echo "=== Compositional Case Study: Stage 2 vs Stage 3 Comparison ==="
echo "Mode: BOTH stages (Stage 2 hard routing + Stage 3 soft routing)"
echo "Stage 2 checkpoint: llm_stage2_best.pth"
echo "Stage 3 checkpoint: llm_stage3_best_portable.pth"
echo "Temperature: 0.01 (deterministic routing for Stage 3)"
echo "Stimuli: 4 minimal (scalable to more via config)"
echo "Layers: [0, 16, 31] (specified in config file)"
echo "Output: 2 matrices per layer (Stage 2 + Stage 3) + comparison plots"
echo ""

# Run compositional case study analysis
python -u analysis_scripts/compositional_case_study.py \
    --config-file configs/compositional_case_study.json \
    --training-config configs/training_config.yaml \
    --device cuda

echo ""
echo "=== Analysis Complete ==="
echo "Results saved to: results/compositional_case_study/"
echo "Output files per layer (X = 0, 16, 31):"
echo "  Stage 2:"
echo "    - stage2_layerX_matrix.json (similarity matrix)"
echo "    - stage2_layerX_labels.json (row/column labels)"
echo "    - stage2_layerX.png (heatmap visualization)"
echo "  Stage 3:"
echo "    - stage3_layerX_matrix.json (similarity matrix)"
echo "    - stage3_layerX_labels.json (row/column labels)"
echo "    - stage3_layerX.png (heatmap visualization)"
echo "  Comparison:"
echo "    - comparison_cross_modal_layerX.png (side-by-side cross-modal comparison)"
echo ""
echo "Total files: 21 (7 files per layer × 3 layers)"
