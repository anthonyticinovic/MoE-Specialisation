#!/bin/bash
#SBATCH --job-name=concept_stage3
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:15:00
#SBATCH --output=../out_slurm/concept_analysis/concept_stage3_%j.out
#SBATCH --error=../err_slurm/concept_analysis/concept_stage3_%j.err

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

echo "=== Stage 3 Representation Alignment Analysis ==="
echo "Mode: Stage 3 (Learned Soft Routing)"
echo "Checkpoint: llm_stage3_best.pth (FULL checkpoint with vision connector)"
echo "Temperature: 0.01 (deterministic routing)"
echo "Image-text pairs: 8 concepts"
echo "Layers: 31 (final layer)"
echo ""

# Run Stage 3 cross-concept similarity analysis
python -u analysis_scripts/cross_concept_similarity_matrix.py \
    --config-file configs/similarity_matrix_stage3.json \
    --mode stage3 \
    --stage3-checkpoint /data/gpfs/projects/COMP90055/aticinovic/outputs/stage3_checkpoints/llm_stage3_best.pth \
    --temperature 0.01 \
    --layers 31 \
    --output-dir results/stage3_alignment/similarity_matrix/ \
    --training-config configs/training_config.yaml \
    --device cuda

echo ""
echo "=== Analysis Complete ==="
echo "Results saved to: results/stage3_alignment/similarity_matrix/"
echo "Check for:"
echo "  - similarity_matrix_layer31.json (raw matrix)"
echo "  - similarity_matrix_layer31.png (heatmap)"
echo "  - labels_layer31.json (row/column labels)"
