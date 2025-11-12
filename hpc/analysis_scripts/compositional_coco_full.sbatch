#!/bin/bash
#SBATCH --job-name=comp_coco_full
#SBATCH -p gpu-a100-preempt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:15:00
#SBATCH --output=../out_slurm/compositional_coco/full_analysis_%j.out
#SBATCH --error=../err_slurm/compositional_coco/full_analysis_%j.err

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

echo ""
echo "================================================================================"
echo "  COMPOSITIONAL CASE STUDY: Complete Stage 2 vs Stage 3 Analysis"
echo "================================================================================"
echo "Concepts: red_apple, green_apple, strawberry, blueberry"
echo "Sampling: N=50 per concept from COCO"
echo "Layers: [0, 16, 31]"
echo "Stages: Stage 2 (hard routing) → Stage 3 (soft routing) → Comparison"
echo "================================================================================"
echo ""

# ============================================================================
# STAGE 1: Run Stage 2 Analysis (Hard Routing)
# ============================================================================
echo ""
echo "================================================================================"
echo "STAGE 2 ANALYSIS (Hard Routing)"
echo "================================================================================"
echo "Expected time: ~30-45 minutes"
echo ""

python -u analysis_scripts/cross_concept_similarity_matrix.py \
    --config-file configs/compositional_coco.json \
    --mode stage2 \
    --output-dir results/compositional_coco/stage2/ \
    --training-config configs/training_config.yaml \
    --device cuda

STAGE2_EXIT=$?
if [ $STAGE2_EXIT -ne 0 ]; then
    echo ""
    echo "❌ ERROR: Stage 2 analysis failed with exit code $STAGE2_EXIT"
    echo "Check error log for details"
    exit $STAGE2_EXIT
fi

echo ""
echo "✅ Stage 2 analysis complete!"
echo "Results saved to: results/compositional_coco/stage2/"
echo ""

# ============================================================================
# STAGE 2: Run Stage 3 Analysis (Soft Routing)
# ============================================================================
echo ""
echo "================================================================================"
echo "STAGE 3 ANALYSIS (Soft Routing)"
echo "================================================================================"
echo "Expected time: ~30-45 minutes"
echo ""

python -u analysis_scripts/cross_concept_similarity_matrix.py \
    --config-file configs/compositional_coco.json \
    --mode stage3 \
    --stage3-checkpoint /data/gpfs/projects/COMP90055/aticinovic/outputs/stage3_checkpoints/llm_stage3_best_portable.pth \
    --temperature 0.01 \
    --output-dir results/compositional_coco/stage3/ \
    --training-config configs/training_config.yaml \
    --device cuda

STAGE3_EXIT=$?
if [ $STAGE3_EXIT -ne 0 ]; then
    echo ""
    echo "❌ ERROR: Stage 3 analysis failed with exit code $STAGE3_EXIT"
    echo "Check error log for details"
    exit $STAGE3_EXIT
fi

echo ""
echo "✅ Stage 3 analysis complete!"
echo "Results saved to: results/compositional_coco/stage3/"
echo ""

# ============================================================================
# STAGE 3: Create Comparison Visualizations
# ============================================================================
echo ""
echo "================================================================================"
echo "COMPARISON VISUALIZATIONS (Stage 2 vs Stage 3)"
echo "================================================================================"
echo "Expected time: ~2 minutes"
echo ""

python -u analysis_scripts/create_stage_comparison.py \
    --stage2-dir results/compositional_coco/stage2/ \
    --stage3-dir results/compositional_coco/stage3/ \
    --layers 0 16 31 \
    --output-dir results/compositional_coco/comparison/ \
    --temperature 0.01

COMPARISON_EXIT=$?
if [ $COMPARISON_EXIT -ne 0 ]; then
    echo ""
    echo "⚠️  WARNING: Comparison visualization failed with exit code $COMPARISON_EXIT"
    echo "Stage 2 and Stage 3 results are still available"
    exit $COMPARISON_EXIT
fi

echo ""
echo "✅ Comparison visualizations complete!"
echo "Results saved to: results/compositional_coco/comparison/"
echo ""

# ============================================================================
# FINAL SUMMARY
# ============================================================================
echo ""
echo "================================================================================"
echo "  ✅ COMPLETE ANALYSIS FINISHED SUCCESSFULLY!"
echo "================================================================================"
echo ""
echo "📁 Results organized in: results/compositional_coco/"
echo ""
echo "📊 Stage 2 (Hard Routing):"
echo "   - results/compositional_coco/stage2/"
echo "   - 3 layers × 4 files per layer = 12 files"
echo ""
echo "📊 Stage 3 (Soft Routing):"
echo "   - results/compositional_coco/stage3/"
echo "   - 3 layers × 4 files per layer = 12 files"
echo ""
echo "📊 Comparison (Side-by-Side):"
echo "   - results/compositional_coco/comparison/"
echo "   - 3 comparison plots (one per layer)"
echo ""
echo "🎯 Key files to review:"
echo "   - comparison/comparison_cross_modal_layer31.png (final layer comparison)"
echo "   - stage2/similarity_matrix_layer31_cross_modal.png (Stage 2 baseline)"
echo "   - stage3/similarity_matrix_layer31_cross_modal.png (Stage 3 alignment)"
echo ""
echo "================================================================================"
echo "Total files created: 27 (12 + 12 + 3)"
echo "================================================================================"
