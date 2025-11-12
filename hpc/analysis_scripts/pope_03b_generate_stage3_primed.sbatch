#!/bin/bash
#SBATCH --job-name=pope_s3_primed
#SBATCH --output=/home/aticinovic/out_slurm/pope_stage3_primed_%j.out
#SBATCH --error=/home/aticinovic/err_slurm/pope_stage3_primed_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8

# POPE Evaluation - Stage 3 with Priming Strategy
# This script tests whether priming Stage 3 with a "fake previous answer"
# can exploit its learned multi-turn conversation behavior to generate
# better yes/no responses.

set -e  # Exit on error

# Load modules
module load GCCcore/11.3.0
module load Python/3.11.3
module load CUDA/11.7.0

echo "=== POPE Stage 3 - Priming Strategy ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Activate environment
source /home/aticinovic/pytorch_latest_venv/bin/activate

# Verify environment
echo "=== VERIFICATION ==="
which python
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
echo ""

# Paths
WORKSPACE="/home/aticinovic/MoE-Specialisation"
RESULTS_DIR="${WORKSPACE}/results/pope_evaluation"
QUESTIONS_DIR="${RESULTS_DIR}"
ANSWERS_DIR="${RESULTS_DIR}/answers_primed"
IMAGE_DIR="/data/gpfs/projects/COMP90055/aticinovic/datasets/coco/val2017"
CHECKPOINT="/data/gpfs/projects/COMP90055/aticinovic/outputs/stage3_checkpoints/llm_stage3_best_portable.pth"

# Create output directory
mkdir -p "${ANSWERS_DIR}"

# Test different priming strategies
STRATEGIES=("simple" "conversational" "none")

for STRATEGY in "${STRATEGIES[@]}"; do
    echo "========================================================================"
    echo "Testing priming strategy: ${STRATEGY}"
    echo "========================================================================"
    
    # Test on all three difficulties
    for DIFFICULTY in random popular adversarial; do
        echo ""
        echo "--- Difficulty: ${DIFFICULTY} ---"
        
        QUESTIONS="${QUESTIONS_DIR}/pope_${DIFFICULTY}.json"
        OUTPUT="${ANSWERS_DIR}/stage3_${DIFFICULTY}_${STRATEGY}.json"
        
        echo "Questions: ${QUESTIONS}"
        echo "Output: ${OUTPUT}"
        echo "Priming: ${STRATEGY}"
        echo ""
        
        python "${WORKSPACE}/analysis_scripts/pope_evaluation/02b_generate_pope_answers_primed.py" \
            --questions "${QUESTIONS}" \
            --output "${OUTPUT}" \
            --image_dir "${IMAGE_DIR}" \
            --checkpoint "${CHECKPOINT}" \
            --stage stage3 \
            --priming "${STRATEGY}" \
            --max_tokens 10 \
            --temperature 0.0 \
            --device cuda
        
        if [ $? -eq 0 ]; then
            echo "✅ ${DIFFICULTY} with ${STRATEGY} priming completed successfully"
        else
            echo "❌ ${DIFFICULTY} with ${STRATEGY} priming failed"
        fi
        echo ""
    done
done

echo ""
echo "=== SUMMARY ==="
echo "Tested priming strategies: ${STRATEGIES[@]}"
echo "Generated answers saved to: ${ANSWERS_DIR}"
echo ""
echo "Next step: Compare results across strategies to see which works best"
echo "  - simple: Prime with 'This image has been analyzed.'"
echo "  - conversational: Prime with full Q&A pair"
echo "  - none: No priming (baseline)"
echo ""
echo "End time: $(date)"
echo "=== JOB COMPLETE ==="
