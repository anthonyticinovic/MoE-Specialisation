#!/bin/bash
#SBATCH --job-name=pope_prime_test
#SBATCH --output=/home/aticinovic/out_slurm/pope_prime_test_%j.out
#SBATCH --error=/home/aticinovic/err_slurm/pope_prime_test_%j.err
#SBATCH --time=00:15:00
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8

# Quick POPE Priming Test - Just 100 questions with simple priming
# Fast test to see if priming strategy works

set -e  # Exit on error

# Load modules
module load GCCcore/11.3.0
module load Python/3.11.3
module load CUDA/11.7.0

echo "=== Quick POPE Priming Test ==="
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
ANSWERS_DIR="${RESULTS_DIR}/answers_primed_test"
IMAGE_DIR="/data/gpfs/projects/COMP90055/aticinovic/datasets/coco/val2017"
CHECKPOINT="/data/gpfs/projects/COMP90055/aticinovic/outputs/stage3_checkpoints/llm_stage3_best_portable.pth"

# Create output directory
mkdir -p "${ANSWERS_DIR}"

echo "========================================================================"
echo "Quick Test: Simple Priming vs No Priming (100 questions each)"
echo "========================================================================"
echo ""

# Create a subset of questions (first 100 from random)
FULL_QUESTIONS="${RESULTS_DIR}/pope_random.json"
TEST_QUESTIONS="${ANSWERS_DIR}/pope_random_100.json"

echo "Creating test subset (100 questions)..."
python -c "
import json
with open('${FULL_QUESTIONS}', 'r') as f:
    data = json.load(f)
subset = data[:100]
with open('${TEST_QUESTIONS}', 'w') as f:
    json.dump(subset, f, indent=2)
print(f'Created test set: {len(subset)} questions')
"
echo ""

# Test 1: Simple priming
echo "========================================================================" 
echo "Test 1: SIMPLE PRIMING (100 questions)"
echo "========================================================================"
echo ""

python "${WORKSPACE}/analysis_scripts/pope_evaluation/02b_generate_pope_answers_primed.py" \
    --questions "${TEST_QUESTIONS}" \
    --output "${ANSWERS_DIR}/stage3_simple.json" \
    --image_dir "${IMAGE_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --stage stage3 \
    --priming simple \
    --max_tokens 10 \
    --temperature 0.0 \
    --device cuda

if [ $? -eq 0 ]; then
    echo "✅ Simple priming completed"
else
    echo "❌ Simple priming failed"
    exit 1
fi
echo ""

# Test 2: No priming (baseline)
echo "========================================================================"
echo "Test 2: NO PRIMING - Baseline (100 questions)"
echo "========================================================================"
echo ""

python "${WORKSPACE}/analysis_scripts/pope_evaluation/02b_generate_pope_answers_primed.py" \
    --questions "${TEST_QUESTIONS}" \
    --output "${ANSWERS_DIR}/stage3_none.json" \
    --image_dir "${IMAGE_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --stage stage3 \
    --priming none \
    --max_tokens 10 \
    --temperature 0.0 \
    --device cuda

if [ $? -eq 0 ]; then
    echo "✅ No priming (baseline) completed"
else
    echo "❌ No priming failed"
    exit 1
fi
echo ""

# Quick comparison
echo "========================================================================"
echo "QUICK COMPARISON"
echo "========================================================================"
echo ""

python -c "
import json
from collections import Counter

def compute_stats(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    predictions = [item['predicted_answer'] for item in data]
    answers = [item['answer'] for item in data]
    
    pred_counts = Counter(predictions)
    correct = sum(1 for p, a in zip(predictions, answers) if p == a)
    answerable = sum(1 for p in predictions if p != 'unclear')
    
    accuracy = (correct / answerable * 100) if answerable > 0 else 0
    
    return {
        'yes_pct': pred_counts['yes'] / len(predictions) * 100,
        'no_pct': pred_counts['no'] / len(predictions) * 100,
        'unclear_pct': pred_counts['unclear'] / len(predictions) * 100,
        'accuracy': accuracy,
        'total': len(predictions)
    }

simple = compute_stats('${ANSWERS_DIR}/stage3_simple.json')
none = compute_stats('${ANSWERS_DIR}/stage3_none.json')

print('Strategy         Accuracy    Yes%     No%    Unclear%')
print('='*60)
print(f'Simple Priming   {simple[\"accuracy\"]:>7.1f}%  {simple[\"yes_pct\"]:>6.1f}%  {simple[\"no_pct\"]:>5.1f}%  {simple[\"unclear_pct\"]:>7.1f}%')
print(f'No Priming       {none[\"accuracy\"]:>7.1f}%  {none[\"yes_pct\"]:>6.1f}%  {none[\"no_pct\"]:>5.1f}%  {none[\"unclear_pct\"]:>7.1f}%')
print('='*60)
print()

improvement = simple['accuracy'] - none['accuracy']
if improvement > 2:
    print(f'✅ PRIMING WORKS! +{improvement:.1f}% accuracy improvement')
elif improvement > 0:
    print(f'⚠️  Minor improvement: +{improvement:.1f}% accuracy')
else:
    print(f'❌ No improvement: {improvement:.1f}% change')
print()
print('Results saved to: ${ANSWERS_DIR}/')
"

echo ""
echo "End time: $(date)"
echo "=== TEST COMPLETE ==="
