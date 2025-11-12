#!/bin/bash
#SBATCH --job-name=llava_wild_full
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/home/aticinovic/MoE-Specialisation/out_slurm/llava_wild_full_%j.out
#SBATCH --error=/home/aticinovic/MoE-Specialisation/err_slurm/llava_wild_full_%j.err

# Complete LLaVA-Wild Evaluation Pipeline
# 1. Evaluate Stage 2
# 2. Evaluate Stage 3
# 3. Compare results and generate visualization

module load anaconda3/2022.05
source activate /data/gpfs/projects/COMP90055/aticinovic/myenv

# Configuration
STAGE2_CHECKPOINT="/data/gpfs/projects/COMP90055/aticinovic/outputs/stage2_checkpoints/llm_stage2_best.pth"
STAGE3_CHECKPOINT="/data/gpfs/projects/COMP90055/aticinovic/outputs/stage3_checkpoints/llm_stage3_best_portable.pth"
LLAVA_JSON="/data/gpfs/projects/COMP90055/aticinovic/datasets/llava/instruct_150k/llava_instruct_150k.json"
IMAGE_DIR="/data/gpfs/projects/COMP90055/aticinovic/datasets/llava/images"
OUTPUT_DIR="/home/aticinovic/MoE-Specialisation/results/llava_wild_evaluation"
SCRIPT_DIR="/home/aticinovic/MoE-Specialisation/analysis_scripts/llava_evaluation"
NUM_SAMPLES=200  # Lightweight evaluation (200 samples)

mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "LLaVA-Wild Full Evaluation Pipeline"
echo "=============================================="
echo "Samples: ${NUM_SAMPLES}"
echo "Output: ${OUTPUT_DIR}"
echo "=============================================="

# Step 1: Evaluate Stage 2
echo ""
echo "🔷 [1/3] Evaluating Stage 2 (Hard Routing)..."
python "${SCRIPT_DIR}/01_llava_wild_eval.py" \
    --llava_json "${LLAVA_JSON}" \
    --image_dir "${IMAGE_DIR}" \
    --checkpoint "${STAGE2_CHECKPOINT}" \
    --stage stage2 \
    --output "${OUTPUT_DIR}/stage2_results.json" \
    --num_samples ${NUM_SAMPLES} \
    --max_tokens 100 \
    --temperature 0.7 \
    --device cuda

if [ $? -ne 0 ]; then
    echo "❌ Stage 2 evaluation failed!"
    exit 1
fi

echo "✅ Stage 2 complete!"

# Step 2: Evaluate Stage 3
echo ""
echo "🔶 [2/3] Evaluating Stage 3 (Soft Routing)..."
python "${SCRIPT_DIR}/01_llava_wild_eval.py" \
    --llava_json "${LLAVA_JSON}" \
    --image_dir "${IMAGE_DIR}" \
    --checkpoint "${STAGE3_CHECKPOINT}" \
    --stage stage3 \
    --output "${OUTPUT_DIR}/stage3_results.json" \
    --num_samples ${NUM_SAMPLES} \
    --max_tokens 100 \
    --temperature 0.7 \
    --device cuda

if [ $? -ne 0 ]; then
    echo "❌ Stage 3 evaluation failed!"
    exit 1
fi

echo "✅ Stage 3 complete!"

# Step 3: Compare results
echo ""
echo "📊 [3/3] Comparing results and generating visualizations..."
python "${SCRIPT_DIR}/02_compare_results.py" \
    --stage2 "${OUTPUT_DIR}/stage2_results.json" \
    --stage3 "${OUTPUT_DIR}/stage3_results.json" \
    --output_dir "${OUTPUT_DIR}"

if [ $? -ne 0 ]; then
    echo "❌ Comparison failed!"
    exit 1
fi

echo "✅ Comparison complete!"

echo ""
echo "=============================================="
echo "✅ FULL EVALUATION COMPLETE"
echo "=============================================="
echo "Results saved to: ${OUTPUT_DIR}"
echo "  - stage2_results.json"
echo "  - stage3_results.json"
echo "  - llava_wild_comparison.png"
echo "=============================================="
