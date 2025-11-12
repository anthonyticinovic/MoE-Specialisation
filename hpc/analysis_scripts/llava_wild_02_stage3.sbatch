#!/bin/bash
#SBATCH --job-name=llava_wild_stage3
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/home/aticinovic/MoE-Specialisation/out_slurm/llava_wild_stage3_%j.out
#SBATCH --error=/home/aticinovic/MoE-Specialisation/err_slurm/llava_wild_stage3_%j.err

# LLaVA-Wild Evaluation - Stage 3 (Soft Routing)
# Tests whether Stage 3 performs better on its TRAINING distribution

module load anaconda3/2022.05
source activate /data/gpfs/projects/COMP90055/aticinovic/myenv

# Configuration
CHECKPOINT="/data/gpfs/projects/COMP90055/aticinovic/outputs/stage3_checkpoints/llm_stage3_best_portable.pth"
LLAVA_JSON="/data/gpfs/projects/COMP90055/aticinovic/datasets/llava/instruct_150k/llava_instruct_150k.json"
IMAGE_DIR="/data/gpfs/projects/COMP90055/aticinovic/datasets/llava/images"
OUTPUT_DIR="/home/aticinovic/MoE-Specialisation/results/llava_wild_evaluation"
NUM_SAMPLES=200  # Lightweight evaluation

mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "LLaVA-Wild Evaluation - Stage 3"
echo "=============================================="
echo "Checkpoint: ${CHECKPOINT}"
echo "Samples: ${NUM_SAMPLES}"
echo "Output: ${OUTPUT_DIR}"
echo "=============================================="

# Run evaluation
python /home/aticinovic/MoE-Specialisation/analysis_scripts/llava_evaluation/01_llava_wild_eval.py \
    --llava_json "${LLAVA_JSON}" \
    --image_dir "${IMAGE_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --stage stage3 \
    --output "${OUTPUT_DIR}/stage3_results.json" \
    --num_samples ${NUM_SAMPLES} \
    --max_tokens 100 \
    --temperature 0.7 \
    --device cuda

echo ""
echo "✅ Stage 3 evaluation complete!"
