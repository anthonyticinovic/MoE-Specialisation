#!/bin/bash
#SBATCH --job-name=pope_stage2
#SBATCH --account=COMP90055
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/home/aticinovic/out_slurm/pope_stage2_%j.out
#SBATCH --error=/home/aticinovic/err_slurm/pope_stage2_%j.err

# Generate POPE answers using Stage 2 model
# Answers yes/no questions for all 3 difficulty levels

set -e

# Load required modules
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
module load CUDA/11.7.0

# Activate virtual environment
source ~/pytorch_latest_venv/bin/activate

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Navigate to workspace
cd /home/aticinovic/MoE-Specialisation

echo "================================================================================"
echo "                    POPE ANSWER GENERATION - STAGE 2                            "
echo "================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "================================================================================"
echo ""

# Stage 2 checkpoint
CHECKPOINT="/data/gpfs/projects/COMP90055/aticinovic/outputs/stage2_checkpoints/llm_stage2_best.pth"
IMAGE_DIR="/data/gpfs/projects/COMP90055/aticinovic/datasets/coco/val2017"
OUTPUT_DIR="results/pope_evaluation"

# Generate answers for all 3 difficulty levels
for DIFFICULTY in random popular adversarial; do
    echo ""
    echo "================================================================================"
    echo "  Processing: ${DIFFICULTY^^} difficulty"
    echo "================================================================================"
    
    python3 analysis_scripts/pope_evaluation/02_generate_pope_answers.py \
        --checkpoint_path "${CHECKPOINT}" \
        --stage_name stage2 \
        --questions_file "results/pope_evaluation/pope_${DIFFICULTY}.json" \
        --image_dir "${IMAGE_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --max_new_tokens 10 \
        --temperature 0.0 \
        --device cuda
done

echo ""
echo "================================================================================"
echo "                    ✅ STAGE 2 ANSWER GENERATION COMPLETE                       "
echo "================================================================================"
echo "End time: $(date)"
echo ""
echo "Results saved to: ${OUTPUT_DIR}/"
echo "  • stage2_random_answers.json"
echo "  • stage2_popular_answers.json"
echo "  • stage2_adversarial_answers.json"
echo "================================================================================"
