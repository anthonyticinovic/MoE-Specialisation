#!/bin/bash
#SBATCH --job-name=test_cap
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=0:20:00
#SBATCH --output=/home/aticinovic/out_slurm/test_caption_%j.out
#SBATCH --error=/home/aticinovic/err_slurm/test_caption_%j.err

set -e  # Exit on error

# Load required modules
module load GCCcore/11.3.0
module load Python/3.11.3
module load CUDA/11.7.0

# Activate virtual environment
source ~/pytorch_latest_venv/bin/activate

# Set paths
WORKSPACE_DIR="/home/aticinovic/MoE-Specialisation"
COCO_DIR="/data/gpfs/projects/COMP90055/aticinovic/datasets/coco"
STAGE2_CHECKPOINT="/data/gpfs/projects/COMP90055/aticinovic/outputs/stage2_checkpoints/llm_stage2_best.pth"
STAGE3_CHECKPOINT="/data/gpfs/projects/COMP90055/aticinovic/outputs/stage3_checkpoints/llm_stage3_best_portable.pth"

cd $WORKSPACE_DIR

echo "================================================================================"
echo "                    QUICK TEST - STAGE 2 (50 images)                           "
echo "================================================================================"
echo "Start: $(date)"
echo ""

python analysis_scripts/karpathy_evaluation/04_generate_captions.py \
    --checkpoint_path "$STAGE2_CHECKPOINT" \
    --stage_name stage2 \
    --images_json results/karpathy_evaluation/karpathy_test_images.json \
    --image_base_dir "$COCO_DIR" \
    --output_dir results/karpathy_evaluation/captioning_test \
    --num_beams 5 \
    --max_length 40 \
    --batch_size 16 \
    --num_images 50 \
    --device cuda

echo ""
echo "================================================================================"
echo "                    ✅ STAGE 2 TEST COMPLETE                                    "
echo "================================================================================"
echo "Completed: $(date)"
echo ""

echo "================================================================================"
echo "                    QUICK TEST - STAGE 3 (50 images)                           "
echo "================================================================================"
echo "Start: $(date)"
echo ""

python analysis_scripts/karpathy_evaluation/04_generate_captions.py \
    --checkpoint_path "$STAGE3_CHECKPOINT" \
    --stage_name stage3 \
    --images_json results/karpathy_evaluation/karpathy_test_images.json \
    --image_base_dir "$COCO_DIR" \
    --output_dir results/karpathy_evaluation/captioning_test \
    --num_beams 5 \
    --max_length 40 \
    --batch_size 16 \
    --num_images 50 \
    --device cuda

echo ""
echo "================================================================================"
echo "                    ✅ STAGE 3 TEST COMPLETE                                    "
echo "================================================================================"
echo "Completed: $(date)"
echo ""

echo "================================================================================"
echo "                    📊 SAMPLE CAPTIONS COMPARISON                               "
echo "================================================================================"
echo ""
echo "Stage 2 samples:"
head -20 results/karpathy_evaluation/captioning_test/stage2_captions.json
echo ""
echo "Stage 3 samples:"
head -20 results/karpathy_evaluation/captioning_test/stage3_captions.json
echo ""
