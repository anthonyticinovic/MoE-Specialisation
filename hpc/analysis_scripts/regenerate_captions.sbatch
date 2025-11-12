#!/bin/bash
#SBATCH --job-name=regen_caps
#SBATCH --account=COMP90055
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/home/aticinovic/out_slurm/regen_captions_%j.out
#SBATCH --error=/home/aticinovic/err_slurm/regen_captions_%j.err

# Regenerate captions with optimal settings for COCO evaluation
# Key changes:
# - max_length=20 (was 40) - matches COCO caption length distribution
# - Early stopping at first period - cleaner single-sentence captions
# - Should dramatically improve CIDEr scores

set -e  # Exit on error

# Load required modules
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
module load CUDA/11.7.0

# Activate virtual environment
source ~/pytorch_latest_venv/bin/activate

# Set environment variables
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# Navigate to workspace
cd /home/aticinovic/MoE-Specialisation

# Paths
COCO_DIR="/data/gpfs/projects/COMP90055/aticinovic/datasets/coco"
STAGE2_CHECKPOINT="/data/gpfs/projects/COMP90055/aticinovic/outputs/stage2_checkpoints/llm_stage2_best.pth"
STAGE3_CHECKPOINT="/data/gpfs/projects/COMP90055/aticinovic/outputs/stage3_checkpoints/llm_stage3_best_portable.pth"
IMAGES_JSON="results/karpathy_evaluation/karpathy_test_images.json"
OUTPUT_DIR="results/karpathy_evaluation/captioning"

echo "================================================================================"
echo "                    REGENERATING CAPTIONS WITH OPTIMAL SETTINGS                 "
echo "================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""
echo "Key Settings:"
echo "  • max_length: 20 tokens (was 40)"
echo "  • Early stopping: At first period (after ≥8 tokens)"
echo "  • Expected caption length: ~10-15 words (COCO-optimal)"
echo "================================================================================"
echo ""

# =============================================================================
# STAGE 2: Generate Captions
# =============================================================================
echo ""
echo "================================================================================"
echo "                        GENERATING CAPTIONS - STAGE 2                           "
echo "================================================================================"
echo "Start: $(date)"
echo ""

python3 analysis_scripts/karpathy_evaluation/04_generate_captions.py \
    --checkpoint_path "$STAGE2_CHECKPOINT" \
    --stage_name stage2 \
    --images_json "$IMAGES_JSON" \
    --image_base_dir "$COCO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_beams 5 \
    --max_length 20 \
    --batch_size 16 \
    --device cuda

echo ""
echo "✅ Stage 2 complete: $(date)"
echo ""

# =============================================================================
# STAGE 3: Generate Captions
# =============================================================================
echo ""
echo "================================================================================"
echo "                        GENERATING CAPTIONS - STAGE 3                           "
echo "================================================================================"
echo "Start: $(date)"
echo ""

python3 analysis_scripts/karpathy_evaluation/04_generate_captions.py \
    --checkpoint_path "$STAGE3_CHECKPOINT" \
    --stage_name stage3 \
    --images_json "$IMAGES_JSON" \
    --image_base_dir "$COCO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_beams 5 \
    --max_length 20 \
    --batch_size 16 \
    --device cuda

echo ""
echo "✅ Stage 3 complete: $(date)"
echo ""

# =============================================================================
# Sample Captions Preview
# =============================================================================
echo ""
echo "================================================================================"
echo "                         CAPTION SAMPLES (First 5)                              "
echo "================================================================================"
echo ""

echo "Stage 2 samples:"
python3 << 'EOF'
import json
with open('results/karpathy_evaluation/captioning/stage2_captions.json') as f:
    caps = json.load(f)
    lengths = [len(c['caption'].split()) for c in caps]
    print(f"  Average length: {sum(lengths)/len(lengths):.1f} words")
    print(f"  Examples:")
    for i, cap in enumerate(caps[:5]):
        print(f"    {i+1}. [{len(cap['caption'].split())}w] {cap['caption']}")
EOF

echo ""
echo "Stage 3 samples:"
python3 << 'EOF'
import json
with open('results/karpathy_evaluation/captioning/stage3_captions.json') as f:
    caps = json.load(f)
    lengths = [len(c['caption'].split()) for c in caps]
    print(f"  Average length: {sum(lengths)/len(lengths):.1f} words")
    print(f"  Examples:")
    for i, cap in enumerate(caps[:5]):
        print(f"    {i+1}. [{len(cap['caption'].split())}w] {cap['caption']}")
EOF

echo ""
echo "================================================================================"
echo "                          ✅ CAPTION GENERATION COMPLETE                        "
echo "================================================================================"
echo "End time: $(date)"
echo ""
echo "Next step: Run evaluation script to compute BLEU, CIDEr, METEOR, ROUGE-L"
echo "  Command: python3 eval_captions_standalone.py"
echo "================================================================================"
