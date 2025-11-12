#!/bin/bash
#SBATCH --job-name=test_prompt
#SBATCH --account=COMP90055
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=00:15:00
#SBATCH --output=/home/aticinovic/out_slurm/test_prompt_%j.out
#SBATCH --error=/home/aticinovic/err_slurm/test_prompt_%j.err

# Environment setup
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
module load CUDA/11.7.0

# Activate environment
source ~/pytorch_latest_venv/bin/activate

# Set environment variables
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# Navigate to workspace
cd /home/aticinovic/MoE-Specialisation

echo "=========================================="
echo "Testing Stage 3 Generation WITH Question Prompt"
echo "=========================================="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo ""

# Test Stage 3 with prompt on 10 images
echo "📝 Generating captions for 10 test images..."
python3 analysis_scripts/karpathy_evaluation/04_generate_captions.py \
  --checkpoint_path /data/gpfs/projects/COMP90055/aticinovic/outputs/stage3_checkpoints/llm_stage3_best_portable.pth \
  --stage_name stage3 \
  --images_json results/karpathy_evaluation/karpathy_test_images.json \
  --image_base_dir /data/gpfs/projects/COMP90055/aticinovic/datasets/coco \
  --output_dir results/karpathy_evaluation/test_with_prompt \
  --num_beams 5 \
  --max_length 40 \
  --batch_size 2 \
  --num_images 10 \
  --device cuda

echo ""
echo "=========================================="
echo "📊 Generated Captions (with prompt):"
echo "=========================================="
python3 -c "
import json
with open('results/karpathy_evaluation/test_with_prompt/stage3_captions.json') as f:
    data = json.load(f)
    for i, item in enumerate(data[:10]):
        print(f'{i+1}. [{item[\"image_id\"]}]: {item[\"caption\"]}')
"

echo ""
echo "=========================================="
echo "Done! $(date)"
echo "=========================================="
