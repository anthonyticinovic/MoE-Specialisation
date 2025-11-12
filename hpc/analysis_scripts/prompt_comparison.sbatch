#!/bin/bash
#SBATCH --job-name=prompt_test
#SBATCH --account=COMP90055
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/home/aticinovic/out_slurm/prompt_comparison_%j.out
#SBATCH --error=/home/aticinovic/err_slurm/prompt_comparison_%j.err

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

CHECKPOINT="/data/gpfs/projects/COMP90055/aticinovic/outputs/stage3_checkpoints/llm_stage3_best_portable.pth"
IMAGES_JSON="results/karpathy_evaluation/karpathy_test_images.json"
IMAGE_BASE="/data/gpfs/projects/COMP90055/aticinovic/datasets/coco"

echo "=========================================="
echo "Testing Multiple Prompts for COCO-Style Captions"
echo "=========================================="
echo "Date: $(date)"
echo ""

# Array of prompts to test
declare -a PROMPTS=(
    "Caption:"
    "Briefly describe what you see."
    "What is in this image?"
    "A photo of"
)

for i in "${!PROMPTS[@]}"; do
    PROMPT="${PROMPTS[$i]}"
    OUTPUT_DIR="results/karpathy_evaluation/prompt_test_$i"
    
    echo "=========================================="
    echo "Test $((i+1))/4: '$PROMPT'"
    echo "=========================================="
    
    # Modify the prompt in the generation script temporarily
    python3 << EOF
# Read the file
with open('analysis_scripts/karpathy_evaluation/04_generate_captions.py', 'r') as f:
    content = f.read()

# Replace the prompt
import re
pattern = r'prompt_text = "[^"]*"'
replacement = f'prompt_text = "$PROMPT"'
content = re.sub(pattern, replacement, content)

# Write it back
with open('analysis_scripts/karpathy_evaluation/04_generate_captions.py', 'w') as f:
    f.write(content)
    
print(f"✅ Set prompt to: '$PROMPT'")
EOF
    
    # Also update the stripping code
    python3 << EOF
with open('analysis_scripts/karpathy_evaluation/04_generate_captions.py', 'r') as f:
    lines = f.readlines()

# Find and update the prompt_text line in the stripping section
for idx, line in enumerate(lines):
    if 'For Stage 3, remove the prompt' in line:
        # Find the next prompt_text assignment
        for j in range(idx, min(idx + 5, len(lines))):
            if 'prompt_text = ' in lines[j] and 'is_stage3' not in lines[j]:
                lines[j] = f'                        prompt_text = "$PROMPT"\n'
                break
        break

with open('analysis_scripts/karpathy_evaluation/04_generate_captions.py', 'w') as f:
    f.writelines(lines)
    
print(f"✅ Updated stripping for: '$PROMPT'")
EOF
    
    # Generate captions
    python3 analysis_scripts/karpathy_evaluation/04_generate_captions.py \
        --checkpoint_path "$CHECKPOINT" \
        --stage_name stage3 \
        --images_json "$IMAGES_JSON" \
        --image_base_dir "$IMAGE_BASE" \
        --output_dir "$OUTPUT_DIR" \
        --num_beams 5 \
        --max_length 40 \
        --batch_size 2 \
        --num_images 10 \
        --device cuda
    
    echo ""
    echo "📊 Sample captions with prompt '$PROMPT':"
    python3 -c "
import json
with open('$OUTPUT_DIR/stage3_captions.json') as f:
    data = json.load(f)
    for item in data[:3]:
        print(f'  • {item[\"caption\"][:80]}...')
"
    echo ""
done

echo "=========================================="
echo "Comparison Complete!"
echo "=========================================="
echo ""
echo "Output directories:"
for i in {0..3}; do
    echo "  Prompt $((i+1)): results/karpathy_evaluation/prompt_test_$i/"
done
