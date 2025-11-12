#!/bin/bash
#SBATCH --job-name=eval_caps
#SBATCH --account=COMP90055
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/aticinovic/out_slurm/eval_captions_%j.out
#SBATCH --error=/home/aticinovic/err_slurm/eval_captions_%j.err

# Evaluate Stage 2 and Stage 3 captions using COCO metrics
# This script computes BLEU-1/2/3/4, CIDEr, METEOR, SPICE, ROUGE-L

set -e  # Exit on error

# Load required modules
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
module load Java/17.0.6

# Activate virtual environment
source ~/pytorch_latest_venv/bin/activate

# Install pycocoevalcap if not already installed (Python 3 compatible version)
pip install pycocoevalcap --quiet 2>/dev/null || echo "pycocoevalcap already installed or installation failed"

# Set environment variables
export PYTHONUNBUFFERED=1

# Navigate to workspace
cd /home/aticinovic/MoE-Specialisation

echo "================================================================================"
echo "                      KARPATHY COCO CAPTION EVALUATION                          "
echo "================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "================================================================================"
echo ""

# Run evaluation
python3 analysis_scripts/karpathy_evaluation/05_evaluate_captioning.py \
    --references_json results/karpathy_evaluation/karpathy_test_references.json \
    --stage2_captions results/karpathy_evaluation/captioning/stage2_captions.json \
    --stage3_captions results/karpathy_evaluation/captioning/stage3_captions.json \
    --output_dir results/karpathy_evaluation/captioning

echo ""
echo "================================================================================"
echo "                          ✅ EVALUATION COMPLETE                                "
echo "================================================================================"
echo "End time: $(date)"
echo ""
echo "Results saved to: results/karpathy_evaluation/captioning/"
echo "  • Metrics JSON: captioning_metrics.json"
echo "  • Comparison table: captioning_comparison.txt"
echo "================================================================================"
