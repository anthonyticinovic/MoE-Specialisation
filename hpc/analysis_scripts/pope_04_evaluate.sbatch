#!/bin/bash
#SBATCH --job-name=pope_eval
#SBATCH --account=COMP90055
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=/home/aticinovic/out_slurm/pope_evaluate_%j.out
#SBATCH --error=/home/aticinovic/err_slurm/pope_evaluate_%j.err

# Evaluate POPE answers and generate comparison
# Computes accuracy, precision, recall, F1 for Stage 2 and Stage 3

set -e

# Load required modules
module purge
module load GCCcore/11.3.0
module load Python/3.11.3

# Activate virtual environment
source ~/pytorch_latest_venv/bin/activate

# Set environment variables
export PYTHONUNBUFFERED=1

# Navigate to workspace
cd /home/aticinovic/MoE-Specialisation

echo "================================================================================"
echo "                            POPE EVALUATION                                     "
echo "================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "================================================================================"
echo ""

# Run evaluation
python3 analysis_scripts/pope_evaluation/03_evaluate_pope.py \
    --stage2_dir results/pope_evaluation \
    --stage3_dir results/pope_evaluation \
    --difficulties random popular adversarial \
    --output_dir results/pope_evaluation

echo ""
echo "================================================================================"
echo "                        ✅ EVALUATION COMPLETE                                  "
echo "================================================================================"
echo "End time: $(date)"
echo ""
echo "Results saved to: results/pope_evaluation/"
echo "  • pope_metrics.json"
echo "  • pope_comparison.txt"
echo "================================================================================"
