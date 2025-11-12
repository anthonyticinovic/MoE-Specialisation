#!/bin/bash
#SBATCH --job-name=pope_gen_q
#SBATCH --account=COMP90055
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/aticinovic/out_slurm/pope_gen_questions_%j.out
#SBATCH --error=/home/aticinovic/err_slurm/pope_gen_questions_%j.err

# Generate POPE questions from COCO val2017 annotations
# Creates yes/no questions for object hallucination evaluation

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
echo "                        POPE QUESTION GENERATION                                "
echo "================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "================================================================================"
echo ""

# Run question generation
python3 analysis_scripts/pope_evaluation/01_generate_pope_questions.py \
    --annotations_file /data/gpfs/projects/COMP90055/aticinovic/datasets/coco/annotations/instances_val2017.json \
    --output_dir results/pope_evaluation \
    --num_images 500 \
    --questions_per_image 3 \
    --seed 42

echo ""
echo "================================================================================"
echo "                        ✅ QUESTION GENERATION COMPLETE                         "
echo "================================================================================"
echo "End time: $(date)"
echo ""
echo "Results saved to: results/pope_evaluation/"
echo "  • pope_random.json"
echo "  • pope_popular.json"
echo "  • pope_adversarial.json"
echo "================================================================================"
