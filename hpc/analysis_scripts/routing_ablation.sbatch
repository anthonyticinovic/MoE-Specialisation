#!/bin/bash
#SBATCH --job-name=routing_ablation
#SBATCH --output=/home/aticinovic/out_slurm/routing_ablation_%j.out
#SBATCH --error=/home/aticinovic/err_slurm/routing_ablation_%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Expert Routing Ablation Study for Stage 2
# Tests whether expert specialization is meaningful by comparing
# normal routing vs. flipped routing

module purge
module load GCCcore/11.3.0
module load Python/3.11.3
module load CUDA/11.7.0

# Activate virtual environment
source ~/pytorch_latest_venv/bin/activate

cd /data/gpfs/projects/COMP90055/aticinovic

echo "Starting Expert Routing Ablation Study"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo ""

# Run ablation experiment
python3 /home/aticinovic/MoE-Specialisation/analysis_scripts/routing_ablation_experiment.py \
    --checkpoint /data/gpfs/projects/COMP90055/aticinovic/outputs/stage2_checkpoints/llm_stage2_best.pth \
    --data /data/gpfs/projects/COMP90055/aticinovic/datasets/coco \
    --num_samples 150 \
    --device cuda

echo ""
echo "Experiment completed at $(date)"
