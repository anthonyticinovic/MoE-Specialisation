#!/bin/bash
#SBATCH --job-name=plot_expert_metrics
#SBATCH --account=COMP90055
#SBATCH --partition=sapphire
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/aticinovic/out_slurm/plot_expert_metrics_%j.out
#SBATCH --error=/home/aticinovic/err_slurm/plot_expert_metrics_%j.err

# ============================================================================
# EXPERT METRICS VISUALIZATION
# ============================================================================
# Generates plots from Stage 3 expert utilization metrics JSON files
#
# Usage:
#   sbatch hpc/analysis_scripts/plot_expert_metrics.sbatch
#
# Outputs:
#   - results/expert_metrics/expert_load_distribution.png
#   - results/expert_metrics/routing_entropy.png
#   - results/expert_metrics/high_confidence_fraction.png
#   - results/expert_metrics/visual_vs_text_routing.png
#   - results/expert_metrics/specialization_evolution.png
#   - results/expert_metrics/aggregate_summary.png
#   - results/expert_metrics/expert_metrics_report.txt
# ============================================================================

echo "========================================"
echo "Expert Metrics Visualization"
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "========================================"

# Load modules
module purge
module load GCCcore/11.3.0
module load Python/3.11.3

# Activate virtual environment
source ~/pytorch_latest_venv/bin/activate

# Set working directory
cd /home/aticinovic/MoE-Specialisation

# Export PYTHONPATH to ensure imports work
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Define paths
METRICS_DIR="/data/gpfs/projects/COMP90055/aticinovic/outputs/expert_metrics"
OUTPUT_DIR="results/expert_metrics"

# Layer selection (optional: modify to select different layers)
# Default: 0 7 15 23 31 (evenly spaced across 32 layers)
# Example custom selection: LAYERS="0 5 10 15 20 25 31"
LAYERS="0 7 15 23 31"

echo ""
echo "Configuration:"
echo "  Metrics directory: $METRICS_DIR"
echo "  Output directory:  $OUTPUT_DIR"
echo "  Selected layers:   $LAYERS"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

python analysis_scripts/plot_expert_metrics.py \
    --metrics_dir "$METRICS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --layers all_layers \
    --epochs 4

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Visualization completed successfully!"
else
    echo "❌ Visualization failed with exit code: $EXIT_CODE"
fi
echo "Job completed at: $(date)"
echo "========================================"

exit $EXIT_CODE
