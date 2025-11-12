#!/bin/bash
#SBATCH --job-name=coco_feasibility
#SBATCH -p sapphire
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=../out_slurm/coco_feasibility_%j.out
#SBATCH --error=../err_slurm/coco_feasibility_%j.err

# Load modules
module purge
module load GCCcore/11.3.0
module load Python/3.11.3

source ~/pytorch_latest_venv/bin/activate

cd $HOME/MoE-Specialisation
export PYTHONPATH="${PWD}:${PYTHONPATH}"

echo "=== COCO Compositional Concept Feasibility Test ==="
echo "Testing: red_apple, green_apple, red_cherry, cherry, green_pear, pear"
echo "Checking if automated sampling from COCO is feasible for N=50 samples/concept"
echo ""

# Path to COCO annotations (adjust if needed)
COCO_ANNOTATIONS="/data/gpfs/projects/COMP90055/aticinovic/datasets/coco/annotations/captions_train2017.json"

python -u analysis_scripts/coco_feasibility_test.py "$COCO_ANNOTATIONS"

echo ""
echo "=== Test Complete ==="
echo "Check output above for:"
echo "  - How many captions match each compositional concept"
echo "  - Whether N≥50 is feasible with strict matching"
echo "  - Sample captions for each concept"
echo "  - Recommendations for next steps"
