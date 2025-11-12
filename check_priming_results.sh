#!/bin/bash
# Quick check script for priming experiment results

echo "Checking POPE Priming Experiment Status..."
echo "=========================================="
echo ""

# Check job status
JOB_ID=18548942
JOB_STATUS=$(squeue -j $JOB_ID 2>/dev/null | tail -1)

if [[ -z "$JOB_STATUS" || "$JOB_STATUS" == *"JOBID"* ]]; then
    echo "✅ Job $JOB_ID has completed"
    echo ""
else
    echo "⏳ Job $JOB_ID is still running:"
    squeue -j $JOB_ID
    echo ""
    echo "Check back later..."
    exit 0
fi

# Check output files
ANSWERS_DIR="/home/aticinovic/MoE-Specialisation/results/pope_evaluation/answers_primed"
echo "Checking generated files:"
ls -lh "$ANSWERS_DIR"/*.json 2>/dev/null | wc -l | xargs echo "   Files found:"

if [ -f "$ANSWERS_DIR/stage3_random_simple.json" ]; then
    echo ""
    echo "✅ Files generated! Running comparison analysis..."
    echo ""
    
    cd /home/aticinovic/MoE-Specialisation
    python analysis_scripts/pope_evaluation/compare_priming_strategies.py
else
    echo ""
    echo "❌ No output files found. Check error log:"
    echo "   ~/err_slurm/pope_stage3_primed_18548942.err"
fi
