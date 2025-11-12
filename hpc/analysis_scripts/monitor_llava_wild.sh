#!/bin/bash
# Quick monitoring script for LLaVA-Wild evaluation

JOB_ID=18554637
OUT_FILE="/home/aticinovic/MoE-Specialisation/out_slurm/llava_wild_full_${JOB_ID}.out"
ERR_FILE="/home/aticinovic/MoE-Specialisation/err_slurm/llava_wild_full_${JOB_ID}.err"

echo "=========================================="
echo "LLaVA-Wild Evaluation Monitor"
echo "Job ID: ${JOB_ID}"
echo "=========================================="

# Check job status
echo ""
echo "📊 Job Status:"
squeue -u aticinovic | grep -E "JOBID|${JOB_ID}" || echo "Job not found (may be completed or pending)"

# Check output file
echo ""
echo "📄 Latest Output (last 30 lines):"
echo "----------------------------------------"
if [ -f "${OUT_FILE}" ]; then
    tail -n 30 "${OUT_FILE}"
else
    echo "Output file not yet created"
fi

# Check for errors
echo ""
echo "⚠️  Error Check:"
echo "----------------------------------------"
if [ -f "${ERR_FILE}" ]; then
    ERROR_COUNT=$(wc -l < "${ERR_FILE}")
    if [ ${ERROR_COUNT} -gt 0 ]; then
        echo "Found ${ERROR_COUNT} lines in error log (last 10):"
        tail -n 10 "${ERR_FILE}"
    else
        echo "No errors logged"
    fi
else
    echo "Error file not yet created"
fi

# Check for completion
echo ""
echo "✅ Completion Check:"
echo "----------------------------------------"
if [ -f "${OUT_FILE}" ]; then
    if grep -q "FULL EVALUATION COMPLETE" "${OUT_FILE}"; then
        echo "✅ Evaluation COMPLETE!"
        
        # Show results summary
        echo ""
        echo "📊 Results Summary:"
        RESULTS_DIR="/home/aticinovic/MoE-Specialisation/results/llava_wild_evaluation"
        
        if [ -f "${RESULTS_DIR}/stage2_results.json" ]; then
            S2_SCORE=$(python3 -c "import json; data=json.load(open('${RESULTS_DIR}/stage2_results.json')); print(f\"{data['summary']['average_score']:.1f}\")")
            echo "  Stage 2: ${S2_SCORE}/100"
        fi
        
        if [ -f "${RESULTS_DIR}/stage3_results.json" ]; then
            S3_SCORE=$(python3 -c "import json; data=json.load(open('${RESULTS_DIR}/stage3_results.json')); print(f\"{data['summary']['average_score']:.1f}\")")
            echo "  Stage 3: ${S3_SCORE}/100"
        fi
        
        if [ -f "${RESULTS_DIR}/llava_wild_comparison.png" ]; then
            echo "  Visualization: ✅ Generated"
        fi
    else
        echo "⏳ Still running..."
        
        # Try to detect current step
        if grep -q "Evaluating Stage 2" "${OUT_FILE}"; then
            echo "  Current: Stage 2 evaluation"
        elif grep -q "Evaluating Stage 3" "${OUT_FILE}"; then
            echo "  Current: Stage 3 evaluation"
        elif grep -q "Comparing results" "${OUT_FILE}"; then
            echo "  Current: Comparison"
        fi
    fi
else
    echo "⏳ Job pending or just started"
fi

echo ""
echo "=========================================="
echo "Commands:"
echo "  Watch live:  tail -f ${OUT_FILE}"
echo "  Check again: bash $0"
echo "=========================================="
