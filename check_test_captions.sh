#!/bin/bash
# Quick script to check test caption results

JOB_ID="18501892"

echo "================================================================================"
echo "                    CHECKING TEST CAPTION JOB $JOB_ID"
echo "================================================================================"
echo ""

# Check if job is still running
echo "1️⃣  Job Status:"
squeue -u $USER -j $JOB_ID 2>/dev/null || echo "   Job completed or not found"
echo ""

# Show last 30 lines of output (shows progress and sample captions)
echo "2️⃣  Latest Output (last 30 lines):"
echo "--------------------------------------------------------------------------------"
tail -30 ~/out_slurm/test_caption_${JOB_ID}.out 2>/dev/null || echo "   No output yet"
echo ""

# Check for errors
echo "3️⃣  Error Log:"
echo "--------------------------------------------------------------------------------"
if [ -f ~/err_slurm/test_caption_${JOB_ID}.err ]; then
    ERROR_SIZE=$(wc -l < ~/err_slurm/test_caption_${JOB_ID}.err)
    if [ "$ERROR_SIZE" -gt 10 ]; then
        echo "   ⚠️  Found $ERROR_SIZE lines of errors/warnings. Last 20 lines:"
        tail -20 ~/err_slurm/test_caption_${JOB_ID}.err
    elif [ "$ERROR_SIZE" -gt 0 ]; then
        echo "   ⚠️  Found $ERROR_SIZE lines:"
        cat ~/err_slurm/test_caption_${JOB_ID}.err
    else
        echo "   ✅ No errors"
    fi
else
    echo "   No error file yet"
fi
echo ""

# Check if caption files were generated
echo "4️⃣  Generated Caption Files:"
echo "--------------------------------------------------------------------------------"
if [ -f results/karpathy_evaluation/captioning_test/stage2_captions.json ]; then
    STAGE2_COUNT=$(grep -c "image_id" results/karpathy_evaluation/captioning_test/stage2_captions.json 2>/dev/null || echo "0")
    echo "   ✅ Stage 2: $STAGE2_COUNT captions generated"
    echo ""
    echo "   📝 Stage 2 Sample Captions (first 3):"
    python3 -c "
import json
with open('results/karpathy_evaluation/captioning_test/stage2_captions.json') as f:
    data = json.load(f)
    for i, item in enumerate(data[:3]):
        print(f'      {i+1}. Image {item[\"image_id\"]}: {item[\"caption\"]}')
" 2>/dev/null || echo "      (Unable to parse JSON)"
else
    echo "   ⏳ Stage 2 captions not generated yet"
fi
echo ""

if [ -f results/karpathy_evaluation/captioning_test/stage3_captions.json ]; then
    STAGE3_COUNT=$(grep -c "image_id" results/karpathy_evaluation/captioning_test/stage3_captions.json 2>/dev/null || echo "0")
    echo "   ✅ Stage 3: $STAGE3_COUNT captions generated"
    echo ""
    echo "   📝 Stage 3 Sample Captions (first 3):"
    python3 -c "
import json
with open('results/karpathy_evaluation/captioning_test/stage3_captions.json') as f:
    data = json.load(f)
    for i, item in enumerate(data[:3]):
        print(f'      {i+1}. Image {item[\"image_id\"]}: {item[\"caption\"]}')
" 2>/dev/null || echo "      (Unable to parse JSON)"
else
    echo "   ⏳ Stage 3 captions not generated yet"
fi
echo ""

echo "================================================================================"
echo "                              QUICK COMMANDS"
echo "================================================================================"
echo "Watch output live:        tail -f ~/out_slurm/test_caption_${JOB_ID}.out"
echo "Watch errors live:        tail -f ~/err_slurm/test_caption_${JOB_ID}.err"
echo "View full output:         less ~/out_slurm/test_caption_${JOB_ID}.out"
echo "Check all Stage 2 caps:   jq '.[].caption' results/karpathy_evaluation/captioning_test/stage2_captions.json | head -20"
echo "Check all Stage 3 caps:   jq '.[].caption' results/karpathy_evaluation/captioning_test/stage3_captions.json | head -20"
echo "Re-run this check:        bash check_test_captions.sh"
echo "================================================================================"
