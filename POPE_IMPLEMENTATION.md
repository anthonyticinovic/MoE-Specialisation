# POPE Evaluation Implementation Summary

## ✅ Implementation Complete

I've implemented a complete POPE (Polling-based Object Probing Evaluation) pipeline for evaluating object hallucination in your Stage 2 and Stage 3 models.

## 📁 Files Created

### Python Scripts
1. **`analysis_scripts/pope_evaluation/01_generate_pope_questions.py`**
   - Generates yes/no questions from COCO val2017 annotations
   - Creates 3 difficulty levels: random, popular, adversarial
   - ~3000 questions per difficulty level

2. **`analysis_scripts/pope_evaluation/02_generate_pope_answers.py`**
   - Generates model answers to POPE questions
   - Adapted from your caption generation code (`04_generate_captions.py`)
   - Handles Stage 2 (hard routing) and Stage 3 (soft routing) differences
   - Extracts yes/no from model outputs

3. **`analysis_scripts/pope_evaluation/03_evaluate_pope.py`**
   - Computes accuracy, precision, recall, F1, yes ratio
   - Creates Stage 2 vs Stage 3 comparison tables
   - Identifies hallucinations (false positives)

### SLURM Job Scripts
4. **`hpc/analysis_scripts/pope_01_generate_questions.sbatch`**
   - CPU job, ~2 minutes
   - No dependencies required

5. **`hpc/analysis_scripts/pope_02_generate_stage2_answers.sbatch`**
   - H100 GPU job, ~30 minutes
   - Generates answers for all 3 difficulties

6. **`hpc/analysis_scripts/pope_03_generate_stage3_answers.sbatch`**
   - H100 GPU job, ~30 minutes
   - Can run in parallel with Stage 2

7. **`hpc/analysis_scripts/pope_04_evaluate.sbatch`**
   - CPU job, ~1 minute
   - Computes final metrics and comparison

### Documentation
8. **`analysis_scripts/pope_evaluation/README.md`**
   - Complete usage guide
   - Workflow explanation
   - Troubleshooting tips

## 🚀 How to Run

```bash
# Step 1: Generate questions (CPU, ~2 min)
sbatch hpc/analysis_scripts/pope_01_generate_questions.sbatch
# Job 18542867 submitted ✅

# Step 2 & 3: Generate answers (GPU, ~30 min each, can run in parallel)
sbatch hpc/analysis_scripts/pope_02_generate_stage2_answers.sbatch
sbatch hpc/analysis_scripts/pope_03_generate_stage3_answers.sbatch

# Step 4: Evaluate (CPU, ~1 min)
sbatch hpc/analysis_scripts/pope_04_evaluate.sbatch
```

**Total time**: ~1 hour (30 min if Stage 2 & 3 run in parallel)

## 📊 Expected Outputs

**Questions** (after Step 1):
- `results/pope_evaluation/pope_random.json`
- `results/pope_evaluation/pope_popular.json`
- `results/pope_evaluation/pope_adversarial.json`

**Answers** (after Steps 2 & 3):
- `results/pope_evaluation/stage2_{difficulty}_answers.json` (×3)
- `results/pope_evaluation/stage3_{difficulty}_answers.json` (×3)

**Metrics** (after Step 4):
- `results/pope_evaluation/pope_metrics.json` - All metrics
- `results/pope_evaluation/pope_comparison.txt` - Comparison table

## 🎯 What POPE Measures

**Object Hallucination**: Does the model claim to see objects that aren't there?

**Metrics**:
- **Accuracy**: Overall correctness
- **Precision**: How many "yes" answers are correct? (low precision = hallucination)
- **Recall**: How many real objects were detected?
- **F1**: Balance between precision and recall
- **Yes Ratio**: Proportion of "yes" answers (high = overconfident/hallucinating)

**Difficulty Levels**:
- **Random**: Negative objects randomly sampled
- **Popular**: Negative objects are common (e.g., "person", "car")
- **Adversarial**: Negative objects co-occur with image objects (hardest!)

## 🔍 Key Implementation Details

### Adapted from Your Codebase
- Uses `karpathy_utils.py` for model loading (same as captioning)
- Follows routing patterns from `04_generate_captions.py`:
  * Stage 2: Hard routing with routing masks
  * Stage 3: Soft routing with gating network
- Uses existing COCO data paths and structure
- Follows your SLURM job conventions

### Answer Extraction
- Extracts yes/no from free-form model output
- Handles variations: "yes", "no", "yeah", "nope", "there is...", etc.
- Falls back to "unclear" if ambiguous

### Routing Handling
- **Stage 2**: Sets `routing_mode = 'hard'` and provides explicit routing masks
- **Stage 3**: Sets `routing_mode = 'soft'` and lets gating network decide

## 📈 Interpreting Results

**Good Performance**:
- Accuracy: 85-90%
- Precision: 80-85% (few hallucinations)
- F1: 80-85%
- Yes Ratio: ~50% (balanced)

**Signs of Hallucination**:
- Low precision (<70%)
- High yes ratio (>60%)
- Many false positives

**Expected Finding**:
- Adversarial difficulty will be hardest
- Stage 2 vs Stage 3 differences will reveal impact of DPO training

## ✅ Next Steps

1. **Wait for Job 18542867 to complete** (~2 minutes)
   - Check: `cat ~/out_slurm/pope_gen_questions_18542867.out`

2. **Submit answer generation jobs**
   - Stage 2 and Stage 3 can run in parallel

3. **Run evaluation**
   - Compare Stage 2 vs Stage 3 hallucination rates

4. **For your thesis**:
   - Include POPE metrics table
   - Discuss hallucination differences between stages
   - Compare with reported POPE scores from other VLMs

## 🐛 Verified Patterns

✅ Uses your exact checkpoint paths
✅ Uses your COCO data directory structure  
✅ Follows your model loading patterns
✅ Handles Stage 2/Stage 3 routing differences
✅ Uses your tokenizer configuration
✅ Follows your SLURM job conventions
✅ Creates output in `results/` directory

## 📚 Reference

POPE is a standard VQA benchmark for hallucination evaluation:
- Paper: Li et al. "Evaluating Object Hallucination in Large Vision-Language Models" (EMNLP 2023)
- Used by: LLaVA, InstructBLIP, MiniGPT-4, and other VLMs
