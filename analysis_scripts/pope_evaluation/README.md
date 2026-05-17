# POPE Evaluation for MoE-Specialisation

POPE (Polling-based Object Probing Evaluation) evaluates object hallucination in vision-language models using yes/no questions.

## Overview

**Purpose**: Measure how often the model "hallucinates" (claims to see) objects that aren't in images.

**Difficulty Levels**:
- **Random**: Negative examples randomly sampled from all objects
- **Popular**: Negative examples from frequently occurring objects (harder)
- **Adversarial**: Negative examples from objects that co-occur with image objects (hardest)

**Metrics**:
- **Accuracy**: Overall correctness
- **Precision**: Of predicted "yes", how many are correct?
- **Recall**: Of true "yes", how many did we catch?
- **F1**: Harmonic mean of precision and recall
- **Yes Ratio**: Proportion of "yes" answers (lower is better - indicates less over-generation)

## Workflow

### Step 1: Generate Questions (CPU, ~2 minutes)

Generates yes/no questions from COCO val2017 annotations:

```bash
sbatch hpc/analysis_scripts/pope_01_generate_questions.sbatch
```

**Output**:
- `results/pope_evaluation/pope_random.json` (~3000 questions)
- `results/pope_evaluation/pope_popular.json` (~3000 questions)
- `results/pope_evaluation/pope_adversarial.json` (~3000 questions)

### Step 2: Generate Stage 2 Answers (GPU, ~30 minutes)

Generates yes/no answers using Stage 2 model:

```bash
sbatch hpc/analysis_scripts/pope_02_generate_stage2_answers.sbatch
```

**Output**:
- `results/pope_evaluation/stage2_random_answers.json`
- `results/pope_evaluation/stage2_popular_answers.json`
- `results/pope_evaluation/stage2_adversarial_answers.json`

### Step 3: Generate Stage 3 Answers (GPU, ~30 minutes)

Generates yes/no answers using Stage 3 model:

```bash
sbatch hpc/analysis_scripts/pope_03_generate_stage3_answers.sbatch
```

**Output**:
- `results/pope_evaluation/stage3_random_answers.json`
- `results/pope_evaluation/stage3_popular_answers.json`
- `results/pope_evaluation/stage3_adversarial_answers.json`

### Step 4: Evaluate and Compare (CPU, ~1 minute)

Computes metrics and creates comparison:

```bash
sbatch hpc/analysis_scripts/pope_04_evaluate.sbatch
```

**Output**:
- `results/pope_evaluation/pope_metrics.json` - All metrics in JSON format
- `results/pope_evaluation/pope_comparison.txt` - Stage 2 vs Stage 3 comparison table

## Quick Start (Full Pipeline)

```bash
# 1. Generate questions
sbatch hpc/analysis_scripts/pope_01_generate_questions.sbatch

# Wait for completion (~2 minutes), then:

# 2. Generate Stage 2 answers
sbatch hpc/analysis_scripts/pope_02_generate_stage2_answers.sbatch

# 3. Generate Stage 3 answers (can run in parallel with Step 2)
sbatch hpc/analysis_scripts/pope_03_generate_stage3_answers.sbatch

# Wait for both to complete (~30 minutes), then:

# 4. Evaluate
sbatch hpc/analysis_scripts/pope_04_evaluate.sbatch
```

## Expected Results

**Interpretation**:
- **High Accuracy**: Model correctly identifies presence/absence of objects
- **High Precision**: Few hallucinations (false positives)
- **High Recall**: Doesn't miss objects (false negatives)
- **Low Yes Ratio**: Conservative about claiming objects exist
- **Adversarial is hardest**: Tests co-occurring objects (e.g., "Is there a knife?" when image has fork)

**Typical Performance**:
- Strong models: 85-90% accuracy, 80-85% F1
- Weak models: 70-80% accuracy, high yes ratio (over-generates)

## File Structure

```
analysis_scripts/pope_evaluation/
├── 01_generate_pope_questions.py    # Question generation
├── 02_generate_pope_answers.py      # Answer generation (+ optional --use-priming)
├── 03_evaluate_pope.py              # Evaluation & comparison
├── compare_priming_strategies.py    # Compare priming strategies
└── pope_utils.py                    # Shared extractors + metrics

hpc/analysis_scripts/
├── pope_01_generate_questions.sbatch
├── pope_02_generate_stage2_answers.sbatch
├── pope_03_generate_stage3_answers.sbatch
└── pope_04_evaluate.sbatch

results/pope_evaluation/
├── pope_random.json                  # Questions
├── pope_popular.json
├── pope_adversarial.json
├── stage2_random_answers.json        # Stage 2 answers
├── stage2_popular_answers.json
├── stage2_adversarial_answers.json
├── stage3_random_answers.json        # Stage 3 answers
├── stage3_popular_answers.json
├── stage3_adversarial_answers.json
├── pope_metrics.json                 # Final metrics
└── pope_comparison.txt               # Comparison table
```

## Customization

### Adjust number of questions:

Edit `pope_01_generate_questions.sbatch`:
```bash
--num_images 500              # Number of images (default: 500)
--questions_per_image 3       # Yes/no pairs per difficulty (default: 3)
```

### Change model temperature:

Edit `pope_02_generate_stage2_answers.sbatch` or `pope_03_generate_stage3_answers.sbatch`:
```bash
--temperature 0.0     # Greedy (default)
--temperature 0.7     # More diverse answers
```

### Stage-3 priming strategy (optional):

The priming experiment (previously a separate `02b` script) is now a flag on
`02_generate_pope_answers.py`. Default behaviour is unchanged; opt in with:

```bash
python 02_generate_pope_answers.py \
    --checkpoint_path /path/to/stage3_best.pth --stage_name stage3 \
    --questions_file results/pope_evaluation/pope_random.json \
    --use-priming --priming simple        # or: conversational | none
```

Then compare strategies (writes nothing; prints a table):

```bash
python compare_priming_strategies.py --results-dir results/pope_evaluation
```

## Troubleshooting

**Issue**: "Image not found" errors
- **Solution**: Check COCO val2017 images exist at `/data/gpfs/projects/COMP90055/aticinovic/datasets/coco/val2017/`

**Issue**: High "unclear" count in answers
- **Solution**: Model may not be generating clear yes/no responses. Check raw_output field in answer JSON.

**Issue**: Out of memory
- **Solution**: Batch size is already 1 for yes/no QA. Reduce max_new_tokens if needed.

## References

- **POPE Paper**: Li et al. "Evaluating Object Hallucination in Large Vision-Language Models" (EMNLP 2023)
- **COCO Dataset**: Lin et al. "Microsoft COCO: Common Objects in Context" (ECCV 2014)
