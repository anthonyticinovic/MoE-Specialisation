# LLaVA-Wild Style Evaluation

Lightweight evaluation testing whether Stage 3 performs better on **in-distribution** tasks (LLaVA-style conversational VQA) compared to **out-of-distribution** tasks (POPE, COCO captioning).

## Hypothesis

**Stage 3 learned the training distribution well but over-specialized:**
- ✅ Should perform reasonably on LLaVA-Wild (matches training format)
- ❌ Fails catastrophically on POPE (30% accuracy, 0% "no" predictions)
- ❌ Fails catastrophically on COCO captioning (0.08 CIDEr vs 0.76 for Stage 2)

## Evaluation Setup

### Dataset
- **Source**: LLaVA-Instruct-150K (same data used for Stage 3 training)
- **Subset**: 200 random samples (lightweight evaluation)
- **Format**: First question-answer pair from each conversation
- **Task**: Generate conversational response to visual question

### Evaluation Metric

Simple heuristic scoring (0-100):
1. **Length** (20-150 tokens ideal for conversational)
2. **Coherence** (low repetition, not "a laptop, a laptop, a laptop")
3. **Relevance** (contains descriptive content words)
4. **Quality** (not generic templates)

### Models Evaluated
- **Stage 2**: Hard routing + COCO caption training
- **Stage 3**: Soft routing + LLaVA instruction tuning

## Usage

### Quick Start (Full Pipeline)

Run all three steps (Stage 2 eval, Stage 3 eval, comparison):

```bash
cd /home/aticinovic/MoE-Specialisation
sbatch hpc/analysis_scripts/llava_wild_03_full_pipeline.sbatch
```

**Expected runtime**: ~1-2 hours (200 samples × 2 stages)

### Individual Steps

**Step 1: Evaluate Stage 2**
```bash
sbatch hpc/analysis_scripts/llava_wild_01_stage2.sbatch
```

**Step 2: Evaluate Stage 3**
```bash
sbatch hpc/analysis_scripts/llava_wild_02_stage3.sbatch
```

**Step 3: Compare Results**
```bash
python analysis_scripts/llava_evaluation/02_compare_results.py \
    --stage2 results/llava_wild_evaluation/stage2_results.json \
    --stage3 results/llava_wild_evaluation/stage3_results.json \
    --output_dir results/llava_wild_evaluation/
```

## Expected Results

### Scenario A: Stage 3 Outperforms Stage 2
- Stage 3 score: **50-70** (in-distribution advantage)
- Stage 2 score: **40-60** (generalizes reasonably)
- **Interpretation**: ✅ Stage 3 learned training distribution well but over-specialized

### Scenario B: Stage 3 Performs Similarly to Stage 2
- Both scores: **40-60** (similar performance)
- **Interpretation**: ⚠️ Stage 3 can handle in-distribution tasks but doesn't excel

### Scenario C: Stage 3 Underperforms Stage 2
- Stage 3 score: **<40** (worse than Stage 2)
- **Interpretation**: ❌ Stage 3's issues are more fundamental than expected

## Output Files

### `stage2_results.json`
```json
{
  "summary": {
    "stage": "stage2",
    "num_samples": 200,
    "average_score": 55.3
  },
  "results": [
    {
      "id": "sample_001",
      "image": "COCO_train2014_000000123456.jpg",
      "question": "What is the person doing in this image?",
      "reference_answer": "The person is riding a skateboard at a skate park.",
      "generated_answer": "A person is performing a trick on a skateboard.",
      "score": 75
    }
  ]
}
```

### `stage3_results.json`
Same format as Stage 2.

### `llava_wild_comparison.png`
Visualization with 4 subplots:
1. **Score Distribution**: Histogram comparing Stage 2 vs Stage 3
2. **Box Plot**: Statistical comparison (median, quartiles)
3. **All Benchmarks**: POPE, COCO, LLaVA-Wild performance comparison
4. **Per-Sample Scatter**: Stage 2 vs Stage 3 scores (above/below diagonal)

## Context: Full Benchmark Comparison

| Benchmark | Task Type | Stage 2 | Stage 3 | Change | Status |
|-----------|-----------|---------|---------|--------|--------|
| **POPE** | Yes/No (OOD) | 71.5% | 30.0% | -41.5% | ❌ Catastrophic failure |
| **COCO Captions** | Captioning (OOD) | 0.76 CIDEr | 0.08 CIDEr | -89.5% | ❌ Catastrophic failure |
| **LLaVA-Wild** | Conversational (ID) | TBD | TBD | TBD | ⏳ Testing now |

**Key Insight**: Stage 3 fails on out-of-distribution tasks but should work on in-distribution tasks (LLaVA-Wild).

## Root Cause Analysis

### Why Stage 3 Fails on POPE/COCO

**Training Data Structure**:
```
[Visual Tokens] Q1 A1 Q2 A2 Q3 A3
                ❌  ✅ ❌  ✅ ❌  ✅
                (masked questions, unmasked answers)
```

**Learned Pattern**:
- Model learned to generate sequential answers: A1 → A2 → A3
- Never saw questions during training (masked with label=-100)
- Cannot respond to single isolated questions (POPE format)

**Evidence**:
1. POPE: 0% "no" predictions (always generates descriptions)
2. COCO: Repetitive text ("a laptop, a laptop, and a computer")
3. Priming test: 97% unclear even with "fake previous answers"

### Why LLaVA-Wild Should Work

**Task Alignment**:
- ✅ Same format as training (conversational Q&A)
- ✅ Same context (elaborative answers, not yes/no)
- ✅ Same visual domain (COCO images)

**Expected Outcome**:
- Stage 3 should generate reasonable conversational responses
- May still have some repetition issues
- But should outperform or match Stage 2

## Thesis Implications

### If Stage 3 Outperforms Stage 2 on LLaVA-Wild:
**Narrative**: "Soft routing + instruction tuning caused over-specialization"
- Stage 3 learned training distribution extremely well
- But cannot generalize to other formats
- Trade-off: Specialization vs Generalization

### If Stage 3 Fails on LLaVA-Wild:
**Narrative**: "Stage 3's issues are more fundamental"
- Question masking broke instruction-following capability
- Even in-distribution tasks fail
- Need to revisit training methodology

## Files

```
analysis_scripts/llava_evaluation/
├── 01_llava_wild_eval.py          # Main evaluation script
├── 02_compare_results.py          # Comparison and visualization
└── README.md                      # This file

hpc/analysis_scripts/
├── llava_wild_01_stage2.sbatch    # Stage 2 job
├── llava_wild_02_stage3.sbatch    # Stage 3 job
└── llava_wild_03_full_pipeline.sbatch  # Complete pipeline

results/llava_wild_evaluation/
├── stage2_results.json            # Stage 2 evaluation results
├── stage3_results.json            # Stage 3 evaluation results
└── llava_wild_comparison.png      # Visualization
```

## Technical Details

### Generation Parameters
- **max_new_tokens**: 100 (conversational length)
- **temperature**: 0.7 (sampling for diversity)
- **device**: H100 GPU

### Evaluation Heuristic

**Score Components** (0-100 scale):
```python
base_score = 50

# Length (20-150 tokens ideal)
if 20 <= tokens <= 150:
    score += 20

# Coherence (>70% unique bigrams)
if unique_bigrams / total_bigrams > 0.7:
    score += 15

# Content (has descriptive words)
if contains_content_words:
    score += 15

# Generic penalty
if generic_template:
    score -= 5
```

**Quality Categories**:
- **Excellent** (80-100): Natural, diverse, descriptive
- **Good** (60-79): Reasonable with minor issues
- **Fair** (40-59): Acceptable but generic/repetitive
- **Poor** (<40): Broken, repetitive, or off-topic

## Limitations

1. **Heuristic Evaluation**: Not as reliable as GPT-4 or human judgments
2. **Small Subset**: 200 samples (vs full 150K)
3. **Single-Turn Only**: First Q&A pair (not full conversations)
4. **Same Seed**: Both models see same 200 samples

## Future Work

If time permits:
1. **Larger Subset**: Evaluate on 1000+ samples
2. **GPT-4 Evaluation**: Use GPT-4 to judge response quality
3. **Multi-Turn**: Test full conversation capability
4. **Other Benchmarks**: Try MMBench, SEED-Bench

## References

- **LLaVA Paper**: [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)
- **POPE**: [Evaluating Object Hallucination in Large Vision-Language Models](https://arxiv.org/abs/2305.10355)
- **Training Code**: `training_scripts/train_stage_3.py`
- **Data Loader**: `data/LLaVA_loader.py`
