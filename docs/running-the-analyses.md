# Running the analyses

Every command here needs a checkpoint you have trained (Stages 0→3 — see the
[README](../README.md)) and the paths filled in in `configs/training_config.yaml`
or the relevant JSON config in `configs/`.

Run from the repository root. Paths resolve through `MOE_CONFIG` if it is set,
otherwise `configs/training_config.yaml`; device defaults to CUDA when it is
available and CPU otherwise. Output goes to `results/`, which is git-ignored.

Each evaluation sub-pipeline also has its own README with the full loop and the
expected outputs: [`pope_evaluation/`](../analysis_scripts/pope_evaluation/README.md),
[`karpathy_evaluation/`](../analysis_scripts/karpathy_evaluation/README.md),
[`llava_evaluation/`](../analysis_scripts/llava_evaluation/README.md).

---

## Expert routing and specialisation

```bash
# Routing ablation: compare normal vs. flipped routing to verify specialisation.
# This is the check the CPU demo runs on every push — see `make demo`.
python analysis_scripts/routing_ablation_experiment.py \
    --checkpoint /path/to/stage2_best.pth \
    --data       /path/to/coco

# Expert utilisation metrics across epochs (reads the JSON files Stage 3 writes)
python analysis_scripts/plot_expert_metrics.py \
    --metrics_dir /path/to/outputs/expert_metrics \
    --layers all_layers
```

`make figures` is a shortcut for the second command, reading the committed
metrics in [`paper_metrics/`](../paper_metrics/README.md) instead of a run of
your own — once those are added.

## Concept-level analysis

```bash
# Cross-concept similarity matrix (2N×2N image-text similarity at each layer)
python analysis_scripts/cross_concept_similarity_matrix.py \
    --config-file configs/similarity_matrix.json \
    --mode stage2   # or stage3

# Cross-modality purity (how separable are expert representations per concept?)
python analysis_scripts/cross_modality_purity.py \
    --concepts dog cat car bus \
    --layers 0 8 16 24 31

# Layer-wise clustering of expert activations
python analysis_scripts/layer_clustering_analysis.py \
    --config configs/clustering_analysis.json

# Compositional case study (colour-object binding)
python analysis_scripts/compositional_case_study.py \
    --config-file configs/compositional_case_study.json

# Stage 2 vs Stage 3 similarity matrix comparison plot
python analysis_scripts/create_stage_comparison.py \
    --stage2-dir results/similarity_matrix/stage2 \
    --stage3-dir results/similarity_matrix/stage3
```

## Benchmark evaluation

### POPE (object hallucination)

```bash
# Generates pope_{random,popular,adversarial}.json into the output dir.
python analysis_scripts/pope_evaluation/01_generate_pope_questions.py \
    --annotations_file /path/to/coco/annotations/instances_val2017.json \
    --output_dir       results/pope_evaluation

# Run once per difficulty (see the POPE README for the full loop).
python analysis_scripts/pope_evaluation/02_generate_pope_answers.py \
    --questions_file  results/pope_evaluation/pope_random.json \
    --image_dir       /path/to/coco/val2017 \
    --checkpoint_path /path/to/checkpoint.pth \
    --output_dir      results/pope_evaluation

python analysis_scripts/pope_evaluation/03_evaluate_pope.py \
    --stage2_dir results/pope_evaluation \
    --output_dir results/pope_evaluation
```

### Karpathy COCO split (retrieval + captioning)

```bash
# Preprocess Karpathy split JSON
python analysis_scripts/karpathy_evaluation/01_preprocess_karpathy.py \
    --karpathy_json /path/to/dataset_coco.json

# Extract embeddings for retrieval
python analysis_scripts/karpathy_evaluation/02_extract_embeddings.py \
    --image_base_dir /path/to/coco \
    --checkpoint_path /path/to/checkpoint.pth

# Evaluate retrieval (R@1, R@5, R@10)
python analysis_scripts/karpathy_evaluation/03_evaluate_retrieval.py

# Generate captions
python analysis_scripts/karpathy_evaluation/04_generate_captions.py \
    --image_base_dir /path/to/coco \
    --checkpoint_path /path/to/checkpoint.pth

# Score captions (CIDEr, BLEU, METEOR, ROUGE)
python analysis_scripts/karpathy_evaluation/05_evaluate_captioning.py

# Visualise results
python analysis_scripts/karpathy_evaluation/06_visualize_results.py
```

### LLaVA-Wild (instruction following)

```bash
python analysis_scripts/llava_evaluation/01_llava_wild_eval.py \
    --checkpoint /path/to/checkpoint.pth

python analysis_scripts/llava_evaluation/02_compare_results.py \
    --stage2 results/llava_wild/stage2 \
    --stage3 results/llava_wild/stage3
```

`02_compare_results.py` pairs the two runs **by position**, so it refuses to run
if they evaluated different numbers of samples. Re-run the shorter side rather
than comparing across a mismatch.
