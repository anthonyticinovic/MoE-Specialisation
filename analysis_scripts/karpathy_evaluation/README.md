# Karpathy COCO Evaluation Pipeline

This directory contains scripts for evaluating Stage 2 and Stage 3 models on the Karpathy COCO test split.

## Overview

The pipeline performs two types of evaluation:
1. **Image-Text Retrieval**: Computes R@1, R@5, R@10 for both I2T and T2I
2. **Image Captioning**: Computes BLEU, CIDEr, METEOR, SPICE, ROUGE-L

## Files

### Evaluation Scripts
- `karpathy_utils.py`: Shared utility functions (model loading, preprocessing, embedding extraction)
- `01_preprocess_karpathy.py`: Parse Karpathy split and prepare test set
- `02_extract_embeddings.py`: Extract image and text embeddings from Layer 31
- `03_evaluate_retrieval.py`: Compute retrieval metrics from embeddings
- `04_generate_captions.py`: Generate captions using beam search
- `05_evaluate_captioning.py`: Evaluate captions with COCO metrics
- `06_visualize_results.py`: Create plots and comprehensive report

### SLURM Job
- `../../hpc/analysis_scripts/evaluate_karpathy_full.sbatch`: Master script that runs all steps

## Quick Start

### Run Full Pipeline

```bash
sbatch hpc/analysis_scripts/evaluate_karpathy_full.sbatch
```

This will:
1. Preprocess Karpathy split (~1 min)
2. Extract embeddings for Stage 2 and Stage 3 (~10 min each)
3. Evaluate retrieval metrics (~5 min)
4. Generate captions for Stage 2 and Stage 3 (~2 hours each)
5. Evaluate captioning metrics (~10 min)
6. Generate visualizations and report (~1 min)

**Total runtime**: ~5 hours on H100 GPU

### Run Individual Steps

You can also run scripts individually for debugging or re-analysis:

```bash
# Step 1: Preprocess
python analysis_scripts/karpathy_evaluation/01_preprocess_karpathy.py \
    --karpathy_json /data/.../coco/karpathy_split/dataset_coco.json \
    --output_dir results/karpathy_evaluation

# Step 2: Extract embeddings
python analysis_scripts/karpathy_evaluation/02_extract_embeddings.py \
    --checkpoint_path /data/.../stage2_checkpoints/llm_stage2_best.pth \
    --stage_name stage2 \
    --retrieval_json results/karpathy_evaluation/karpathy_test_retrieval.json \
    --image_base_dir /data/.../coco \
    --output_dir results/karpathy_evaluation/retrieval

# Step 3: Evaluate retrieval
python analysis_scripts/karpathy_evaluation/03_evaluate_retrieval.py \
    --embeddings_dir results/karpathy_evaluation/retrieval \
    --output_dir results/karpathy_evaluation/retrieval

# Step 4: Generate captions
python analysis_scripts/karpathy_evaluation/04_generate_captions.py \
    --checkpoint_path /data/.../stage2_checkpoints/llm_stage2_best.pth \
    --stage_name stage2 \
    --images_json results/karpathy_evaluation/karpathy_test_images.json \
    --image_base_dir /data/.../coco \
    --output_dir results/karpathy_evaluation/captioning \
    --num_beams 5 \
    --max_length 20

# Step 5: Evaluate captions
python analysis_scripts/karpathy_evaluation/05_evaluate_captioning.py \
    --references_json results/karpathy_evaluation/karpathy_test_references.json \
    --stage2_captions results/karpathy_evaluation/captioning/stage2_captions.json \
    --stage3_captions results/karpathy_evaluation/captioning/stage3_captions.json \
    --output_dir results/karpathy_evaluation/captioning

# Step 6: Visualize
python analysis_scripts/karpathy_evaluation/06_visualize_results.py \
    --retrieval_metrics results/karpathy_evaluation/retrieval/retrieval_metrics.json \
    --captioning_metrics results/karpathy_evaluation/captioning/captioning_metrics.json \
    --output_dir results/karpathy_evaluation
```

## Output Structure

```
results/karpathy_evaluation/
├── karpathy_test_retrieval.json          # 5K images + 25K captions
├── karpathy_test_images.json             # 5K image entries
├── karpathy_test_references.json         # COCO-format references
├── retrieval/
│   ├── stage2_image_embeddings.npy       # (5000, D) embeddings
│   ├── stage2_text_embeddings.npy        # (25000, D) embeddings
│   ├── stage3_image_embeddings.npy
│   ├── stage3_text_embeddings.npy
│   ├── retrieval_metrics.json            # R@1/5/10 for I2T and T2I
│   └── retrieval_comparison.txt          # Formatted table
├── captioning/
│   ├── stage2_captions.json              # Generated captions
│   ├── stage3_captions.json
│   ├── captioning_metrics.json           # BLEU, CIDEr, etc.
│   └── captioning_comparison.txt         # Formatted table
├── retrieval_comparison.png              # Bar charts
├── captioning_comparison.png
├── combined_comparison.png               # Full comparison
└── evaluation_report.txt                 # Comprehensive report
```

## Dependencies

Required packages (already installed in `pytorch_latest_venv`):
- torch
- torchvision
- transformers
- numpy
- matplotlib
- seaborn
- tqdm
- pycocoevalcap
- pycocotools
- nltk

## Dataset

Requires:
- COCO 2014 images: `/data/.../coco/train2014/`, `/data/.../coco/val2014/`
- Karpathy split JSON: `/data/.../coco/karpathy_split/dataset_coco.json`

## Notes

- **Layer extraction**: Uses Layer 31 (final layer) for consistency with compositional analysis
- **Pooling**: Mean pooling across sequence dimension for both images and text
- **Generation**: Beam search with 5 beams, max length 20 tokens
- **Retrieval**: Cosine similarity between normalized embeddings
- **Metrics**: Standard COCO evaluation metrics (via pycocoevalcap)

## Expected Results

Typical ranges for COCO models:
- **Retrieval R@1**: 20-60% (depending on model size and training)
- **CIDEr**: 0.5-1.2 (primary captioning metric)
- **BLEU-4**: 0.2-0.4

Results will enable:
1. Comparison between Stage 2 (hard routing) and Stage 3 (soft routing)
2. Benchmarking against published baselines (CLIP, BLIP, ALBEF, etc.)
3. Validation of compositional analysis findings
4. Publication-ready metrics for thesis

## Troubleshooting

### Missing images
If you see warnings about missing images, check:
- Image paths are relative to `--image_base_dir`
- COCO 2014 images are in correct locations
- File permissions are correct

### CUDA out of memory
Reduce batch sizes:
- `--batch_size 16` for embeddings (default: 32)
- `--batch_size 8` for caption generation (default: 16)

### pycocoevalcap errors
Ensure NLTK data is downloaded:
```python
import nltk
nltk.download('wordnet')
nltk.download('punkt')
```

## Citation

Karpathy split reference:
```
@inproceedings{karpathy2015deep,
  title={Deep visual-semantic alignments for generating image descriptions},
  author={Karpathy, Andrej and Fei-Fei, Li},
  booktitle={CVPR},
  year={2015}
}
```
