#!/usr/bin/env python3
"""
Standalone caption evaluation script - no torch required.
Evaluates Stage 2 and Stage 3 captions using COCO metrics.
"""

import json
from pathlib import Path
import sys
import os

# Change to script directory
os.chdir(Path(__file__).parent)

# Try to import from pip-installed version first, fallback to local coco-caption
try:
    from pycocoevalcap.eval import COCOEvalCap
    from pycocotools.coco import COCO
except ImportError:
    print("Using local coco-caption directory...")
    sys.path.insert(0, str(Path(__file__).parent / 'coco-caption'))
    from pycocoevalcap.eval import COCOEvalCap
    from pycocotools.coco import COCO


def print_banner(text):
    """Print a formatted banner."""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")


def evaluate_captions(references_json, generated_json):
    """Evaluate generated captions against references."""
    print(f"\n📊 Evaluating captions...")
    print(f"   References: {references_json}")
    print(f"   Generated: {generated_json}")
    
    # Load references
    coco = COCO(references_json)
    
    # Add 'info' field if missing (required by loadRes)
    if 'info' not in coco.dataset:
        coco.dataset['info'] = {
            'description': 'Karpathy COCO test split',
            'version': '1.0',
            'year': 2024
        }
    
    # Load generated captions
    coco_res = coco.loadRes(generated_json)
    
    # Run evaluation (skip SPICE due to Java compatibility issues)
    coco_eval = COCOEvalCap(coco, coco_res)
    # Remove SPICE scorer which requires Java 8
    coco_eval.params['image_id'] = coco.getImgIds()
    try:
        # Try full evaluation first
        coco_eval.evaluate()
    except:
        # If SPICE fails, evaluate without it
        print("   ⚠️  SPICE evaluation failed (Java compatibility), computing other metrics...")
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider
        
        # Get ground truth and results
        gts = {}
        res = {}
        imgIds = coco.getImgIds()
        for imgId in imgIds:
            gts[imgId] = coco.imgToAnns[imgId]
            res[imgId] = coco_res.imgToAnns[imgId]
        
        # Tokenize
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)
        
        # Compute each metric
        coco_eval.eval = {}
        for scorer, method in [(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                               (Meteor(),"METEOR"),
                               (Rouge(), "ROUGE_L"),
                               (Cider(), "CIDEr")]:
            score, scores = scorer.compute_score(gts, res)
            if isinstance(method, list):
                for sc, m in zip(score, method):
                    coco_eval.eval[m] = sc
            else:
                coco_eval.eval[method] = score
    
    # Extract metrics
    metrics = {}
    for metric, score in coco_eval.eval.items():
        metrics[metric] = float(score)
        print(f"   {metric}: {score:.4f}")
    
    return metrics


def format_comparison_table(stage2_metrics, stage3_metrics):
    """Format a comparison table for Stage 2 vs Stage 3."""
    
    lines = []
    lines.append("\n" + "="*80)
    lines.append("CAPTIONING METRICS COMPARISON: STAGE 2 vs STAGE 3")
    lines.append("="*80)
    
    lines.append("\nAll Metrics:")
    lines.append("-" * 80)
    lines.append(f"{'Metric':<20} {'Stage 2':<15} {'Stage 3':<15} {'Δ (S3-S2)':<15}")
    lines.append("-" * 80)
    
    # Order metrics for display
    metric_order = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE']
    
    for metric in metric_order:
        if metric in stage2_metrics and metric in stage3_metrics:
            s2_val = stage2_metrics[metric]
            s3_val = stage3_metrics[metric]
            delta = s3_val - s2_val
            delta_str = f"{delta:+.4f}"
            
            lines.append(f"{metric:<20} {s2_val:>7.4f}        {s3_val:>7.4f}        {delta_str:>10}")
    
    lines.append("="*80)
    
    # Summary
    lines.append("\nKey Observations:")
    lines.append("-" * 80)
    
    # CIDEr is the most important metric for image captioning
    if 'CIDEr' in stage2_metrics and 'CIDEr' in stage3_metrics:
        cider_s2 = stage2_metrics['CIDEr']
        cider_s3 = stage3_metrics['CIDEr']
        if cider_s3 > cider_s2:
            winner = "Stage 3"
            margin = cider_s3 - cider_s2
        else:
            winner = "Stage 2"
            margin = cider_s2 - cider_s3
        
        lines.append(f"• CIDEr (primary metric): {winner} performs better by {margin:.4f}")
    
    # BLEU-4 is commonly reported
    if 'Bleu_4' in stage2_metrics and 'Bleu_4' in stage3_metrics:
        b4_s2 = stage2_metrics['Bleu_4']
        b4_s3 = stage3_metrics['Bleu_4']
        if b4_s3 > b4_s2:
            winner = "Stage 3"
            margin = b4_s3 - b4_s2
        else:
            winner = "Stage 2"
            margin = b4_s2 - b4_s3
        
        lines.append(f"• BLEU-4: {winner} performs better by {margin:.4f}")
    
    # SPICE measures semantic understanding
    if 'SPICE' in stage2_metrics and 'SPICE' in stage3_metrics:
        spice_s2 = stage2_metrics['SPICE']
        spice_s3 = stage3_metrics['SPICE']
        if spice_s3 > spice_s2:
            winner = "Stage 3"
            margin = spice_s3 - spice_s2
        else:
            winner = "Stage 2"
            margin = spice_s2 - spice_s3
        
        lines.append(f"• SPICE (semantic): {winner} performs better by {margin:.4f}")
    
    lines.append("="*80)
    
    return "\n".join(lines)


# Main execution
print_banner("KARPATHY COCO CAPTIONING EVALUATION")

# File paths
refs_json = "results/karpathy_evaluation/karpathy_test_references.json"
stage2_json = "results/karpathy_evaluation/captioning/stage2_captions.json"
stage3_json = "results/karpathy_evaluation/captioning/stage3_captions.json"
output_dir = Path("results/karpathy_evaluation/captioning")

# Evaluate Stage 2
print("\n" + "="*80)
print("STAGE 2 EVALUATION")
print("="*80)
stage2_metrics = evaluate_captions(refs_json, stage2_json)

# Evaluate Stage 3
print("\n" + "="*80)
print("STAGE 3 EVALUATION")
print("="*80)
stage3_metrics = evaluate_captions(refs_json, stage3_json)

# Save metrics
output_dir.mkdir(parents=True, exist_ok=True)

all_metrics = {
    'stage2': stage2_metrics,
    'stage3': stage3_metrics
}

metrics_path = output_dir / 'captioning_metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(all_metrics, f, indent=2)
print(f"\n💾 Saved metrics: {metrics_path}")

# Print comparison table
comparison = format_comparison_table(stage2_metrics, stage3_metrics)
print(comparison)

# Save comparison to text file
comparison_path = output_dir / 'captioning_comparison.txt'
with open(comparison_path, 'w') as f:
    f.write(comparison)
print(f"\n💾 Saved comparison: {comparison_path}")

print_banner("✅ CAPTIONING EVALUATION COMPLETE")
