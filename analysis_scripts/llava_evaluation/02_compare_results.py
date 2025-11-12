"""
Compare LLaVA-Wild evaluation results between Stage 2 and Stage 3.

This analysis tests the hypothesis:
- Stage 3 performs BETTER on in-distribution tasks (LLaVA-style conversational VQA)
- Stage 3 performs WORSE on out-of-distribution tasks (POPE, COCO captioning)
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_results(json_path):
    """Load evaluation results."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def analyze_results(stage2_data, stage3_data):
    """
    Compare Stage 2 vs Stage 3 performance.
    """
    print("=" * 80)
    print("LLAVA-WILD EVALUATION COMPARISON")
    print("=" * 80)
    
    # Summary metrics
    s2_summary = stage2_data['summary']
    s3_summary = stage3_data['summary']
    
    print(f"\n📊 OVERALL SCORES:")
    print(f"{'Metric':<30} {'Stage 2':<15} {'Stage 3':<15} {'Difference':<15}")
    print("-" * 80)
    
    s2_score = s2_summary['average_score']
    s3_score = s3_summary['average_score']
    diff = s3_score - s2_score
    
    print(f"{'Average Score (0-100)':<30} {s2_score:<15.1f} {s3_score:<15.1f} {diff:+.1f}")
    print(f"{'Samples Evaluated':<30} {s2_summary['num_samples']:<15} {s3_summary['num_samples']:<15}")
    
    # Distribution analysis
    s2_scores = [r['score'] for r in stage2_data['results']]
    s3_scores = [r['score'] for r in stage3_data['results']]
    
    if len(s2_scores) == 0 or len(s3_scores) == 0:
        print(f"\n❌ ERROR: No samples were evaluated!")
        print(f"   Stage 2: {len(s2_scores)} samples")
        print(f"   Stage 3: {len(s3_scores)} samples")
        print(f"\n   This likely means image files were not found.")
        print(f"   Check the image directory path and image filenames.")
        return None
    
    print(f"\n📈 SCORE DISTRIBUTION:")
    print(f"{'Statistic':<30} {'Stage 2':<15} {'Stage 3':<15}")
    print("-" * 80)
    print(f"{'Mean':<30} {np.mean(s2_scores):<15.1f} {np.mean(s3_scores):<15.1f}")
    print(f"{'Median':<30} {np.median(s2_scores):<15.1f} {np.median(s3_scores):<15.1f}")
    print(f"{'Std Dev':<30} {np.std(s2_scores):<15.1f} {np.std(s3_scores):<15.1f}")
    print(f"{'Min':<30} {min(s2_scores):<15.1f} {min(s3_scores):<15.1f}")
    print(f"{'Max':<30} {max(s2_scores):<15.1f} {max(s3_scores):<15.1f}")
    
    # Quality analysis (score ranges)
    def score_category_count(scores):
        excellent = sum(1 for s in scores if s >= 80)
        good = sum(1 for s in scores if 60 <= s < 80)
        fair = sum(1 for s in scores if 40 <= s < 60)
        poor = sum(1 for s in scores if s < 40)
        return excellent, good, fair, poor
    
    s2_exc, s2_good, s2_fair, s2_poor = score_category_count(s2_scores)
    s3_exc, s3_good, s3_fair, s3_poor = score_category_count(s3_scores)
    
    print(f"\n🏆 QUALITY DISTRIBUTION:")
    print(f"{'Category':<30} {'Stage 2':<15} {'Stage 3':<15}")
    print("-" * 80)
    print(f"{'Excellent (80-100)':<30} {s2_exc:<15} {s3_exc:<15}")
    print(f"{'Good (60-79)':<30} {s2_good:<15} {s3_good:<15}")
    print(f"{'Fair (40-59)':<30} {s2_fair:<15} {s3_fair:<15}")
    print(f"{'Poor (<40)':<30} {s2_poor:<15} {s3_poor:<15}")
    
    # Sample comparisons
    print(f"\n📝 SAMPLE COMPARISONS (First 5):")
    print("=" * 80)
    
    for i in range(min(5, len(stage2_data['results']))):
        s2_result = stage2_data['results'][i]
        s3_result = stage3_data['results'][i]
        
        print(f"\nSample {i+1}: {s2_result['image']}")
        print(f"Question: {s2_result['question'][:80]}...")
        print(f"Reference: {s2_result['reference_answer'][:80]}...")
        print(f"\nStage 2 ({s2_result['score']:.0f}): {s2_result['generated_answer'][:100]}...")
        print(f"Stage 3 ({s3_result['score']:.0f}): {s3_result['generated_answer'][:100]}...")
        print("-" * 80)
    
    # Context: POPE and COCO results
    print(f"\n🔬 CONTEXT - ALL BENCHMARKS:")
    print("=" * 80)
    print("Benchmark               Metric                Stage 2      Stage 3      Change")
    print("-" * 80)
    print(f"{'POPE (OOD)':<23} {'Accuracy':<25} {71.5:<12.1f} {30.0:<12.1f} {-41.5:+.1f} ❌")
    print(f"{'COCO Captions (OOD)':<23} {'CIDEr':<25} {0.76:<12.2f} {0.08:<12.2f} {-0.68:+.2f} ❌")
    print(f"{'LLaVA-Wild (ID)':<23} {'Quality Score':<25} {s2_score:<12.1f} {s3_score:<12.1f} {diff:+.1f} {'✅' if diff > 0 else '❌'}")
    
    print(f"\n💡 INTERPRETATION:")
    if s3_score > s2_score:
        print("   ✅ Stage 3 OUTPERFORMS Stage 2 on in-distribution task (LLaVA-Wild)")
        print("   ✅ This confirms Stage 3 learned the training distribution well")
        print("   ⚠️  But Stage 3 FAILS catastrophically on out-of-distribution tasks")
        print("   📖 Thesis conclusion: Soft routing + instruction tuning = over-specialization")
    else:
        print("   ❌ Stage 3 does NOT outperform Stage 2 even on in-distribution task")
        print("   ⚠️  This suggests Stage 3's issues are more fundamental than expected")
        print("   📖 Need to investigate: Is LLaVA-Wild actually in-distribution?")
    
    print("=" * 80)
    
    return {
        'stage2_mean': np.mean(s2_scores),
        'stage3_mean': np.mean(s3_scores),
        'stage2_scores': s2_scores,
        'stage3_scores': s3_scores
    }

def create_plots(stage2_data, stage3_data, output_dir):
    """Create comparison visualizations."""
    s2_scores = [r['score'] for r in stage2_data['results']]
    s3_scores = [r['score'] for r in stage3_data['results']]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Score distributions (histogram)
    ax = axes[0, 0]
    bins = np.arange(0, 101, 10)
    ax.hist(s2_scores, bins=bins, alpha=0.6, label='Stage 2', color='blue')
    ax.hist(s3_scores, bins=bins, alpha=0.6, label='Stage 3', color='orange')
    ax.axvline(np.mean(s2_scores), color='blue', linestyle='--', linewidth=2, label=f'Stage 2 Mean: {np.mean(s2_scores):.1f}')
    ax.axvline(np.mean(s3_scores), color='orange', linestyle='--', linewidth=2, label=f'Stage 3 Mean: {np.mean(s3_scores):.1f}')
    ax.set_xlabel('Score (0-100)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('LLaVA-Wild Score Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Box plot comparison
    ax = axes[0, 1]
    ax.boxplot([s2_scores, s3_scores], labels=['Stage 2', 'Stage 3'])
    ax.set_ylabel('Score (0-100)', fontsize=12)
    ax.set_title('Score Comparison (Box Plot)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. All benchmarks comparison
    ax = axes[1, 0]
    benchmarks = ['POPE\n(OOD)', 'COCO\n(OOD)', 'LLaVA-Wild\n(In-Dist)']
    # Normalize to 0-100 scale for comparison
    s2_values = [71.5, 76.0, np.mean(s2_scores)]  # POPE accuracy, COCO CIDEr*100, LLaVA score
    s3_values = [30.0, 8.0, np.mean(s3_scores)]   # POPE accuracy, COCO CIDEr*100, LLaVA score
    
    x = np.arange(len(benchmarks))
    width = 0.35
    
    ax.bar(x - width/2, s2_values, width, label='Stage 2', color='blue', alpha=0.7)
    ax.bar(x + width/2, s3_values, width, label='Stage 3', color='orange', alpha=0.7)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance Across Benchmarks', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add annotations
    for i, (s2, s3) in enumerate(zip(s2_values, s3_values)):
        ax.text(i - width/2, s2 + 2, f'{s2:.0f}', ha='center', va='bottom', fontsize=10)
        ax.text(i + width/2, s3 + 2, f'{s3:.0f}', ha='center', va='bottom', fontsize=10)
    
    # 4. Per-sample comparison scatter
    ax = axes[1, 1]
    ax.scatter(s2_scores, s3_scores, alpha=0.5, s=30)
    max_score = max(max(s2_scores), max(s3_scores))
    ax.plot([0, max_score], [0, max_score], 'r--', linewidth=2, label='Equal performance')
    ax.set_xlabel('Stage 2 Score', fontsize=12)
    ax.set_ylabel('Stage 3 Score', fontsize=12)
    ax.set_title('Per-Sample Score Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Count winners
    s2_wins = sum(1 for s2, s3 in zip(s2_scores, s3_scores) if s2 > s3)
    s3_wins = sum(1 for s2, s3 in zip(s2_scores, s3_scores) if s3 > s2)
    ties = len(s2_scores) - s2_wins - s3_wins
    ax.text(0.05, 0.95, f'Stage 2 wins: {s2_wins}\nStage 3 wins: {s3_wins}\nTies: {ties}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir) / 'llava_wild_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Saved visualization: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Compare LLaVA-Wild results')
    parser.add_argument('--stage2', type=str, required=True, help='Stage 2 results JSON')
    parser.add_argument('--stage3', type=str, required=True, help='Stage 3 results JSON')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    
    args = parser.parse_args()
    
    # Load results
    print("\n📂 Loading results...")
    stage2_data = load_results(args.stage2)
    stage3_data = load_results(args.stage3)
    
    # Analyze
    stats = analyze_results(stage2_data, stage3_data)
    
    if stats is None:
        print("\n❌ Analysis failed due to missing data. Exiting.")
        return
    
    # Create plots
    print("\n📊 Creating visualizations...")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    create_plots(stage2_data, stage3_data, args.output_dir)
    
    print("\n✅ Analysis complete!\n")

if __name__ == '__main__':
    main()
