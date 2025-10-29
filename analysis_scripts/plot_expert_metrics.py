#!/usr/bin/env python3
"""
Expert Metrics Visualization Script

Analyzes and visualizes MoE expert utilization patterns from Stage 3 training.
Generates per-layer and aggregate metrics plots showing:
1. Expert load distribution across layers
2. Routing entropy across layers  
3. High confidence fraction across layers
4. Visual vs Text routing patterns across layers
5. Expert specialization evolution across epochs

Usage:
    python analysis_scripts/plot_expert_metrics.py --metrics_dir /path/to/expert_metrics --output_dir results/expert_metrics
"""

import argparse
import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Set publication-quality matplotlib defaults
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
})

def load_expert_metrics(metrics_path):
    """Load expert metrics JSON file."""
    with open(metrics_path, 'r') as f:
        return json.load(f)

def extract_epoch_number(filename):
    """Extract epoch number from filename like 'expert_metrics_epoch_3.json'."""
    import re
    match = re.search(r'epoch_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def plot_expert_load_distribution(all_metrics, output_dir, selected_layers=None):
    """
    Plot expert load distribution for specific layers across all epochs.
    Shows how work is distributed between expert_0 and expert_1 at selected layers.
    
    Args:
        selected_layers: List of layer indices to plot. If None, plots all layers.
    """
    # Updated signature: add selected_epochs
def plot_expert_load_distribution(all_metrics, output_dir, selected_layers=None, selected_epochs=None):
    if selected_layers is None:
        selected_layers = [0, 7, 15, 23, 31]
    if selected_epochs is None:
        epochs = sorted(all_metrics.keys())
    else:
        epochs = selected_epochs
    fig, ax = plt.subplots(figsize=(12, 6))
    x_positions = np.arange(len(selected_layers))
    width = 0.35 / len(epochs)
    colors = plt.cm.viridis(np.linspace(0, 1, len(epochs)))
    for epoch_idx, (epoch, color) in enumerate(zip(epochs, colors)):
        metrics = all_metrics[epoch]
        expert_0_loads = []
        expert_1_loads = []
        
        for layer_idx in selected_layers:
            layer_data = metrics['per_layer'][layer_idx]
            load_dist = layer_data['expert_load_distribution']
            expert_0_loads.append(load_dist.get('expert_0', 0))
            expert_1_loads.append(load_dist.get('expert_1', 0))
        
        # Offset bars for each epoch
        offset = width * (epoch_idx - len(epochs)/2 + 0.5)
        ax.bar(x_positions + offset, expert_0_loads, width, 
               label=f'Epoch {epoch} - Expert 0', color=color, alpha=0.7)
        ax.bar(x_positions + offset, expert_1_loads, width, 
               label=f'Epoch {epoch} - Expert 1', color=color, alpha=0.4, hatch='//')
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Expert Load (%)')
    ax.set_title(f'Expert Load Distribution at Selected Layers')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'L{l}' for l in selected_layers], rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'expert_load_distribution.png'))
    plt.close()
    print(f"  ✅ Saved: expert_load_distribution.png")

def plot_routing_entropy(all_metrics, output_dir, selected_layers=None):
    """
    Plot routing entropy for specific layers across all epochs.
    Lower entropy = more decisive/confident routing.
    
    Args:
        selected_layers: List of layer indices to plot. If None, uses default selection.
    """
def plot_routing_entropy(all_metrics, output_dir, selected_layers=None, selected_epochs=None):
    if selected_layers is None:
        selected_layers = [0, 7, 15, 23, 31]
    if selected_epochs is None:
        epochs = sorted(all_metrics.keys())
    else:
        epochs = selected_epochs
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(epochs)))
    for epoch, color in zip(epochs, colors):
        entropies_across_layers = []
        
        for layer_idx in selected_layers:
            metrics = all_metrics[epoch]
            layer_data = metrics['per_layer'][layer_idx]
            entropies_across_layers.append(layer_data['avg_routing_entropy'])
        
        ax.plot(selected_layers, entropies_across_layers, label=f'Epoch {epoch}', 
                color=color, marker='o', markersize=8, linewidth=2.5)
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Average Routing Entropy')
    ax.set_title('Routing Entropy Across Layers\n(Lower = More Decisive Routing)')
    ax.legend(loc='best')
    ax.set_xticks(selected_layers)
    ax.set_xticklabels([f'L{l}' for l in selected_layers], rotation=45, ha='right')
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'routing_entropy.png'))
    plt.close()
    print(f"  ✅ Saved: routing_entropy.png")

def plot_high_confidence_fraction(all_metrics, output_dir, selected_layers=None):
    """
    Plot high confidence routing fraction for specific layers across all epochs.
    Shows what fraction of routing decisions are made with >70% confidence.
    
    Args:
        selected_layers: List of layer indices to plot. If None, uses default selection.
    """
def plot_high_confidence_fraction(all_metrics, output_dir, selected_layers=None, selected_epochs=None):
    if selected_layers is None:
        selected_layers = [0, 7, 15, 23, 31]
    if selected_epochs is None:
        epochs = sorted(all_metrics.keys())
    else:
        epochs = selected_epochs
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(epochs)))
    for epoch, color in zip(epochs, colors):
        high_conf_across_layers = []
        
        for layer_idx in selected_layers:
            metrics = all_metrics[epoch]
            layer_data = metrics['per_layer'][layer_idx]
            high_conf_across_layers.append(layer_data['high_confidence_fraction'])
        
        ax.plot(selected_layers, high_conf_across_layers, label=f'Epoch {epoch}', 
                color=color, marker='o', markersize=8, linewidth=2.5)
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('High Confidence Fraction')
    ax.set_title('High Confidence Routing Fraction Across Layers\n(Fraction of Decisions with >70% Confidence)')
    ax.legend(loc='best')
    ax.set_xticks(selected_layers)
    ax.set_xticklabels([f'L{l}' for l in selected_layers], rotation=45, ha='right')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'high_confidence_fraction.png'))
    plt.close()
    print(f"  ✅ Saved: high_confidence_fraction.png")

def plot_visual_vs_text_routing(all_metrics, output_dir, selected_layers=None):
    """
    Plot visual vs text token routing patterns for specific layers.
    Shows what % of visual tokens go to expert_1 vs % of text tokens go to expert_1.
    This reveals modality-specific specialization patterns.
    
    Args:
        selected_layers: List of layer indices to plot. If None, uses default selection.
    """
def plot_visual_vs_text_routing(all_metrics, output_dir, selected_layers=None, selected_epochs=None):
    if selected_layers is None:
        selected_layers = [0, 7, 15, 23, 31]
    if selected_epochs is None:
        epochs = sorted(all_metrics.keys())
    else:
        epochs = selected_epochs
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(epochs)))
    for epoch, color in zip(epochs, colors):
        visual_expert1_across_layers = []
        text_expert1_across_layers = []
        
        for layer_idx in selected_layers:
            metrics = all_metrics[epoch]
            layer_data = metrics['per_layer'][layer_idx]
            routing = layer_data['visual_vs_text_routing']
            
            # Get % of visual tokens going to expert_1
            if 'visual' in routing and 'expert_1' in routing['visual']:
                visual_expert1_across_layers.append(routing['visual']['expert_1'])
            else:
                visual_expert1_across_layers.append(0)
            
            # Get % of text tokens going to expert_1
            if 'text' in routing and 'expert_1' in routing['text']:
                text_expert1_across_layers.append(routing['text']['expert_1'])
            else:
                text_expert1_across_layers.append(0)
        
        # Plot with different markers for visual vs text
        ax.plot(selected_layers, visual_expert1_across_layers, label=f'Epoch {epoch} - Visual', 
                color=color, marker='o', markersize=8, linewidth=2.5, linestyle='-')
        ax.plot(selected_layers, text_expert1_across_layers, label=f'Epoch {epoch} - Text', 
                color=color, marker='s', markersize=8, linewidth=2.5, linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('% Tokens Routed to Expert 1')
    ax.set_title('Visual vs Text Token Routing Across Layers\n(% Routed to Expert 1)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    ax.set_xticks(selected_layers)
    ax.set_xticklabels([f'L{l}' for l in selected_layers], rotation=45, ha='right')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visual_vs_text_routing.png'))
    plt.close()
    print(f"  ✅ Saved: visual_vs_text_routing.png")

def plot_specialization_evolution(all_metrics, output_dir, selected_epochs=None):
    """
    Plot how expert specialization evolves across epochs.
    Shows aggregate % of visual/text tokens routed to each expert over training.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    if selected_epochs is None:
        epochs = sorted(all_metrics.keys())
    else:
        epochs = selected_epochs
    
    # Extract aggregate routing patterns
    visual_to_expert0 = []
    visual_to_expert1 = []
    text_to_expert0 = []
    text_to_expert1 = []
    
    for epoch in epochs:
        metrics = all_metrics[epoch]
        agg = metrics['aggregate']
        
        if 'visual_routing' in agg:
            visual_to_expert0.append(agg['visual_routing'].get('expert_0', 0))
            visual_to_expert1.append(agg['visual_routing'].get('expert_1', 0))
        else:
            visual_to_expert0.append(0)
            visual_to_expert1.append(0)
        
        if 'text_routing' in agg:
            text_to_expert0.append(agg['text_routing'].get('expert_0', 0))
            text_to_expert1.append(agg['text_routing'].get('expert_1', 0))
        else:
            text_to_expert0.append(0)
            text_to_expert1.append(0)
    
    # Plot 1: Visual Token Routing Evolution
    ax1.plot(epochs, visual_to_expert0, label='Expert 0', marker='o', linewidth=2.5, color='#1f77b4', markersize=10)
    ax1.plot(epochs, visual_to_expert1, label='Expert 1', marker='s', linewidth=2.5, color='#ff7f0e', markersize=10)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('% Visual Tokens Routed to Expert')
    ax1.set_title('Visual Token Routing Evolution\n(Aggregate Across All Layers)')
    ax1.legend()
    ax1.set_ylim(0, 100)
    ax1.set_xticks(epochs)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Text Token Routing Evolution
    ax2.plot(epochs, text_to_expert0, label='Expert 0', marker='o', linewidth=2.5, color='#1f77b4', markersize=10)
    ax2.plot(epochs, text_to_expert1, label='Expert 1', marker='s', linewidth=2.5, color='#ff7f0e', markersize=10)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('% Text Tokens Routed to Expert')
    ax2.set_title('Text Token Routing Evolution\n(Aggregate Across All Layers)')
    ax2.legend()
    ax2.set_ylim(0, 100)
    ax2.set_xticks(epochs)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'specialization_evolution.png'))
    plt.close()
    print(f"  ✅ Saved: specialization_evolution.png")

def plot_aggregate_summary(all_metrics, output_dir):
    """
    Plot aggregate summary statistics for the latest epoch.
    Shows overall expert utilization patterns (simplified version).
    """
    # Use the latest epoch
    latest_epoch = max(all_metrics.keys())
    metrics = all_metrics[latest_epoch]
    agg = metrics['aggregate']
    
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
    
    colors_bar = ['#1f77b4', '#ff7f0e']
    
    # Plot 1: Expert Load Distribution (Aggregate)
    ax1 = fig.add_subplot(gs[0, 0])
    experts = list(agg['expert_load_distribution'].keys())
    loads = list(agg['expert_load_distribution'].values())
    ax1.bar(experts, loads, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Load (%)', fontsize=12)
    ax1.set_title('Aggregate Expert Load Distribution', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 100)
    for i, (expert, load) in enumerate(zip(experts, loads)):
        ax1.text(i, load + 3, f'{load:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Visual Routing (Aggregate)
    ax2 = fig.add_subplot(gs[0, 1])
    if 'visual_routing' in agg:
        visual_experts = list(agg['visual_routing'].keys())
        visual_loads = list(agg['visual_routing'].values())
        ax2.bar(visual_experts, visual_loads, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('% Visual Tokens', fontsize=12)
        ax2.set_title('Visual Token Routing', fontsize=13, fontweight='bold')
        ax2.set_ylim(0, 100)
        for i, (expert, load) in enumerate(zip(visual_experts, visual_loads)):
            ax2.text(i, load + 3, f'{load:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Text Routing (Aggregate)
    ax3 = fig.add_subplot(gs[0, 2])
    if 'text_routing' in agg:
        text_experts = list(agg['text_routing'].keys())
        text_loads = list(agg['text_routing'].values())
        ax3.bar(text_experts, text_loads, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax3.set_ylabel('% Text Tokens', fontsize=12)
        ax3.set_title('Text Token Routing', fontsize=13, fontweight='bold')
        ax3.set_ylim(0, 100)
        for i, (expert, load) in enumerate(zip(text_experts, text_loads)):
            ax3.text(i, load + 3, f'{load:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
        ax3.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f'Aggregate Expert Metrics Summary (Epoch {latest_epoch})', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
    plt.savefig(os.path.join(output_dir, 'aggregate_summary.png'))
    plt.close()
    print(f"  ✅ Saved: aggregate_summary.png")

def generate_report(all_metrics, output_dir):
    """Generate a text report summarizing key findings."""
    report_path = os.path.join(output_dir, 'expert_metrics_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EXPERT UTILIZATION METRICS REPORT\n")
        f.write("="*80 + "\n\n")
        
        for epoch in sorted(all_metrics.keys()):
            metrics = all_metrics[epoch]
            agg = metrics['aggregate']
            
            f.write(f"\nEPOCH {epoch}\n")
            f.write("-" * 40 + "\n")
            
            f.write(f"\n1. Expert Load Distribution (Aggregate):\n")
            for expert, load in agg['expert_load_distribution'].items():
                f.write(f"   {expert}: {load:.2f}%\n")
            
            f.write(f"\n2. Routing Entropy: {agg['avg_routing_entropy']:.4f}\n")
            f.write(f"   (Lower = more decisive routing)\n")
            
            f.write(f"\n3. High Confidence Fraction: {agg['high_confidence_fraction']:.2%}\n")
            f.write(f"   (Fraction with >70% confidence)\n")
            
            f.write(f"\n4. Visual Token Routing:\n")
            if 'visual_routing' in agg:
                for expert, load in agg['visual_routing'].items():
                    f.write(f"   {expert}: {load:.2f}%\n")
            
            f.write(f"\n5. Text Token Routing:\n")
            if 'text_routing' in agg:
                for expert, load in agg['text_routing'].items():
                    f.write(f"   {expert}: {load:.2f}%\n")
            
            # Compute specialization score
            if 'visual_routing' in agg and 'text_routing' in agg:
                visual_e0 = agg['visual_routing'].get('expert_0', 50)
                text_e0 = agg['text_routing'].get('expert_0', 50)
                specialization_divergence = abs(visual_e0 - text_e0)
                f.write(f"\n6. Modality Specialization Divergence: {specialization_divergence:.2f}%\n")
                f.write(f"   (Difference in expert_0 preference between modalities)\n")
                if specialization_divergence > 30:
                    f.write(f"   ✓ Strong modality specialization detected!\n")
                elif specialization_divergence > 15:
                    f.write(f"   ✓ Moderate modality specialization\n")
                else:
                    f.write(f"   ⚠ Weak modality specialization\n")
            
            f.write("\n" + "="*80 + "\n")
    
    print(f"  ✅ Saved: expert_metrics_report.txt")

def main():
    parser = argparse.ArgumentParser(description='Visualize expert utilization metrics from Stage 3 training')
    parser.add_argument('--metrics_dir', type=str, required=True,
                        help='Directory containing expert metrics JSON files')
    parser.add_argument('--output_dir', type=str, default='results/expert_metrics',
                        help='Directory to save output plots')
    parser.add_argument('--layers', type=str, default='0 7 15 23 31',
                        help="Layer indices to plot (e.g. '0 7 15 23 31' or 'all_layers')")
    parser.add_argument('--epochs', type=str, default=None,
                        help="Epochs to plot (e.g. '1,2,5' or '1-5,7'). Default: all epochs.")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("EXPERT METRICS VISUALIZATION")
    print("="*80)
    print(f"📂 Metrics directory: {args.metrics_dir}")
    print(f"📊 Output directory:  {args.output_dir}\n")
    
    # Find all expert metrics files
    metrics_files = glob.glob(os.path.join(args.metrics_dir, 'expert_metrics_epoch_*.json'))
    if not metrics_files:
        print(f"❌ No expert metrics files found in {args.metrics_dir}")
        print(f"   Expected files matching pattern: expert_metrics_epoch_*.json")
        return
    print(f"📋 Found {len(metrics_files)} epoch(s) of metrics:")
    # Load all metrics
    all_metrics = {}
    for metrics_file in sorted(metrics_files):
        epoch = extract_epoch_number(os.path.basename(metrics_file))
        if epoch is not None:
            metrics = load_expert_metrics(metrics_file)
            all_metrics[epoch] = metrics
            print(f"   ✓ Epoch {epoch}: {os.path.basename(metrics_file)}")
    if not all_metrics:
        print("❌ Failed to load any metrics files")
        return

    # Parse epochs argument
    available_epochs = sorted(all_metrics.keys())
    if args.epochs:
        selected_epochs = set()
        for part in args.epochs.split(','):
            part = part.strip()
            if '-' in part:
                start, end = part.split('-')
                selected_epochs.update(range(int(start), int(end)+1))
            else:
                selected_epochs.add(int(part))
        selected_epochs = sorted(e for e in selected_epochs if e in available_epochs)
        if not selected_epochs:
            print(f"❌ No matching epochs found for --epochs {args.epochs}")
            return
    else:
        selected_epochs = available_epochs

    # Filter all_metrics to selected epochs
    all_metrics = {e: all_metrics[e] for e in selected_epochs}

    # Parse layers argument
    # If 'all_layers', use all available layers from the first epoch
    if args.layers.strip() == 'all_layers':
        first_epoch = next(iter(all_metrics.values()))
        num_layers = len(first_epoch['per_layer'])
        selected_layers = list(range(num_layers))
    else:
        selected_layers = [int(x) for x in args.layers.strip().split()]

    print(f"\n{'='*80}")
    print("GENERATING PLOTS")
    print("="*80)
    print(f"📍 Selected layers: {selected_layers}")
    print(f"📍 Selected epochs: {selected_epochs}\n")

    # Generate all plots
    print("📈 Generating per-layer plots...")
    plot_expert_load_distribution(all_metrics, args.output_dir, selected_layers)
    plot_routing_entropy(all_metrics, args.output_dir, selected_layers)
    plot_high_confidence_fraction(all_metrics, args.output_dir, selected_layers)
    plot_visual_vs_text_routing(all_metrics, args.output_dir, selected_layers)

    print("\n📈 Generating specialization evolution plot...")
    plot_specialization_evolution(all_metrics, args.output_dir)

    print("\n📈 Generating aggregate summary...")
    plot_aggregate_summary(all_metrics, args.output_dir)

    print("\n📝 Generating text report...")
    generate_report(all_metrics, args.output_dir)

    print(f"\n{'='*80}")
    print("✅ COMPLETE!")
    print("="*80)
    print(f"\n📁 All plots saved to: {args.output_dir}/")
    print("\nGenerated files:")
    print("  • expert_load_distribution.png")
    print("  • routing_entropy.png")
    print("  • high_confidence_fraction.png")
    print("  • visual_vs_text_routing.png")
    print("  • specialization_evolution.png")
    print("  • aggregate_summary.png")
    print("  • expert_metrics_report.txt")
    print()

if __name__ == "__main__":
    main()
