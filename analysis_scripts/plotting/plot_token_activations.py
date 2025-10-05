import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def parse_data(data):
    """
    Parses the raw JSON data into structured lists for plotting.
    """
    per_layer_data = data['per_layer_details']
    # Dynamically determine the number of experts from the first layer
    first_layer_key = list(per_layer_data.keys())[0]
    num_experts = len(per_layer_data[first_layer_key]['modality_specialization'])

    # Initialize data structures
    expert_data = {
        'vision': [[] for _ in range(num_experts)],
        'text': [[] for _ in range(num_experts)]
    }
    layer_ids = []
    avg_entropies = []
    avg_top1_scores = []

    # Sort layers numerically
    sorted_layer_keys = sorted(per_layer_data.keys(), key=lambda x: int(x.split('_')[1]))

    for layer_key in sorted_layer_keys:
        layer_ids.append(int(layer_key.split('_')[1]))
        layer_data = per_layer_data[layer_key]

        # --- 1. Modality Specialization ---
        for i in range(num_experts):
            expert_key = f"expert_{i}"
            specialization = layer_data['modality_specialization'][expert_key]
            
            # Convert percentage string to float
            vision_share = float(specialization['vision_token_share'].strip('%'))
            text_share = float(specialization['text_token_share'].strip('%'))
            
            expert_data['vision'][i].append(vision_share)
            expert_data['text'][i].append(text_share)


        if 'average_entropy' in layer_data:
            avg_entropies.append(layer_data['average_entropy'])
        else:
            avg_entropies.append(np.nan)

        # --- 3. Average Top-1 Score (Proxy for Router Confidence) ---
        scores = []
        if 'top_activating_tokens' in layer_data:
            for expert_idx_str, tokens in layer_data['top_activating_tokens'].items():
                for token_info in tokens:
                    scores.append(token_info[0]) # The score is the first element
            if scores:
                avg_top1_scores.append(np.mean(scores))
            else:
                avg_top1_scores.append(np.nan)
        else:
            avg_top1_scores.append(np.nan)


    return {
        'layer_ids': layer_ids,
        'num_experts': num_experts,
        'specialization': expert_data,
        'avg_entropies': avg_entropies,
        'avg_top1_scores': avg_top1_scores
    }


def plot_specialization_lines(parsed_data, output_dir):
    """
    Plots Vision and Text token share per expert across all layers.
    This is the best plot to visualize the 'jersey swapping' phenomenon.
    """
    print("Generating specialization line plot...")
    layer_ids = parsed_data['layer_ids']
    num_experts = parsed_data['num_experts']
    specialization = parsed_data['specialization']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    
    colors = plt.cm.viridis(np.linspace(0, 1, num_experts))

    # Plot 1: Vision Token Share
    for i in range(num_experts):
        ax1.plot(layer_ids, specialization['vision'][i], marker='o', linestyle='-', color=colors[i], label=f'Expert {i}')
    ax1.set_title('Vision Token Routing Specialization Across Layers', fontsize=16)
    ax1.set_ylabel('Share of Vision Tokens (%)')
    ax1.set_ylim(0, 100)
    ax1.axhline(50, color='gray', linestyle='--', linewidth=1)
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plot 2: Text Token Share
    for i in range(num_experts):
        ax2.plot(layer_ids, specialization['text'][i], marker='o', linestyle='-', color=colors[i], label=f'Expert {i}')
    ax2.set_title('Text Token Routing Specialization Across Layers', fontsize=16)
    ax2.set_ylabel('Share of Text Tokens (%)')
    ax2.set_xlabel('Layer ID')
    ax2.set_ylim(0, 100)
    ax2.axhline(50, color='gray', linestyle='--', linewidth=1)
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'specialization_across_layers.png'))
    plt.close()

def plot_entropy(parsed_data, output_dir):
    """
    Plots the average routing entropy across layers, if available.
    """
    layer_ids = parsed_data['layer_ids']
    avg_entropies = parsed_data['avg_entropies']

    if all(np.isnan(avg_entropies)):
        print("Skipping entropy plot: No entropy data found in JSON.")
        return

    print("Generating entropy plot...")
    plt.figure(figsize=(15, 6))
    plt.plot(layer_ids, avg_entropies, marker='o', linestyle='-')
    plt.title('Average Routing Entropy Across Layers', fontsize=16)
    plt.xlabel('Layer ID')
    plt.ylabel('Average Entropy (bits)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'entropy_across_layers.png'))
    plt.close()
    
def plot_entropy_across_layers(parsed_data, output_dir):
    """
    Plots the average routing entropy across layers as a measure of uncertainty.
    """
    layer_ids = parsed_data['layer_ids']
    avg_entropies = parsed_data['avg_entropies']
    
    if all(np.isnan(avg_entropies)):
        print("Skipping entropy plot: No per-layer entropy data found in JSON.")
        return
        
    print("Generating router uncertainty (entropy) plot...")
    plt.figure(figsize=(15, 6))
    plt.plot(layer_ids, avg_entropies, marker='o', linestyle='-', color='green')
    
    plt.title('Router Uncertainty Across Layers (Average Entropy)', fontsize=16)
    plt.xlabel('Layer ID')
    plt.ylabel('Average Entropy (bits)')
    # For a 2-expert system, max entropy is log2(2) = 1.0
    plt.ylim(0, 1.0) 
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'entropy_across_layers.png'))
    plt.close()

def plot_global_metrics(data, output_dir):
    """
    Generates a PNG table of global metrics and a bar chart for load balancing.
    """
    if 'global_metrics' not in data:
        print("Skipping global metrics: 'global_metrics' key not found in JSON.")
        return

    print("Generating global metrics summary...")
    global_metrics = data['global_metrics']

    # --- 1. Generate and Save the Bar Chart for Load Balancing ---
    if 'load_balancing' in global_metrics:
        lb_stats = global_metrics['load_balancing']
        expert_labels = sorted(lb_stats.keys())
        shares = [float(lb_stats[key].strip('%')) for key in expert_labels]
        
        plt.figure(figsize=(8, 6))
        sns.barplot(x=expert_labels, y=shares, palette='viridis')
        
        plt.title('Global Load Balancing: Overall Token Share per Expert', fontsize=16)
        plt.ylabel('Share of Total Tokens (%)')
        plt.xlabel('Expert ID')
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', linewidth=0.7)
        
        for index, value in enumerate(shares):
            plt.text(index, value + 1, f'{value:.2f}%', ha='center')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'global_load_balancing.png'))
        plt.close()

    # --- 2. Generate and Save a PNG Table of All Global Metrics ---
    table_data = []
    if 'token_counts' in global_metrics:
        table_data.append(['Total Vision Tokens', global_metrics['token_counts']['total_vision_tokens']])
        table_data.append(['Total Text Tokens', global_metrics['token_counts']['total_text_tokens']])
        table_data.append(['Vision to Text Ratio', global_metrics['token_counts']['vision_to_text_ratio']])
    
    if 'routing_entropy' in global_metrics:
        table_data.append(['', '']) # Add a separator for clarity
        table_data.append(['Overall Avg Entropy', f"{global_metrics['routing_entropy']['overall_average']:.4f}"])
        table_data.append(['Avg Entropy (Vision)', f"{global_metrics['routing_entropy']['average_for_vision_tokens']:.4f}"])
        table_data.append(['Avg Entropy (Text)', f"{global_metrics['routing_entropy']['average_for_text_tokens']:.4f}"])

    # Create the plot for the table
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off')
    ax.set_title('Global Metrics Summary', fontsize=16, weight='bold')

    table = ax.table(
        cellText=[[row[1]] for row in table_data],
        rowLabels=[row[0] for row in table_data],
        colLabels=['Value'],
        loc='center',
        cellLoc='center',
        rowLoc='left'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8) # Adjust cell height and width

    # Style the table
    for (row, col), cell in table.get_celld().items():
        if (row == 0): # Header row
            cell.set_text_props(weight='bold')
        if table_data[row-1][0] == '': # Separator row
             cell.set_height(0.1)
             
    plt.savefig(os.path.join(output_dir, 'global_metrics_summary.png'), bbox_inches='tight', dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot expert specialization results from a JSON file.")
    parser.add_argument("json_file", type=str, help="Path to the expert specialization report JSON file.")
    parser.add_argument("--output_dir", type=str, default="plot_results", help="Directory to save the plots.")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and parse data
    print(f"Loading data from {args.json_file}...")
    with open(args.json_file, 'r') as f:
        data = json.load(f)
    
    parsed_data = parse_data(data)

    # Generate plots
    plot_specialization_lines(parsed_data, args.output_dir)
    plot_entropy_across_layers(parsed_data, args.output_dir)
    plot_global_metrics(data, args.output_dir)
    
    print(f"\n✅ Plots saved to '{args.output_dir}' directory.")


if __name__ == "__main__":
    main()