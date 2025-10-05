import torch
import torch.nn as nn
import json
import yaml
import os
import argparse
import heapq
import re
import numpy as np
from tqdm import tqdm

from transformers import (
    LlamaTokenizer,
    AutoModelForCausalLM, 
    AutoProcessor, 
    CLIPVisionModel, 
    AutoConfig
)
from torch.utils.data import DataLoader

# Important: Import your custom model classes
from models.custom_mistral import MistralMoEConfig, MistralMoEForCausalLM
from models import VisionLanguageConnector
from data import COCO_Loader

# Register the custom architecture with Hugging Face Transformers
AutoConfig.register("mistral_moe", MistralMoEConfig)
AutoModelForCausalLM.register(MistralMoEConfig, MistralMoEForCausalLM)

# This global dictionary will be populated by our forward hooks
activation_capture = {}

def get_activation_hook(layer_name):
    """
    Creates a forward hook function that captures the softmax output of the router gate.
    """
    def hook(model, input, output):
        # The output of the gate is the router_logits
        router_logits = output
        probabilities = torch.softmax(router_logits, dim=-1)
        activation_capture[layer_name] = probabilities.detach().cpu()
    return hook

def find_latest_checkpoint(checkpoint_dir, pattern):
    """Finds the checkpoint file with the highest epoch number."""
    if not os.path.exists(checkpoint_dir):
        return None
    
    epoch_files = {}
    for filename in os.listdir(checkpoint_dir):
        match = re.match(pattern, filename)
        if match:
            epoch_number = int(match.group(1))
            epoch_files[epoch_number] = filename
            
    if not epoch_files:
        return None
        
    latest_epoch = max(epoch_files.keys())
    return os.path.join(checkpoint_dir, epoch_files[latest_epoch])

def analyze_specialization(args, paths):
    """
    Main function to run the expert specialization analysis.
    """
    print("--- Starting Expert Specialization Analysis ---")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    # --- 1. Load Models and Tokenizer ---
    print("Loading models and tokenizer...")

    # --- FIX: Use the correct path for the tokenizer ---
    # The tokenizer files are in the original Mistral directory, not the custom MoE directory.
    print(f"Loading tokenizer from: {paths['tokenizer_path']}")
    tokenizer = LlamaTokenizer.from_pretrained(paths["tokenizer_path"])
    tokenizer.pad_token = tokenizer.eos_token
    clip_processor = AutoProcessor.from_pretrained(paths["clip_path"])

    # Now load the custom model from its specific path
    print(f"Loading base MoE model from: {paths['base_model_path']}")
    llm = AutoModelForCausalLM.from_pretrained(
        paths["base_model_path"],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(DEVICE)

    print(f"Loading Stage 2 expert weights from: {paths['stage2_checkpoint_path']}")
    expert_weights = torch.load(paths['stage2_checkpoint_path'], map_location="cpu")
    llm.load_state_dict(expert_weights, strict=False)

    print(f"Loading Stage 2.5 router weights from: {paths['stage2_5_checkpoint_path']}")
    router_weights = torch.load(paths['stage2_5_checkpoint_path'], map_location="cpu")
    llm.load_state_dict(router_weights, strict=False)

    llm.eval()
    for layer in llm.model.layers:
        if hasattr(layer.mlp, "routing_mode"):
            layer.mlp.routing_mode = 'soft'

    vision_encoder = CLIPVisionModel.from_pretrained(paths["clip_path"]).to(DEVICE)
    vision_connector = VisionLanguageConnector().to(DEVICE)
    vision_connector.load_state_dict(torch.load(paths["connector_path"], map_location=DEVICE))
    vision_encoder.eval()
    vision_connector.eval()



    # --- 2. Attach Forward Hooks ---
    print("Attaching forward hooks to MoE router gates...")
    hooks = []
    num_experts = 0
    layer_names = []
    for i, layer in enumerate(llm.model.layers):
        layer_name = f"layer_{i}"
        layer_names.append(layer_name)
        gate = layer.mlp.gate
        hook = gate.register_forward_hook(get_activation_hook(layer_name))
        hooks.append(hook)
        if hasattr(gate, 'out_features'):
            num_experts = gate.out_features
    print(f"Found {num_experts} experts. Hooks attached to {len(hooks)} layers.")

    # --- 3. Prepare Data Structures for Analysis ---
    top_k_tokens_per_layer = {name: {expert_idx: [] for expert_idx in range(num_experts)} for name in layer_names}
    total_routing_stats_per_layer = {name: {
        "vision_tokens_to_expert": [0] * num_experts,
        "text_tokens_to_expert": [0] * num_experts
    } for name in layer_names}

    # --- 4. Load Dataset ---
    print(f"Loading dataset... (will analyze {args.num_samples} samples)")
    dataset = COCO_Loader(
        image_dir=paths["image_dir"],
        annotations_file=paths["annotations_file"],
        clip_processor=clip_processor,
        tokenizer=tokenizer,
        subset_fraction=1.0,
        split="val"
    )
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    total_vision_tokens_processed = 0
    total_text_tokens_processed = 0
    global_expert_counts = [0] * num_experts
    total_entropy_accumulator = 0.0
    vision_entropy_accumulator = 0.0
    text_entropy_accumulator = 0.0
    token_counter = 0
    # --- 5. The Analysis Loop ---
    for i, (image, input_ids, attention_mask) in tqdm(enumerate(data_loader), total=args.num_samples, desc="Analyzing samples"):
        if i >= args.num_samples:
            break
        
        try:
            image_id = dataset.annotations[i]['image_id']
        except (AttributeError, IndexError, KeyError):
            image_id = f"unknown_index_{i}"

        image, input_ids = image.to(DEVICE), input_ids.to(DEVICE)
        
        with torch.no_grad():
            patch_embeddings = vision_encoder(image).last_hidden_state
            visual_soft_tokens = vision_connector(patch_embeddings)
            text_embeddings = llm.model.embed_tokens(input_ids)

        num_visual_tokens = visual_soft_tokens.shape[1]
        num_text_tokens = text_embeddings.shape[1]

        # Increment the corrected global counters once per sample
        total_vision_tokens_processed += num_visual_tokens
        total_text_tokens_processed += num_text_tokens

        combined_embeddings = torch.cat([visual_soft_tokens, text_embeddings], dim=1)
        llm(inputs_embeds=combined_embeddings.to(torch.bfloat16))

        for layer_name, probabilities in activation_capture.items():
            top_scores, top_experts = torch.topk(probabilities, 1, dim=-1)
            
            # Calculate entropy for all tokens (add 1e-9 for numerical stability)
            entropies = -torch.sum(probabilities * torch.log2(probabilities + 1e-9), dim=-1)
            
            #Store the average entropy for this specific layer and batch
            if 'entropies' not in total_routing_stats_per_layer[layer_name]:
                total_routing_stats_per_layer[layer_name]['entropies'] = []
            total_routing_stats_per_layer[layer_name]['entropies'].append(entropies.mean().item())

            # Accumulate entropy, separating by modality
            vision_entropy_accumulator += torch.sum(entropies[:num_visual_tokens]).item()
            text_entropy_accumulator += torch.sum(entropies[num_visual_tokens:]).item()
            
            # Update global expert utilization counts (note: this counts per-layer routing)
            # We will average this out later.
            unique_experts, counts = torch.unique(top_experts, return_counts=True)
            for expert_idx, count in zip(unique_experts, counts):
                global_expert_counts[expert_idx.item()] += count.item()

            for token_idx in range(probabilities.shape[0]):
                score = top_scores[token_idx].item()
                expert_idx = top_experts[token_idx].item()

                if token_idx < num_visual_tokens:
                    total_routing_stats_per_layer[layer_name]["vision_tokens_to_expert"][expert_idx] += 1
                    token_type, context = "<VISION>", "A vision token from the image."
                else:
                    total_routing_stats_per_layer[layer_name]["text_tokens_to_expert"][expert_idx] += 1
                    
                    text_token_idx = token_idx - num_visual_tokens
                    token_type = tokenizer.decode(input_ids[0, text_token_idx])
                    start = max(0, text_token_idx - 5)
                    end = min(num_text_tokens, text_token_idx + 6)
                    context = tokenizer.decode(input_ids[0, start:end])
               

                # Add the counter as a tie-breaker
                token_example = (score, token_counter, {"token_type": token_type, "context": context.replace("\n", " "), "source_image_id": image_id})
                token_counter += 1 # Increment for the next token

                if len(top_k_tokens_per_layer[layer_name][expert_idx]) < args.top_k:
                    heapq.heappush(top_k_tokens_per_layer[layer_name][expert_idx], token_example)
                else:
                    heapq.heappushpop(top_k_tokens_per_layer[layer_name][expert_idx], token_example)

        activation_capture.clear()

    # --- 6. Remove Hooks ---
    for hook in hooks:
        hook.remove()

    num_layers = len(layer_names)

    # Each token is processed by one layer's router at a time, but we sum entropy over all layers
    total_entropy_events = (total_vision_tokens_processed + total_text_tokens_processed) * num_layers

    print("\n--- DEBUGGING METRICS ---")
    print(f"Total Vision Tokens Processed: {total_vision_tokens_processed}")
    print(f"Total Text Tokens Processed:   {total_text_tokens_processed}")
    print(f"Number of Layers:              {num_layers}")

    divisor_check = (total_vision_tokens_processed + total_text_tokens_processed) * num_layers
    print(f"Expected Divisor:              {divisor_check}")
    print(f"Actual Divisor Used:           {total_entropy_events}")

    print(f"\nVision Entropy Sum:            {vision_entropy_accumulator}")
    print(f"Text Entropy Sum:              {text_entropy_accumulator}")
    manual_avg = (vision_entropy_accumulator + text_entropy_accumulator) / total_entropy_events
    print(f"Manually Calculated Avg:       {manual_avg:.4f}")
    print("---------------------------\n")

    
    # Calculate average entropies
    avg_vision_entropy = vision_entropy_accumulator / (total_vision_tokens_processed * num_layers) if total_vision_tokens_processed > 0 else 0
    avg_text_entropy = text_entropy_accumulator / (total_text_tokens_processed * num_layers) if total_text_tokens_processed > 0 else 0
    avg_total_entropy = (vision_entropy_accumulator + text_entropy_accumulator) / total_entropy_events if total_entropy_events > 0 else 0

    # Calculate global load balancing percentages
    total_tokens_routed = sum(global_expert_counts)
    load_balancing_pct = [(count / total_tokens_routed) * 100 for count in global_expert_counts] if total_tokens_routed > 0 else [0] * num_experts
    
    # --- 7. Prepare Full Results ---
    print("\n\n--- Analysis Complete: Generating Report ---")

    # Calculate global token counts from the first layer's stats
    first_layer_stats = total_routing_stats_per_layer[layer_names[0]]
    
    full_results = {
        "global_metrics": {
            "token_counts": {
                "total_vision_tokens": total_vision_tokens_processed,
                "total_text_tokens": total_text_tokens_processed,
                "vision_to_text_ratio": f"1:{total_text_tokens_processed / total_vision_tokens_processed:.2f}" if total_vision_tokens_processed > 0 else "N/A"
            },
            "routing_entropy": {
                "overall_average": avg_total_entropy,
                "average_for_vision_tokens": avg_vision_entropy,
                "average_for_text_tokens": avg_text_entropy
            },
            "load_balancing": {
                f"expert_{i}_share": f"{share:.2f}%" for i, share in enumerate(load_balancing_pct)
            }
        },
        "per_layer_details": {} # This will be populated next
    }

    for layer_name in layer_names:
        for expert_idx in range(num_experts):
            # Sort the top k tokens
            top_k_list = top_k_tokens_per_layer[layer_name][expert_idx]
            # Ensure sorting is correct for the tuple (score, counter, dict)
            top_k_list.sort(key=lambda x: x[0], reverse=True)
        
        modality_split_report = {}
        stats = total_routing_stats_per_layer[layer_name]
        for expert_idx in range(num_experts):
            vision_pct = (stats["vision_tokens_to_expert"][expert_idx] / total_vision_tokens_processed * 100) if total_vision_tokens_processed > 0 else 0
            text_pct = (stats["text_tokens_to_expert"][expert_idx] / total_text_tokens_processed * 100) if total_text_tokens_processed > 0 else 0
            modality_split_report[f"expert_{expert_idx}"] = {
                "vision_token_share": f"{vision_pct:.2f}%",
                "text_token_share": f"{text_pct:.2f}%"
            }
        
        avg_layer_entropy = np.mean(total_routing_stats_per_layer[layer_name].get('entropies', [0]))

        # Add the layer's results into the "per_layer_details" key
        full_results["per_layer_details"][layer_name] = {
            "modality_specialization": modality_split_report,
            "top_activating_tokens": top_k_tokens_per_layer[layer_name],
            "average_entropy": avg_layer_entropy
        }

    # --- 8. Print Report to Console Based on Analysis Target ---
    target = args.layer_to_analyze
    if target.isdigit():
        target_layer = f"layer_{target}"
        print(f"\n--- Detailed Report for {target_layer} ---")
        report_data = full_results.get(target_layer, {})
        if report_data:
            print_report_for_layer(report_data, args.top_k)
        else:
            print(f"ERROR: Layer {target} not found in results.")
    
    elif target == "all":
        for layer_name, report_data in full_results.items():
            print(f"\n--- Detailed Report for {layer_name} ---")
            print_report_for_layer(report_data, args.top_k)
            print("\n" + "="*80 + "\n")
            
    elif target == "average":
        print("\n--- Average Modality Specialization Across All Layers ---")
        avg_stats = {"vision": [0] * num_experts, "text": [0] * num_experts}
        
        # --- FIX: Use the correct global token counters ---
        total_vision = total_vision_tokens_processed
        total_text = total_text_tokens_processed
        # --- END FIX ---
        
        # This part correctly sums the per-expert counts across layers
        for layer_name in layer_names:
            layer_stats = total_routing_stats_per_layer[layer_name]
            for i in range(num_experts):
                avg_stats["vision"][i] += layer_stats["vision_tokens_to_expert"][i]
                avg_stats["text"][i] += layer_stats["text_tokens_to_expert"][i]

        total_vision_routing_events = total_vision_tokens_processed * num_layers
        total_text_routing_events = total_text_tokens_processed * num_layers
        
        for i in range(num_experts):
            # The denominator 'total_vision' is now correct
            vision_pct = (avg_stats["vision"][i] / total_vision_routing_events * 100) if total_vision > 0 else 0
            text_pct = (avg_stats["text"][i] / total_text_routing_events * 100) if total_text_routing_events > 0 else 0
            print(f"  Expert {i}:")
            print(f"    - Handled {vision_pct:.2f}% of all VISION tokens.")
            print(f"    - Handled {text_pct:.2f}% of all TEXT tokens.")
        print("\nNote: Top tokens are layer-specific and not shown in average mode.")

    output_path = os.path.join(paths["output_dir"], "expert_specialization_report.json")
    with open(output_path, "w") as f:
        json.dump(full_results, f, indent=4, default=str)
    print(f"\n✅ Full per-layer report saved to {output_path}")

def print_report_for_layer(report_data, top_k):
    """Helper function to print a formatted report for a single layer's data."""
    print("\n--- Modality Specialization Report ---")
    for expert_str, report in report_data["modality_specialization"].items():
        print(f"  {expert_str}:")
        print(f"    - Handled {report['vision_token_share']} of all VISION tokens.")
        print(f"    - Handled {report['text_token_share']} of all TEXT tokens.")

    print("\n--- Top Activating Tokens ---")
    for expert_idx, examples in report_data["top_activating_tokens"].items():
        print(f"\n--- Top {top_k} Tokens for Expert {expert_idx} ---")
        for score, data in examples:
            print(f'  Score: {score:.4f} | Image ID: {str(data["source_image_id"]):<12} | Type: {data["token_type"]:<10} | Context: "{data["context"]}"')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze MoE expert specialization.")
    
    # --- NON-PATH ARGUMENTS ---
    parser.add_argument("--num_samples", type=int, default=100, help="Number of validation samples to analyze.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top activating examples to save.")
    parser.add_argument("--layer_to_analyze", type=str, default="average", 
                        help="Which layer(s) to analyze. Can be an integer (e.g., '16'), 'all', or 'average'.")

    args = parser.parse_args()
    
    # --- PATHS ---
    config_path = "./configs/training_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    output_dir = config["paths"]["output_dir"]

    # --- FIX: Define separate paths for the tokenizer and the custom model ---
    paths = {
        "base_model_path": "/data/gpfs/projects/COMP90055/aticinovic/models/Mistral-7B-MoE",
        "tokenizer_path": config["paths"]["mistral_local_path"], # Use the path from config for the tokenizer
        "stage2_checkpoint_path": os.path.join(output_dir, "stage2_checkpoints", "llm_stage2_best.pth"),
        "stage2_5_checkpoint_path": os.path.join(output_dir, "stage2_5_checkpoints/archive", "llm_stage2_5_best.pth"),
        "clip_path": config["paths"]["clip_local_path"],
        "connector_path": os.path.join(output_dir, "archive/vision_connector_stage1_best.pth"),
        "image_dir": config["paths"]["image_dir"],
        "annotations_file": config["paths"]["annotations_file"],
        "output_dir": output_dir
    }

    analyze_specialization(args, paths)
