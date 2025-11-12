#!/usr/bin/env python3
"""
Quick test to see if Stage 3 prompt improvements work.
Tests a single POPE question with different prompts.
"""

import torch
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent / 'karpathy_evaluation'))

from karpathy_utils import load_model_checkpoint, load_and_preprocess_image

def test_pope_prompt(checkpoint_path, image_path, question, device='cuda'):
    """Test different prompts for POPE question."""
    
    print(f"Loading Stage 3 model from: {checkpoint_path}")
    model, processor, tokenizer = load_model_checkpoint(checkpoint_path, device=device)
    model.eval()
    
    # Set soft routing for Stage 3
    for layer in model.llm.model.layers:
        if hasattr(layer.mlp, "routing_mode"):
            layer.mlp.routing_mode = 'soft'
    
    print(f"Loading image: {image_path}")
    image = load_and_preprocess_image(image_path, processor)
    pixel_values = image.unsqueeze(0).to(device)
    
    # Process vision
    vision_outputs = model.vision_encoder(pixel_values=pixel_values)
    patch_embeddings = vision_outputs.last_hidden_state
    visual_soft_tokens = model.connector(patch_embeddings).to(torch.bfloat16)
    
    # Test different prompts
    prompts = [
        f"{question} Answer with yes or no.",  # Old prompt
        f"{question}\nAnswer yes or no only.",  # New prompt
        f"{question}\nAnswer:",  # Minimal prompt
        f"{question}\nProvide a brief yes or no answer.",  # Alternative
        question,  # Just the question
    ]
    
    print("\n" + "="*80)
    print(f"QUESTION: {question}")
    print("="*80)
    
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            print(f"\n[Prompt {i+1}] {repr(prompt)}")
            
            # Tokenize
            prompt_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
            bos_ids = torch.tensor([[tokenizer.bos_token_id]]).to(device)
            input_ids = torch.cat([bos_ids, prompt_ids], dim=1)
            
            # Get embeddings
            text_embeddings = model.llm.get_input_embeddings()(input_ids)
            combined_embeddings = torch.cat([visual_soft_tokens, text_embeddings], dim=1)
            attention_mask = torch.ones(combined_embeddings.shape[:2], device=device)
            
            # Generate with greedy decoding
            generated_ids = input_ids[0].tolist()
            max_tokens = 10
            
            for step in range(max_tokens):
                outputs = model.llm(
                    inputs_embeds=combined_embeddings,
                    attention_mask=attention_mask,
                    use_cache=False
                )
                
                logits = outputs.logits[0, -1, :]
                next_token_id = torch.argmax(logits).item()
                
                # Stop if EOS
                if next_token_id == tokenizer.eos_token_id:
                    break
                
                generated_ids.append(next_token_id)
                
                # Update embeddings for next token
                next_token_embed = model.llm.get_input_embeddings()(
                    torch.tensor([[next_token_id]], device=device)
                )
                combined_embeddings = torch.cat([combined_embeddings, next_token_embed], dim=1)
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((1, 1), device=device)
                ], dim=1)
            
            # Decode
            answer_ids = generated_ids[len(input_ids[0]):]
            answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
            
            print(f"→ Answer: {repr(answer)}")
            print(f"   Tokens: {answer_ids}")


if __name__ == "__main__":
    checkpoint = "/data/gpfs/projects/COMP90055/aticinovic/outputs/stage3_checkpoints/llm_stage3_best_portable.pth"
    
    # Use an image from POPE test set
    image_path = "/data/gpfs/projects/COMP90055/aticinovic/datasets/coco/val2017/000000142790.jpg"
    question = "Is there a truck in the image?"
    
    test_pope_prompt(checkpoint, image_path, question)
