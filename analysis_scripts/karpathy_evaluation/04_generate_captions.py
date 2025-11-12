#!/usr/bin/env python3
"""
Step 4: Generate captions for Karpathy test images using beam search.
"""

import torch
import argparse
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import time

from karpathy_utils import (
    load_model_checkpoint,
    load_and_preprocess_image,
    save_json,
    load_json,
    format_time,
    print_banner
)


def generate_captions(
    model,
    processor,
    tokenizer,
    images_data: List[Dict],
    image_base_dir: str,
    num_beams: int = 5,
    max_length: int = 20,
    temperature: float = 1.0,
    batch_size: int = 16,
    device: str = 'cuda',
    stage_name: str = 'stage2'
) -> List[Dict]:
    """
    Generate captions for images using beam search.
    
    Args:
        model: The vision-language model
        processor: Image processor
        tokenizer: Text tokenizer
        images_data: List of image entries with 'image_id', 'filepath', 'cocoid'
        image_base_dir: Base directory for images
        num_beams: Number of beams for beam search
        max_length: Maximum caption length
        temperature: Sampling temperature
        batch_size: Batch size for generation
        device: Device to use
        
    Returns:
        results: List of {image_id: int, caption: str} for COCO eval format
    """
    model.eval()
    model.to(device)
    
    results = []
    num_images = len(images_data)
    
    print(f"\n🔮 Generating captions for {num_images} images...")
    print(f"   Beam search: {num_beams} beams")
    print(f"   Max length: {max_length} tokens")
    print(f"   Batch size: {batch_size}")
    print(f"   Device: {device}")
    
    start_time = time.time()
    
    # Process in batches
    num_batches = (num_images + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Generating"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, num_images)
            batch_data = images_data[batch_start:batch_end]
            
            # Load and preprocess images
            images = []
            valid_indices = []
            
            for idx, img_entry in enumerate(batch_data):
                image_path = Path(image_base_dir) / img_entry['filepath']
                try:
                    image = load_and_preprocess_image(str(image_path), processor)
                    images.append(image)
                    valid_indices.append(idx)
                except Exception as e:
                    print(f"\n⚠️  Error loading {image_path}: {e}")
                    continue
            
            if not images:
                continue
            
            # Stack images: (batch, 3, 224, 224)
            pixel_values = torch.stack(images).to(device)
            
            # Generate captions with beam search
            # Note: This assumes your model has a generate() method
            # You may need to adapt this based on your model's interface
            try:
                # Option 1: If model has built-in generate
                outputs = model.generate(
                    pixel_values=pixel_values,
                    num_beams=num_beams,
                    max_length=max_length,
                    temperature=temperature,
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                # Decode generated tokens
                for idx, output in enumerate(outputs):
                    caption = tokenizer.decode(output, skip_special_tokens=True)
                    caption = caption.strip()
                    
                    # Get corresponding image entry
                    data_idx = valid_indices[idx]
                    img_entry = batch_data[data_idx]
                    
                    results.append({
                        'image_id': img_entry['coco_id'],  # Use COCO ID for eval
                        'caption': caption
                    })
                    
            except AttributeError:
                # Option 2: Manual generation if model doesn't have generate()
                print("\n⚠️  Model doesn't have generate() method. Using manual generation.")
                
                # Detect stage type to use correct routing
                is_stage3 = (stage_name == 'stage3')
                if is_stage3:
                    print("   📍 Detected Stage 3 checkpoint - using SOFT routing (gating network)")
                    # Set all MoE layers to soft routing mode
                    for layer in model.llm.model.layers:
                        if hasattr(layer.mlp, "routing_mode"):
                            layer.mlp.routing_mode = 'soft'
                else:
                    print("   📍 Detected Stage 2 checkpoint - using HARD routing (routing masks)")
                
                # Process vision through encoder and connector (like training script)
                vision_outputs = model.vision_encoder(pixel_values=pixel_values)
                patch_embeddings = vision_outputs.last_hidden_state  # (batch, 257, 1024)
                
                # Project to LLM space using connector
                visual_soft_tokens = model.connector(patch_embeddings)
                visual_soft_tokens = visual_soft_tokens.to(torch.bfloat16)  # Match LLM dtype
                
                # Autoregressive generation for each image
                for batch_idx in range(visual_soft_tokens.shape[0]):
                    # Get visual tokens for this image
                    visual_tokens = visual_soft_tokens[batch_idx:batch_idx+1]  # (1, 257, 4096)
                    
                    # For Stage 3 (LLaVA-trained), add a question prompt to match training distribution
                    # For Stage 2 (COCO-trained), just use BOS
                    if is_stage3:
                        # Prompt options (ranked by COCO-likeness):
                        # 1. "A photo of" - common captioning prefix, encourages direct noun phrases
                        # 2. "Caption:" - shortest, most direct, mimics annotation format  
                        # 3. "Briefly describe what you see." - encourages conciseness
                        # 4. "What is in this image?" - simple LLaVA-style question
                        # Current: Testing "A photo of" for most COCO-like direct captions
                        prompt_text = "A photo of"
                        prompt_ids = tokenizer(prompt_text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
                        
                        # Combine: BOS + prompt tokens
                        bos_ids = torch.tensor([[tokenizer.bos_token_id]]).to(device)
                        input_ids = torch.cat([bos_ids, prompt_ids], dim=1)
                    else:
                        # Stage 2: Just BOS
                        input_ids = torch.tensor([[tokenizer.bos_token_id]]).to(device)
                    
                    # Get text embeddings
                    text_embeddings = model.llm.get_input_embeddings()(input_ids)
                    
                    # Combine visual and text embeddings
                    combined_embeddings = torch.cat([visual_tokens, text_embeddings], dim=1)
                    
                    # Create routing masks (vision=0, text=1)
                    num_visual_tokens = visual_tokens.shape[1]
                    num_text_tokens = input_ids.shape[1]
                    routing_mask = torch.cat([
                        torch.zeros(num_visual_tokens, dtype=torch.long, device=device),
                        torch.ones(num_text_tokens, dtype=torch.long, device=device)
                    ]).unsqueeze(0)
                    
                    # Create attention mask
                    attention_mask = torch.ones(combined_embeddings.shape[:2], device=device)
                    
                    # Set routing mask ONLY for Stage 2 (hard routing)
                    if not is_stage3:
                        for layer in model.llm.model.layers:
                            layer.mlp.routing_mask = routing_mask
                    
                    # Generate tokens autoregressively
                    # Track all generated tokens (for Stage 3, include the prompt)
                    generated_ids = input_ids[0].tolist()
                    
                    # Get period token ID for early stopping (cleaner single-sentence captions)
                    period_tokens = tokenizer.encode(".", add_special_tokens=False)
                    period_id = period_tokens[0] if period_tokens else None
                    
                    for step in range(max_length - 1):
                        # Forward pass
                        outputs = model.llm(
                            inputs_embeds=combined_embeddings,
                            attention_mask=attention_mask,
                            use_cache=False
                        )
                        
                        # Get next token logits
                        next_token_logits = outputs.logits[:, -1, :]
                        
                        # Apply temperature and get next token
                        if temperature > 0:
                            next_token_logits = next_token_logits / temperature
                        next_token = torch.argmax(next_token_logits, dim=-1)
                        
                        # Stop if EOS
                        if next_token.item() == tokenizer.eos_token_id:
                            break
                        
                        generated_ids.append(next_token.item())
                        
                        # Early stopping: stop after first sentence (period token)
                        # This produces cleaner, more COCO-like captions
                        if period_id is not None and next_token.item() == period_id:
                            # Check if we have at least a few words (avoid stopping on abbreviations)
                            if len(generated_ids) >= 8:  # At least ~6-7 words
                                break
                        
                        # Append token embedding for next step
                        next_token_embedding = model.llm.get_input_embeddings()(next_token.unsqueeze(0))
                        combined_embeddings = torch.cat([combined_embeddings, next_token_embedding], dim=1)
                        
                        # Update routing mask
                        routing_mask = torch.cat([
                            routing_mask,
                            torch.ones((1, 1), dtype=torch.long, device=device)
                        ], dim=1)
                        
                        # Update attention mask
                        attention_mask = torch.cat([
                            attention_mask,
                            torch.ones((1, 1), device=device)
                        ], dim=1)
                        
                        # Update routing mask ONLY for Stage 2
                        if not is_stage3:
                            for layer in model.llm.model.layers:
                                layer.mlp.routing_mask = routing_mask
                    
                    # Decode generated sequence
                    caption = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    caption = caption.strip()
                    
                    # For Stage 3, remove the prompt from the generated caption
                    if is_stage3:
                        prompt_text = "A photo of"
                        if caption.startswith(prompt_text):
                            caption = caption[len(prompt_text):].strip()
                    
                    # Get corresponding image entry
                    data_idx = valid_indices[batch_idx]
                    img_entry = batch_data[data_idx]
                    
                    results.append({
                        'image_id': img_entry['coco_id'],
                        'caption': caption
                    })
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ Generated {len(results)} captions in {format_time(elapsed)}")
    if len(results) > 0:
        print(f"   Average: {elapsed / len(results):.2f}s per image")
        
        # Show some examples
        print("\n📝 Sample captions:")
        for i in range(min(5, len(results))):
            print(f"   Image {results[i]['image_id']}: {results[i]['caption']}")
    else:
        print("   ⚠️  WARNING: No captions were generated!")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Generate captions for Karpathy test images')
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        required=True,
        help='Path to model checkpoint (.pth file)'
    )
    parser.add_argument(
        '--stage_name',
        type=str,
        required=True,
        choices=['stage2', 'stage3'],
        help='Stage name (stage2 or stage3)'
    )
    parser.add_argument(
        '--images_json',
        type=str,
        default='results/karpathy_evaluation/karpathy_test_images.json',
        help='Path to images JSON'
    )
    parser.add_argument(
        '--image_base_dir',
        type=str,
        default='/data/gpfs/projects/COMP90055/aticinovic/data/coco',
        help='Base directory for images'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/karpathy_evaluation/captioning',
        help='Output directory for generated captions'
    )
    parser.add_argument(
        '--num_beams',
        type=int,
        default=5,
        help='Number of beams for beam search'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=20,
        help='Maximum caption length'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for generation'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device (cuda or cpu)'
    )
    parser.add_argument(
        '--num_images',
        type=int,
        default=None,
        help='Number of images to process (default: all). Use a small number for quick testing.'
    )
    
    args = parser.parse_args()
    
    print_banner(f"CAPTION GENERATION - {args.stage_name.upper()}")
    
    # Load model
    print("\n📦 Loading model...")
    model, processor, tokenizer = load_model_checkpoint(
        args.checkpoint_path,
        device=args.device
    )
    print(f"   Checkpoint: {args.checkpoint_path}")
    
    # Load images data
    print("\n📂 Loading images data...")
    images_data = load_json(args.images_json)
    
    # Optionally limit number of images for testing
    if args.num_images is not None and args.num_images < len(images_data):
        print(f"   Found {len(images_data)} images, using first {args.num_images} for testing")
        images_data = images_data[:args.num_images]
    else:
        print(f"   Found {len(images_data)} images")
    
    # Generate captions
    captions = generate_captions(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        images_data=images_data,
        image_base_dir=args.image_base_dir,
        num_beams=args.num_beams,
        max_length=args.max_length,
        temperature=args.temperature,
        batch_size=args.batch_size,
        device=args.device,
        stage_name=args.stage_name
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f'{args.stage_name}_captions.json'
    save_json(captions, str(output_path))
    
    print(f"\n💾 Saved captions: {output_path}")
    
    print_banner(f"✅ {args.stage_name.upper()} CAPTION GENERATION COMPLETE")


if __name__ == "__main__":
    main()
