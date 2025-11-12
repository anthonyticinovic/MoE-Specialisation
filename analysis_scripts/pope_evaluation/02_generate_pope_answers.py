#!/usr/bin/env python3
"""
Step 2: Generate POPE answers from Stage 2 or Stage 3 model.
Generates yes/no answers for POPE questions.
"""

import torch
import argparse
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import time
import json
import sys
import re

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'karpathy_evaluation'))

from karpathy_utils import (
    load_model_checkpoint,
    load_and_preprocess_image,
    save_json,
    format_time,
    print_banner
)


def extract_yes_no_answer(text: str, question: str = None) -> str:
    """
    Extract yes/no answer from model output.
    Handles both concise answers (Stage 2) and elaborate answers (Stage 3).
    
    Args:
        text: Generated text from model
        question: Optional - the original question, used to check if queried object is mentioned
        
    Returns:
        'yes', 'no', or 'unclear'
    """
    text_lower = text.lower().strip()
    
    # Direct matches (most common for Stage 2)
    if text_lower.startswith('yes'):
        return 'yes'
    if text_lower.startswith('no'):
        return 'no'
    
    # Check first few words
    words = text_lower.split()
    if len(words) > 0:
        first_word = words[0].strip('.,!?')
        if first_word == 'yes':
            return 'yes'
        if first_word == 'no':
            return 'no'
    
    # Strong negative indicators (explicit negation)
    strong_negative_phrases = ['there is no', 'there are no', 'there isn\'t', 'there aren\'t', 
                               'not visible', 'cannot see', 'no visible', 'absence of',
                               'no sign of', 'does not show', 'does not feature']
    for phrase in strong_negative_phrases:
        if phrase in text_lower[:80]:
            return 'no'
    
    # Strong affirmative indicators (explicit affirmation)
    strong_affirmative_phrases = ['yes,', 'yes there', 'yes it', 'there is a', 'there are', 
                                  'shows a', 'features a', 'depicts a', 'contains a',
                                  'includes a', 'has a', 'with a', 'shows the', 'features the']
    for phrase in strong_affirmative_phrases:
        if phrase in text_lower[:80]:
            return 'yes'
    
    # For Stage 3: Check if the queried object is actually mentioned in the response
    # This prevents treating generic captions as "yes" answers
    if question:
        import re
        # Extract object from question: "Is there a/an X in the image?"
        match = re.search(r'is there (?:a |an )?(\w+)', question.lower())
        if match:
            queried_object = match.group(1)
            object_mentioned = queried_object in text_lower[:80]
            
            if object_mentioned:
                # Object is mentioned - check if it's in a descriptive context (likely yes)
                # vs. a negative context
                # If definite article precedes object, it's describing it (yes)
                if f'the {queried_object}' in text_lower[:80]:
                    return 'yes'
                # If possessive or descriptive phrase
                if any(p in text_lower[:80] for p in [f'{queried_object} is', f'{queried_object} in',
                                                        f'{queried_object} on', f'{queried_object} at']):
                    return 'yes'
            else:
                # Object NOT mentioned but we have descriptive text
                # Could be describing something else in the image (maybe no, maybe unclear)
                # If it's a generic truncated description, mark as unclear
                if len(text_lower) < 20 or text_lower.endswith((',', 'and', 'or', 'with', 'in', 'a')):
                    # Truncated or incomplete - unclear
                    return 'unclear'
                # If it describes other objects, tentatively say no
                # But this is weak evidence
                return 'unclear'
    
    # Fallback: If no strong indicators and no question context, check generic patterns
    # Descriptive patterns WITHOUT object mention are unclear (could be anything)
    descriptive_patterns = ['the image features', 'the image shows', 'the image depicts',
                           'the scene features', 'the scene shows']
    for pattern in descriptive_patterns:
        if pattern in text_lower[:50]:
            # Generic description without clear answer - unclear
            return 'unclear'
    
    # Definite article at start suggests describing something, but we don't know what
    if text_lower.startswith('the ') and len(words) > 2:
        # Could be describing the queried object or something else
        return 'unclear'
    
    return 'unclear'


def generate_pope_answers(
    model,
    processor,
    tokenizer,
    questions: List[Dict],
    image_dir: str,
    max_new_tokens: int = 10,
    temperature: float = 0.0,  # Greedy decoding for yes/no
    batch_size: int = 1,  # Process one at a time for yes/no questions
    device: str = 'cuda',
    stage_name: str = 'stage2'
) -> List[Dict]:
    """
    Generate yes/no answers for POPE questions.
    
    Args:
        model: The vision-language model
        processor: Image processor
        tokenizer: Text tokenizer
        questions: List of POPE questions with 'image_id', 'question', 'answer'
        image_dir: Directory containing val2017 images
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 for greedy)
        batch_size: Batch size (1 for yes/no QA)
        device: Device to use
        stage_name: 'stage2' or 'stage3'
        
    Returns:
        results: List of questions with 'predicted_answer' added
    """
    model.eval()
    model.to(device)
    
    results = []
    num_questions = len(questions)
    
    print(f"\n🔮 Generating POPE answers for {num_questions} questions...")
    print(f"   Max tokens: {max_new_tokens}")
    print(f"   Temperature: {temperature} ({'greedy' if temperature == 0 else 'sampling'})")
    print(f"   Device: {device}")
    print(f"   Stage: {stage_name}")
    
    # Detect stage type for routing
    is_stage3 = (stage_name == 'stage3')
    if is_stage3:
        print("   📍 Using Stage 3 (SOFT routing)")
        # Set all MoE layers to soft routing mode
        for layer in model.llm.model.layers:
            if hasattr(layer.mlp, "routing_mode"):
                layer.mlp.routing_mode = 'soft'
    else:
        print("   📍 Using Stage 2 (HARD routing)")
    
    start_time = time.time()
    
    # Group questions by image_id for efficiency
    questions_by_image = {}
    for q in questions:
        img_id = q['image_id']
        if img_id not in questions_by_image:
            questions_by_image[img_id] = []
        questions_by_image[img_id].append(q)
    
    unique_images = list(questions_by_image.keys())
    print(f"   Processing {len(unique_images)} unique images")
    
    unclear_count = 0
    
    with torch.no_grad():
        for img_id in tqdm(unique_images, desc="Answering"):
            # Load image once for all questions about this image
            # COCO val2017 filename format: 000000XXXXXX.jpg
            image_filename = f"{img_id:012d}.jpg"
            image_path = Path(image_dir) / image_filename
            
            if not image_path.exists():
                print(f"\n⚠️  Image not found: {image_path}")
                # Mark all questions for this image as unclear
                for question_data in questions_by_image[img_id]:
                    result = question_data.copy()
                    result['predicted_answer'] = 'unclear'
                    results.append(result)
                continue
            
            try:
                # Load and preprocess image
                image = load_and_preprocess_image(str(image_path), processor)
                pixel_values = image.unsqueeze(0).to(device)  # Add batch dim
                
                # Process vision through encoder and connector
                vision_outputs = model.vision_encoder(pixel_values=pixel_values)
                patch_embeddings = vision_outputs.last_hidden_state  # (1, 257, 1024)
                
                # Project to LLM space
                visual_soft_tokens = model.connector(patch_embeddings)
                visual_soft_tokens = visual_soft_tokens.to(torch.bfloat16)
                
                # Answer each question about this image
                for question_data in questions_by_image[img_id]:
                    question_text = question_data['question']
                    
                    # Format prompt for yes/no question
                    # For Stage 3 (instruction-tuned): Use strong, explicit anti-hallucination prompt
                    # For Stage 2 (caption-only): Simple prompting
                    if is_stage3:
                        # Strong anti-hallucination prompt with explicit constraints
                        prompt = f"{question_text}\nAnswer only 'yes' or 'no'. Do not generate descriptions."
                    else:
                        prompt = f"{question_text} Answer:"
                    
                    # Tokenize prompt
                    prompt_ids = tokenizer(
                        prompt,
                        return_tensors='pt',
                        add_special_tokens=False
                    ).input_ids.to(device)
                    
                    # Add BOS token
                    bos_ids = torch.tensor([[tokenizer.bos_token_id]]).to(device)
                    input_ids = torch.cat([bos_ids, prompt_ids], dim=1)
                    
                    # Get text embeddings
                    text_embeddings = model.llm.get_input_embeddings()(input_ids)
                    
                    # Combine visual and text embeddings
                    combined_embeddings = torch.cat([visual_soft_tokens, text_embeddings], dim=1)
                    
                    # Create routing masks (vision=0, text=1)
                    num_visual_tokens = visual_soft_tokens.shape[1]
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
                    
                    # Generate answer tokens autoregressively
                    generated_ids = input_ids[0].tolist()
                    
                    for step in range(max_new_tokens):
                        # Forward pass
                        outputs = model.llm(
                            inputs_embeds=combined_embeddings,
                            attention_mask=attention_mask,
                            use_cache=False
                        )
                        
                        # Get next token logits
                        next_token_logits = outputs.logits[:, -1, :]
                        
                        # Apply temperature
                        if temperature > 0:
                            next_token_logits = next_token_logits / temperature
                            next_token = torch.multinomial(
                                torch.softmax(next_token_logits, dim=-1),
                                num_samples=1
                            ).squeeze(-1)
                        else:
                            # Greedy decoding
                            next_token = torch.argmax(next_token_logits, dim=-1)
                        
                        # Stop if EOS or newline (yes/no answers are short)
                        if next_token.item() == tokenizer.eos_token_id:
                            break
                        
                        generated_ids.append(next_token.item())
                        
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
                    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    
                    # Remove the prompt to get just the answer
                    if prompt in generated_text:
                        answer_text = generated_text[len(prompt):].strip()
                    else:
                        answer_text = generated_text.strip()
                    
                    # Extract yes/no answer (pass question for context-aware extraction)
                    predicted = extract_yes_no_answer(answer_text, question_text)
                    
                    if predicted == 'unclear':
                        unclear_count += 1
                    
                    # Add result
                    result = question_data.copy()
                    result['predicted_answer'] = predicted
                    result['raw_output'] = answer_text
                    results.append(result)
                    
            except Exception as e:
                print(f"\n❌ Error processing image {img_id}: {e}")
                # Mark all questions for this image as unclear
                for question_data in questions_by_image[img_id]:
                    result = question_data.copy()
                    result['predicted_answer'] = 'unclear'
                    result['raw_output'] = f'ERROR: {str(e)}'
                    results.append(result)
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ Generated {len(results)} answers in {format_time(elapsed)}")
    print(f"   Average: {elapsed / len(results):.2f}s per question")
    print(f"   Unclear answers: {unclear_count} ({unclear_count/len(results)*100:.1f}%)")
    
    # Show some examples
    print(f"\n📝 Sample answers:")
    for i in range(min(5, len(results))):
        q = results[i]
        print(f"   Q: {q['question']}")
        print(f"   A: {q['predicted_answer']} (GT: {q['answer']}) | Raw: {q['raw_output'][:50]}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Generate POPE answers using trained model')
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
        '--questions_file',
        type=str,
        required=True,
        help='Path to POPE questions JSON file'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default='/data/gpfs/projects/COMP90055/aticinovic/datasets/coco/val2017',
        help='Directory containing COCO val2017 images'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/pope_evaluation',
        help='Output directory for answers'
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=10,
        help='Maximum new tokens to generate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Sampling temperature (0.0 for greedy)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device (cuda or cpu)'
    )
    
    args = parser.parse_args()
    
    print_banner(f"POPE ANSWER GENERATION - {args.stage_name.upper()}")
    
    # Load model
    print("\n📦 Loading model...")
    model, processor, tokenizer = load_model_checkpoint(
        args.checkpoint_path,
        device=args.device
    )
    print(f"   Checkpoint: {args.checkpoint_path}")
    
    # Load questions
    print(f"\n📂 Loading questions from: {args.questions_file}")
    with open(args.questions_file, 'r') as f:
        questions = json.load(f)
    print(f"   Found {len(questions)} questions")
    
    # Generate answers
    answers = generate_pope_answers(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        questions=questions,
        image_dir=args.image_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device,
        stage_name=args.stage_name
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract difficulty from filename (e.g., pope_random.json -> random)
    questions_filename = Path(args.questions_file).stem  # e.g., "pope_random"
    difficulty = questions_filename.split('_')[-1] if '_' in questions_filename else 'unknown'
    
    output_path = output_dir / f'{args.stage_name}_{difficulty}_answers.json'
    save_json(answers, str(output_path))
    
    print(f"\n💾 Saved answers: {output_path}")
    
    print_banner(f"✅ {args.stage_name.upper()} ANSWER GENERATION COMPLETE")


if __name__ == "__main__":
    main()
