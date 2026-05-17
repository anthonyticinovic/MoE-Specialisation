#!/usr/bin/env python3
"""
Step 2: Generate POPE answers from Stage 2 or Stage 3 model.
Generates yes/no answers for POPE questions.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "karpathy_evaluation"))
sys.path.insert(0, str(Path(__file__).parent))

from karpathy_utils import (
    format_time,
    load_and_preprocess_image,
    load_model_checkpoint,
    print_banner,
    save_json,
)
from pope_utils import extract_yes_no_answer, extract_yes_no_answer_primed


def generate_pope_answers(
    model,
    processor,
    tokenizer,
    questions: list[dict],
    image_dir: str,
    max_new_tokens: int = 10,
    temperature: float = 0.0,  # Greedy decoding for yes/no
    batch_size: int = 1,  # Process one at a time for yes/no questions
    device: str = "cuda",
    stage_name: str = "stage2",
) -> list[dict]:
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
    is_stage3 = stage_name == "stage3"
    if is_stage3:
        print("   📍 Using Stage 3 (SOFT routing)")
        # Set all MoE layers to soft routing mode
        for layer in model.llm.model.layers:
            if hasattr(layer.mlp, "routing_mode"):
                layer.mlp.routing_mode = "soft"
    else:
        print("   📍 Using Stage 2 (HARD routing)")

    start_time = time.time()

    # Group questions by image_id for efficiency
    questions_by_image = {}
    for q in questions:
        img_id = q["image_id"]
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
                    result["predicted_answer"] = "unclear"
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
                    question_text = question_data["question"]

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
                        prompt, return_tensors="pt", add_special_tokens=False
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
                    routing_mask = torch.cat(
                        [
                            torch.zeros(num_visual_tokens, dtype=torch.long, device=device),
                            torch.ones(num_text_tokens, dtype=torch.long, device=device),
                        ]
                    ).unsqueeze(0)

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
                            use_cache=False,
                        )

                        # Get next token logits
                        next_token_logits = outputs.logits[:, -1, :]

                        # Apply temperature
                        if temperature > 0:
                            next_token_logits = next_token_logits / temperature
                            next_token = torch.multinomial(
                                torch.softmax(next_token_logits, dim=-1), num_samples=1
                            ).squeeze(-1)
                        else:
                            # Greedy decoding
                            next_token = torch.argmax(next_token_logits, dim=-1)

                        # Stop if EOS or newline (yes/no answers are short)
                        if next_token.item() == tokenizer.eos_token_id:
                            break

                        generated_ids.append(next_token.item())

                        # Append token embedding for next step
                        next_token_embedding = model.llm.get_input_embeddings()(
                            next_token.unsqueeze(0)
                        )
                        combined_embeddings = torch.cat(
                            [combined_embeddings, next_token_embedding], dim=1
                        )

                        # Update routing mask
                        routing_mask = torch.cat(
                            [routing_mask, torch.ones((1, 1), dtype=torch.long, device=device)],
                            dim=1,
                        )

                        # Update attention mask
                        attention_mask = torch.cat(
                            [attention_mask, torch.ones((1, 1), device=device)], dim=1
                        )

                        # Update routing mask ONLY for Stage 2
                        if not is_stage3:
                            for layer in model.llm.model.layers:
                                layer.mlp.routing_mask = routing_mask

                    # Decode generated sequence
                    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

                    # Remove the prompt to get just the answer
                    if prompt in generated_text:
                        answer_text = generated_text[len(prompt) :].strip()
                    else:
                        answer_text = generated_text.strip()

                    # Extract yes/no answer (pass question for context-aware extraction)
                    predicted = extract_yes_no_answer(answer_text, question_text)

                    if predicted == "unclear":
                        unclear_count += 1

                    # Add result
                    result = question_data.copy()
                    result["predicted_answer"] = predicted
                    result["raw_output"] = answer_text
                    results.append(result)

            except Exception as e:
                print(f"\n❌ Error processing image {img_id}: {e}")
                # Mark all questions for this image as unclear
                for question_data in questions_by_image[img_id]:
                    result = question_data.copy()
                    result["predicted_answer"] = "unclear"
                    result["raw_output"] = f"ERROR: {str(e)}"
                    results.append(result)

    elapsed = time.time() - start_time

    print(f"\n✅ Generated {len(results)} answers in {format_time(elapsed)}")
    print(f"   Average: {elapsed / len(results):.2f}s per question")
    print(f"   Unclear answers: {unclear_count} ({unclear_count / len(results) * 100:.1f}%)")

    # Show some examples
    print("\n📝 Sample answers:")
    for i in range(min(5, len(results))):
        q = results[i]
        print(f"   Q: {q['question']}")
        print(f"   A: {q['predicted_answer']} (GT: {q['answer']}) | Raw: {q['raw_output'][:50]}")

    return results


def generate_answers_primed(
    model,
    questions,
    image_dir,
    processor,
    tokenizer,
    device="cuda",
    max_new_tokens=10,
    temperature=0.0,
    stage_name="stage3",
    priming_strategy="simple",
):
    """
    Generate POPE answers using priming strategy for Stage 3.

    Priming strategies:
    - 'simple': Prime with "The image contains objects."
    - 'conversational': Prime with full Q&A pair
    - 'none': No priming (baseline for comparison)

    Args:
        model: VLM model
        questions: List of question dicts with 'image_id', 'question', 'answer'
        image_dir: Path to COCO val2017 images
        processor: CLIP processor
        tokenizer: Text tokenizer
        device: Device to run on
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = greedy)
        stage_name: 'stage2' or 'stage3'
        priming_strategy: 'simple', 'conversational', or 'none'

    Returns:
        results: List of questions with 'predicted_answer' added
    """
    model.eval()
    model.to(device)

    results = []
    num_questions = len(questions)

    print("\n🔮 Generating POPE answers with PRIMING strategy")
    print(f"   Questions: {num_questions}")
    print(f"   Max tokens: {max_new_tokens}")
    print(f"   Temperature: {temperature} ({'greedy' if temperature == 0 else 'sampling'})")
    print(f"   Device: {device}")
    print(f"   Stage: {stage_name}")
    print(f"   Priming: {priming_strategy}")

    # Detect stage type for routing
    is_stage3 = stage_name == "stage3"
    if is_stage3:
        print("   📍 Using Stage 3 (SOFT routing)")
        # Set all MoE layers to soft routing mode
        for layer in model.llm.model.layers:
            if hasattr(layer.mlp, "routing_mode"):
                layer.mlp.routing_mode = "soft"
    else:
        print("   📍 Using Stage 2 (HARD routing)")

    start_time = time.time()

    # Group questions by image_id for efficiency
    questions_by_image = {}
    for q in questions:
        img_id = q["image_id"]
        if img_id not in questions_by_image:
            questions_by_image[img_id] = []
        questions_by_image[img_id].append(q)

    unique_images = list(questions_by_image.keys())
    print(f"   Processing {len(unique_images)} unique images")

    unclear_count = 0

    with torch.no_grad():
        for img_id in tqdm(unique_images, desc="Answering"):
            # Load image once for all questions about this image
            image_filename = f"{img_id:012d}.jpg"
            image_path = Path(image_dir) / image_filename

            if not image_path.exists():
                print(f"\n⚠️  Image not found: {image_path}")
                for question_data in questions_by_image[img_id]:
                    result = question_data.copy()
                    result["predicted_answer"] = "unclear"
                    results.append(result)
                continue

            try:
                # Load and preprocess image
                pixel_values = load_and_preprocess_image(str(image_path), processor)
                pixel_values = pixel_values.unsqueeze(0).to(device)

                # Process vision through encoder and connector
                vision_outputs = model.vision_encoder(pixel_values=pixel_values)
                patch_embeddings = vision_outputs.last_hidden_state

                # Project to LLM space
                visual_soft_tokens = model.connector(patch_embeddings)
                visual_soft_tokens = visual_soft_tokens.to(torch.bfloat16)

                # Answer each question about this image
                for question_data in questions_by_image[img_id]:
                    question_text = question_data["question"]

                    # Construct prompt with priming
                    if priming_strategy == "simple":
                        # Prime with a simple statement, then ask the actual question
                        # This mimics the A1 → A2 pattern Stage 3 learned
                        primer = "This image has been analyzed."
                        prompt = f"{primer} {question_text} Answer yes or no:"

                    elif priming_strategy == "conversational":
                        # Prime with full Q&A to mimic LLaVA multi-turn format
                        # Q1: Generic, A1: Generic, Q2: Actual question
                        primer_q = "What type of image is this?"
                        primer_a = "This is a photograph."
                        prompt = f"{primer_q} {primer_a} {question_text} Answer:"

                    else:  # 'none'
                        # No priming (baseline)
                        prompt = f"{question_text} Answer yes or no:"

                    # Tokenize prompt
                    prompt_ids = tokenizer(
                        prompt, return_tensors="pt", add_special_tokens=False
                    ).input_ids.to(device)

                    # Add BOS token
                    bos_ids = torch.tensor([[tokenizer.bos_token_id]]).to(device)
                    input_ids = torch.cat([bos_ids, prompt_ids], dim=1)

                    # Get text embeddings
                    text_embeddings = model.llm.get_input_embeddings()(input_ids)

                    # Combine visual and text embeddings
                    combined_embeddings = torch.cat([visual_soft_tokens, text_embeddings], dim=1)

                    # Create attention mask
                    attention_mask = torch.ones(combined_embeddings.shape[:2], device=device)

                    # Generate answer tokens autoregressively
                    generated_ids = input_ids[0].tolist()

                    for step in range(max_new_tokens):
                        # Forward pass
                        outputs = model.llm(
                            inputs_embeds=combined_embeddings,
                            attention_mask=attention_mask,
                            use_cache=False,
                        )

                        # Get next token logits
                        next_token_logits = outputs.logits[:, -1, :]

                        # Apply temperature
                        if temperature > 0:
                            next_token_logits = next_token_logits / temperature
                            next_token = torch.multinomial(
                                torch.softmax(next_token_logits, dim=-1), num_samples=1
                            ).item()
                        else:
                            next_token = torch.argmax(next_token_logits, dim=-1).item()

                        # Stop if EOS
                        if next_token == tokenizer.eos_token_id:
                            break

                        # Append token
                        generated_ids.append(next_token)

                        # Update embeddings for next step
                        next_token_tensor = torch.tensor([[next_token]]).to(device)
                        next_embedding = model.llm.get_input_embeddings()(next_token_tensor)
                        combined_embeddings = torch.cat(
                            [combined_embeddings, next_embedding], dim=1
                        )
                        attention_mask = torch.cat(
                            [attention_mask, torch.ones((1, 1), device=device)], dim=1
                        )

                    # Decode generated tokens (only the new ones)
                    prompt_length = len(input_ids[0])
                    generated_answer_ids = generated_ids[prompt_length:]
                    generated_text = tokenizer.decode(
                        generated_answer_ids, skip_special_tokens=True
                    )

                    # Extract yes/no from generated text
                    predicted_answer = extract_yes_no_answer_primed(generated_text, question_text)

                    if predicted_answer == "unclear":
                        unclear_count += 1

                    # Store result
                    result = question_data.copy()
                    result["predicted_answer"] = predicted_answer
                    result["raw_model_output"] = generated_text
                    result["priming_strategy"] = priming_strategy
                    results.append(result)

            except Exception as e:
                print(f"\n❌ Error processing image {img_id}: {e}")
                for question_data in questions_by_image[img_id]:
                    result = question_data.copy()
                    result["predicted_answer"] = "unclear"
                    result["raw_model_output"] = f"ERROR: {str(e)}"
                    results.append(result)

    elapsed = time.time() - start_time
    unclear_pct = (unclear_count / len(results)) * 100 if results else 0

    print(f"\n✅ Generated {len(results)} answers in {elapsed / 60:.1f} minutes")
    print(f"   Unclear answers: {unclear_count} ({unclear_pct:.1f}%)")
    print(f"   Speed: {len(results) / elapsed:.2f} questions/sec")

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate POPE answers using trained model")
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--stage_name",
        type=str,
        required=True,
        choices=["stage2", "stage3"],
        help="Stage name (stage2 or stage3)",
    )
    parser.add_argument(
        "--questions_file", type=str, required=True, help="Path to POPE questions JSON file"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="COCO val2017 image dir (default: <parent of paths.image_dir>/val2017 from config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/pope_evaluation",
        help="Output directory for answers",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=10, help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature (0.0 for greedy)"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument(
        "--use-priming",
        action="store_true",
        help="Use the Stage-3 priming generation strategy (default: standard prompting)",
    )
    parser.add_argument(
        "--priming",
        type=str,
        default="simple",
        choices=["simple", "conversational", "none"],
        help="Priming strategy (only used with --use-priming)",
    )

    args = parser.parse_args()

    if args.image_dir is None:
        from analysis_scripts._lib import get_paths

        args.image_dir = str(Path(get_paths()["image_dir"]).parent / "val2017")

    print_banner(f"POPE ANSWER GENERATION - {args.stage_name.upper()}")

    # Load model
    print("\n📦 Loading model...")
    model, processor, tokenizer = load_model_checkpoint(args.checkpoint_path, device=args.device)
    print(f"   Checkpoint: {args.checkpoint_path}")

    # Load questions
    print(f"\n📂 Loading questions from: {args.questions_file}")
    with open(args.questions_file) as f:
        questions = json.load(f)
    print(f"   Found {len(questions)} questions")

    # Generate answers
    if args.use_priming:
        answers = generate_answers_primed(
            model=model,
            questions=questions,
            image_dir=args.image_dir,
            processor=processor,
            tokenizer=tokenizer,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            stage_name=args.stage_name,
            priming_strategy=args.priming,
        )
    else:
        answers = generate_pope_answers(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            questions=questions,
            image_dir=args.image_dir,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=args.device,
            stage_name=args.stage_name,
        )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract difficulty from filename (e.g., pope_random.json -> random)
    questions_filename = Path(args.questions_file).stem  # e.g., "pope_random"
    difficulty = questions_filename.split("_")[-1] if "_" in questions_filename else "unknown"

    output_path = output_dir / f"{args.stage_name}_{difficulty}_answers.json"
    save_json(answers, str(output_path))

    print(f"\n💾 Saved answers: {output_path}")

    print_banner(f"✅ {args.stage_name.upper()} ANSWER GENERATION COMPLETE")


if __name__ == "__main__":
    main()
