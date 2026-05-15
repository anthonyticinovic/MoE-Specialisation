"""
POPE Answer Generation - Stage 3 Priming Strategy

This version exploits Stage 3's learned behavior from LLaVA training.
Stage 3 was trained to generate sequential answers (A1 → A2 → A3) because
questions were masked during training. We can exploit this by:

1. Priming with a fake "previous answer" to make the model think it's in
   the middle of a multi-turn conversation
2. Using the learned pattern to generate the next (actual) answer
3. Forcing yes/no format by constraining generation

Key insight: Stage 3 learned "after one answer, generate another answer"
so we give it a fake first answer and ask for the real answer as the "second" one.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "karpathy_evaluation"))

from karpathy_utils import (
    load_and_preprocess_image,
    load_model_checkpoint,
    print_banner,
    save_json,
)


def extract_yes_no_answer(text: str, question: str = None) -> str:
    """
    Extract yes/no answer from generated text using multiple strategies.

    Args:
        text: Generated text from model
        question: Original question (optional, for context-aware extraction)

    Returns:
        'yes', 'no', or 'unclear'
    """
    text_lower = text.lower().strip()

    # Strategy 1: Direct yes/no at start (most reliable)
    if text_lower.startswith("yes"):
        return "yes"
    if text_lower.startswith("no"):
        return "no"

    # Strategy 2: Pattern matching for common formats
    if re.match(r"^yes[,.\s]", text_lower):
        return "yes"
    if re.match(r"^no[,.\s]", text_lower):
        return "no"

    # Strategy 3: Check first few words
    first_word = text_lower.split()[0] if text_lower.split() else ""
    if first_word in ["yes", "yeah", "yep", "yup"]:
        return "yes"
    if first_word in ["no", "nope", "nah"]:
        return "no"

    # Strategy 4: For Stage 3 - check if it's trying to describe (means it failed to answer)
    descriptive_starts = [
        "the image",
        "there is",
        "there are",
        "this image",
        "in the image",
        "the photo",
        "this photo",
        "a ",
        "an ",
        "it shows",
        "it depicts",
    ]
    for desc_start in descriptive_starts:
        if text_lower.startswith(desc_start):
            # It's generating a description instead of yes/no
            # Try to infer from content if object is mentioned
            if question:
                # Extract object from question: "Is there a dog" -> "dog"
                match = re.search(r"is there (?:a |an )?(\w+)", question.lower())
                if match:
                    queried_object = match.group(1)
                    # Check if object is mentioned in first 80 chars of response
                    if queried_object in text_lower[:80]:
                        # Object mentioned in description = implicit yes
                        # But only if it's clearly referring to THE object
                        # e.g., "the dog is" = yes, but "a dog" might be hallucination
                        if f"the {queried_object}" in text_lower[:80]:
                            return "yes"
            return "unclear"

    # Strategy 5: Contains yes/no somewhere in first sentence
    first_sentence = text_lower.split(".")[0] if "." in text_lower else text_lower
    if "yes" in first_sentence and "no" not in first_sentence:
        return "yes"
    if "no" in first_sentence and "yes" not in first_sentence:
        return "no"

    # Strategy 6: Affirmative/negative words
    affirmative_words = ["correct", "indeed", "absolutely", "certainly"]
    negative_words = ["not", "none", "never", "incorrect"]

    words_in_first_sentence = first_sentence.split()[:10]
    has_affirmative = any(word in affirmative_words for word in words_in_first_sentence)
    has_negative = any(word in negative_words for word in words_in_first_sentence)

    if has_affirmative and not has_negative:
        return "yes"
    if has_negative and not has_affirmative:
        return "no"

    # Unable to determine
    return "unclear"


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
                    predicted_answer = extract_yes_no_answer(generated_text, question_text)

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
    parser = argparse.ArgumentParser(description="Generate POPE answers with priming strategy")
    parser.add_argument("--questions", type=str, required=True, help="Path to questions JSON")
    parser.add_argument("--output", type=str, required=True, help="Path to save answers JSON")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to COCO val2017")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--stage", type=str, default="stage3", choices=["stage2", "stage3"])
    parser.add_argument(
        "--priming",
        type=str,
        default="simple",
        choices=["simple", "conversational", "none"],
        help="Priming strategy",
    )
    parser.add_argument("--max_tokens", type=int, default=10, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    print_banner(f"POPE PRIMING EXPERIMENT - {args.stage.upper()} - {args.priming.upper()}")

    print("\n📋 Configuration:")
    print(f"   Questions: {args.questions}")
    print(f"   Output: {args.output}")
    print(f"   Checkpoint: {args.checkpoint}")
    print(f"   Priming: {args.priming}")

    # Load questions
    print("\n� Loading questions...")
    with open(args.questions) as f:
        questions = json.load(f)
    print(f"   Found {len(questions)} questions")

    # Load model
    print("\n📦 Loading model...")
    model, processor, tokenizer = load_model_checkpoint(args.checkpoint, device=args.device)
    print("   ✅ Model loaded")

    # Generate answers
    results = generate_answers_primed(
        model=model,
        questions=questions,
        image_dir=args.image_dir,
        processor=processor,
        tokenizer=tokenizer,
        device=args.device,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        stage_name=args.stage,
        priming_strategy=args.priming,
    )

    # Save results
    save_json(results, args.output)
    print(f"\n💾 Saved answers: {args.output}")

    print_banner(f"✅ PRIMING EXPERIMENT COMPLETE - {args.priming.upper()}")


if __name__ == "__main__":
    main()
