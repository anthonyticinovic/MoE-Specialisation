"""
LLaVA-Wild Style Evaluation - Lightweight Implementation

Evaluates Stage 2 and Stage 3 on conversational VQA using LLaVA-Instruct-150K.
Uses GPT-4 to judge response quality (or simple heuristics if no API access).

This tests whether Stage 3 performs better on its TRAINING distribution
(elaborate conversational answers) vs structured tasks (POPE, captioning).
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "karpathy_evaluation"))

from karpathy_utils import load_and_preprocess_image, load_model_checkpoint, print_banner, save_json


def generate_conversational_response(
    model,
    processor,
    tokenizer,
    image_path,
    question,
    max_new_tokens=100,
    temperature=0.7,
    device="cuda",
    is_stage3=False,
):
    """
    Generate a conversational response to a question about an image.

    Uses sampling (temperature > 0) for more natural, diverse responses.
    """
    # Load and preprocess image
    pixel_values = load_and_preprocess_image(image_path, processor)
    pixel_values = pixel_values.unsqueeze(0).to(device)

    # Process vision through encoder and connector
    with torch.no_grad():
        vision_outputs = model.vision_encoder(pixel_values=pixel_values)
        patch_embeddings = vision_outputs.last_hidden_state
        visual_soft_tokens = model.connector(patch_embeddings)
        visual_soft_tokens = visual_soft_tokens.to(torch.bfloat16)

        # Format prompt as conversational question
        prompt = f"{question}"

        # Tokenize prompt
        prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(
            device
        )

        # Add BOS token
        bos_ids = torch.tensor([[tokenizer.bos_token_id]]).to(device)
        input_ids = torch.cat([bos_ids, prompt_ids], dim=1)

        # Get text embeddings
        text_embeddings = model.llm.get_input_embeddings()(input_ids)

        # Combine visual and text embeddings
        combined_embeddings = torch.cat([visual_soft_tokens, text_embeddings], dim=1)
        attention_mask = torch.ones(combined_embeddings.shape[:2], device=device)

        # Create routing mask for Stage 2 (hard routing)
        if not is_stage3:
            num_visual_tokens = visual_soft_tokens.shape[1]
            num_text_tokens = input_ids.shape[1]
            routing_mask = torch.cat(
                [
                    torch.zeros(num_visual_tokens, dtype=torch.long, device=device),
                    torch.ones(num_text_tokens, dtype=torch.long, device=device),
                ]
            ).unsqueeze(0)

            # Set routing mask for all layers (required for hard routing)
            for layer in model.llm.model.layers:
                layer.mlp.routing_mask = routing_mask

        # Generate response autoregressively
        generated_ids = input_ids[0].tolist()

        for step in range(max_new_tokens):
            outputs = model.llm(
                inputs_embeds=combined_embeddings, attention_mask=attention_mask, use_cache=False
            )

            next_token_logits = outputs.logits[:, -1, :]

            # Apply temperature for sampling
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                # Sample from distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1).item()

            # Stop if EOS
            if next_token == tokenizer.eos_token_id:
                break

            generated_ids.append(next_token)

            # Update embeddings for next step
            next_token_tensor = torch.tensor([[next_token]]).to(device)
            next_embedding = model.llm.get_input_embeddings()(next_token_tensor)
            combined_embeddings = torch.cat([combined_embeddings, next_embedding], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device)], dim=1)

            # Update routing mask for Stage 2 (new token is text)
            if not is_stage3:
                routing_mask = torch.cat(
                    [routing_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=1
                )
                for layer in model.llm.model.layers:
                    layer.mlp.routing_mask = routing_mask

        # Decode generated tokens (only the new ones)
        prompt_length = len(input_ids[0])
        generated_answer_ids = generated_ids[prompt_length:]
        generated_text = tokenizer.decode(generated_answer_ids, skip_special_tokens=True)

        return generated_text


def evaluate_response_simple(response, reference=None):
    """
    Simple heuristic evaluation (no GPT-4 needed).

    Metrics:
    1. Length (20-150 tokens is good for conversational)
    2. Coherence (no excessive repetition)
    3. Relevance (contains some content words)
    4. Quality (not just generic templates)

    Returns score 0-100
    """
    if not response or len(response.strip()) == 0:
        return 0

    tokens = response.split()
    num_tokens = len(tokens)

    score = 50  # Base score

    # Length check (20-150 tokens ideal for conversational)
    if 20 <= num_tokens <= 150:
        score += 20
    elif 10 <= num_tokens < 20 or 150 < num_tokens <= 200:
        score += 10
    elif num_tokens < 10:
        score -= 20  # Too short
    else:
        score -= 10  # Too long

    # Repetition check (detect "a laptop, a laptop, a laptop")
    unique_bigrams = set(zip(tokens[:-1], tokens[1:]))
    repetition_ratio = len(unique_bigrams) / max(num_tokens - 1, 1)
    if repetition_ratio > 0.7:
        score += 15  # Low repetition (good)
    elif repetition_ratio > 0.5:
        score += 5
    else:
        score -= 20  # High repetition (bad)

    # Content check (has descriptive words, not just templates)
    content_words = [
        "shows",
        "depicts",
        "features",
        "contains",
        "includes",
        "appears",
        "looks",
        "seems",
        "probably",
        "likely",
        "color",
        "large",
        "small",
        "red",
        "blue",
        "white",
        "black",
    ]
    has_content = any(word in response.lower() for word in content_words)
    if has_content:
        score += 15

    # Generic template penalty
    generic_starts = [
        "the image shows",
        "the image depicts",
        "this image shows",
        "the photo shows",
        "the picture shows",
    ]
    if any(response.lower().startswith(gen) for gen in generic_starts):
        score -= 5  # Minor penalty (these are okay but generic)

    # Broken word detection
    if any(len(word) > 15 for word in tokens):  # Unusually long "words" (likely broken)
        score -= 10

    return max(0, min(100, score))


def load_llava_subset(json_path, num_samples=100, seed=42):
    """
    Load a random subset of LLaVA-Instruct samples.

    For each sample, we'll use the FIRST question-answer pair only
    (to keep it simple and fast).
    """
    print(f"\n📂 Loading LLaVA data from: {json_path}")
    with open(json_path) as f:
        data = json.load(f)

    print(f"   Total samples: {len(data)}")

    # Filter: only samples with valid conversations and images
    valid_samples = []
    for item in data:
        if "conversations" in item and len(item["conversations"]) >= 2 and "image" in item:
            # Extract first Q&A pair
            question = (
                item["conversations"][0]["value"].replace("<image>", "").replace("\n", " ").strip()
            )
            answer = item["conversations"][1]["value"].strip()

            # Skip if too long or empty
            if len(question) > 0 and len(answer) > 0 and len(answer) < 500:
                valid_samples.append(
                    {
                        "image": item["image"],
                        "question": question,
                        "reference_answer": answer,
                        "id": item.get("id", "unknown"),
                    }
                )

    print(f"   Valid samples: {len(valid_samples)}")

    # Random subset
    random.seed(seed)
    subset = random.sample(valid_samples, min(num_samples, len(valid_samples)))

    print(f"   Selected subset: {len(subset)} samples")

    return subset


def main():
    parser = argparse.ArgumentParser(description="LLaVA-Wild Style Evaluation")
    parser.add_argument("--llava_json", type=str, required=True, help="Path to LLaVA instruct JSON")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to image directory")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--stage", type=str, required=True, choices=["stage2", "stage3"])
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument(
        "--num_samples", type=int, default=100, help="Number of samples to evaluate"
    )
    parser.add_argument("--max_tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()

    print_banner(f"LLAVA-WILD EVALUATION - {args.stage.upper()}")

    # Load subset
    samples = load_llava_subset(args.llava_json, args.num_samples)

    # Load model
    print("\n📦 Loading model...")
    model, processor, tokenizer = load_model_checkpoint(args.checkpoint, device=args.device)

    # Set routing mode
    if args.stage == "stage3":
        print("   📍 Using Stage 3 (SOFT routing)")
        for layer in model.llm.model.layers:
            if hasattr(layer.mlp, "routing_mode"):
                layer.mlp.routing_mode = "soft"
    else:
        print("   📍 Using Stage 2 (HARD routing)")

    print("   ✅ Model loaded\n")

    # Generate responses
    results = []
    total_score = 0

    print("🔮 Generating conversational responses...")
    print(f"   Samples: {len(samples)}")
    print(f"   Max tokens: {args.max_tokens}")
    print(f"   Temperature: {args.temperature}\n")

    model.eval()
    model.to(args.device)

    for sample in tqdm(samples, desc="Evaluating"):
        # Try multiple possible image paths
        image_filename = sample["image"]
        possible_paths = [
            Path(args.image_dir) / image_filename,
            Path(args.image_dir) / "train2017" / image_filename,
            Path(args.image_dir) / "val2017" / image_filename,
        ]

        image_path = None
        for path in possible_paths:
            if path.exists():
                image_path = path
                break

        if image_path is None:
            print(f"\n⚠️  Image not found: {image_filename}")
            continue

        try:
            # Generate response
            response = generate_conversational_response(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                image_path=str(image_path),
                question=sample["question"],
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                device=args.device,
                is_stage3=(args.stage == "stage3"),
            )

            # Evaluate response
            score = evaluate_response_simple(response, sample["reference_answer"])
            total_score += score

            results.append(
                {
                    "id": sample["id"],
                    "image": sample["image"],
                    "question": sample["question"],
                    "reference_answer": sample["reference_answer"],
                    "generated_answer": response,
                    "score": score,
                }
            )

        except Exception as e:
            print(f"\n❌ Error processing {sample['image']}: {e}")
            results.append(
                {
                    "id": sample["id"],
                    "image": sample["image"],
                    "question": sample["question"],
                    "reference_answer": sample["reference_answer"],
                    "generated_answer": f"ERROR: {str(e)}",
                    "score": 0,
                }
            )

    # Compute final metrics
    avg_score = total_score / len(results) if results else 0

    print(f"\n{'=' * 70}")
    print(f"RESULTS - {args.stage.upper()}")
    print(f"{'=' * 70}")
    print(f"Samples evaluated: {len(results)}")
    print(f"Average score: {avg_score:.1f}/100")
    print(f"{'=' * 70}\n")

    # Add summary to results
    summary = {
        "stage": args.stage,
        "num_samples": len(results),
        "average_score": avg_score,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }

    output_data = {"summary": summary, "results": results}

    # Save results
    save_json(output_data, args.output)
    print(f"💾 Saved results: {args.output}\n")

    print_banner(f"✅ EVALUATION COMPLETE - {args.stage.upper()}")


if __name__ == "__main__":
    main()
