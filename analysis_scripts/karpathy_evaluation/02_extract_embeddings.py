#!/usr/bin/env python3
"""
Step 2: Extract image and text embeddings for retrieval evaluation.
Extracts final layer activations from vision and text encoders.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Running this file directly puts *its own* directory on sys.path, not the repo
# root, so the first-party imports below would fail. Add the root explicitly.
# This replaces sys.path edits that were spread across several modules and
# depended on import order to take effect before they were needed.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis_scripts.karpathy_evaluation.karpathy_utils import (  # noqa: E402
    format_time,
    load_and_preprocess_image,
    load_json,
    load_model_checkpoint,
    log_banner,
    mean_pool_embeddings,
)
from models.utils.common import get_device, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def extract_image_embeddings(
    model,
    processor,
    images_data: list,
    image_base_dir: str,
    layer_idx: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """
    Extract embeddings for all images.

    Args:
        model: Loaded VLM model
        processor: Image processor
        images_data: List of image metadata from JSON
        image_base_dir: Base directory for COCO images
        layer_idx: Which layer to extract from
        batch_size: Batch size for processing
        device: Device to use

    Returns:
        embeddings: (num_images, hidden_dim) numpy array
    """
    model.eval()
    all_embeddings = []

    logger.info(f"\nExtracting image embeddings from layer {layer_idx}...")
    logger.info(f"   Total images: {len(images_data)}")
    logger.info(f"   Batch size: {batch_size}")

    start_time = time.time()

    for i in tqdm(range(0, len(images_data), batch_size), desc="Processing batches"):
        batch_images = images_data[i : i + batch_size]
        batch_pixel_values = []

        # Load and preprocess images
        for img_meta in batch_images:
            filepath = img_meta["filepath"]  # e.g., 'val2014'
            filename = img_meta["filename"]  # e.g., 'COCO_val2014_000000123456.jpg'
            image_path = f"{image_base_dir}/{filepath}/{filename}"

            try:
                pixel_values = load_and_preprocess_image(image_path, processor)
                batch_pixel_values.append(pixel_values)
            except Exception as e:
                logger.warning(f"\n Error loading {image_path}: {e}")
                # Use zero tensor as fallback
                batch_pixel_values.append(torch.zeros(1, 3, 224, 224))

        # Stack batch
        batch_pixel_values = torch.cat(batch_pixel_values, dim=0).to(device)

        # Forward pass through vision encoder
        with torch.no_grad():
            # Get CLIP vision features
            vision_outputs = model.vision_encoder(pixel_values=batch_pixel_values)
            patch_embeddings = vision_outputs.last_hidden_state  # (batch, num_patches, hidden_dim)

            # Project to LLM space through connector
            visual_soft_tokens = model.connector(
                patch_embeddings
            )  # (batch, num_patches, llm_hidden_dim)

            # Convert to bfloat16 to match LLM dtype
            visual_soft_tokens = visual_soft_tokens.to(torch.bfloat16)

            # Set routing mask for vision-only inputs (all 0s = vision expert)
            routing_mask = torch.zeros(
                visual_soft_tokens.shape[:2], dtype=torch.long, device=device
            )
            for layer in model.llm.model.layers:
                layer.mlp.routing_mask = routing_mask

            # Pass through LLM to get final layer representations
            # Note: We use inputs_embeds since we already have embeddings
            outputs = model.llm.model(inputs_embeds=visual_soft_tokens, output_hidden_states=True)

            # Get last layer hidden states
            hidden_states = outputs.hidden_states[-1]  # (batch, num_patches, hidden_dim)

            # Mean pool across spatial tokens
            pooled = mean_pool_embeddings(hidden_states)

            # Convert to float32 before numpy (numpy doesn't support bfloat16)
            all_embeddings.append(pooled.float().cpu().numpy())

    # Concatenate all batches
    embeddings = np.vstack(all_embeddings)

    elapsed = time.time() - start_time
    logger.info(f"   Extracted {len(embeddings)} image embeddings")
    logger.info(f"   Shape: {embeddings.shape}")
    logger.info(f"   Time: {format_time(elapsed)}")

    return embeddings


def extract_text_embeddings(
    model, tokenizer, captions_data: list, layer_idx: int, batch_size: int, device: str
) -> np.ndarray:
    """
    Extract embeddings for all captions.

    Args:
        model: Loaded VLM model
        tokenizer: Text tokenizer
        captions_data: List of caption metadata from JSON
        layer_idx: Which layer to extract from
        batch_size: Batch size for processing
        device: Device to use

    Returns:
        embeddings: (num_captions, hidden_dim) numpy array
    """
    model.eval()
    all_embeddings = []

    logger.info(f"\nExtracting text embeddings from layer {layer_idx}...")
    logger.info(f"   Total captions: {len(captions_data)}")
    logger.info(f"   Batch size: {batch_size}")

    start_time = time.time()

    for i in tqdm(range(0, len(captions_data), batch_size), desc="Processing batches"):
        batch_captions = captions_data[i : i + batch_size]
        texts = [cap["text"] for cap in batch_captions]

        # Tokenize
        encoded = tokenizer(
            texts, padding=True, truncation=True, max_length=77, return_tensors="pt"
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # Forward pass through LLM text encoder
        with torch.no_grad():
            # Set routing mask for text-only inputs (all 1s = text expert)
            routing_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)
            for layer in model.llm.model.layers:
                layer.mlp.routing_mask = routing_mask

            outputs = model.llm.model(
                input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
            )

            # Get last layer hidden states
            hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)

            # Mean pool
            pooled = mean_pool_embeddings(hidden_states, attention_mask)

            all_embeddings.append(pooled.cpu().numpy())

    # Concatenate all batches
    embeddings = np.vstack(all_embeddings)

    elapsed = time.time() - start_time
    logger.info(f"   Extracted {len(embeddings)} text embeddings")
    logger.info(f"   Shape: {embeddings.shape}")
    logger.info(f"   Time: {format_time(elapsed)}")

    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings for retrieval")
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--stage_name", type=str, required=True, choices=["stage2", "stage3"], help="Stage name"
    )
    parser.add_argument(
        "--retrieval_json",
        type=str,
        default="results/karpathy_evaluation/karpathy_test_retrieval.json",
        help="Path to retrieval JSON",
    )
    parser.add_argument(
        "--image_base_dir",
        type=str,
        required=True,
        help="Base directory for COCO images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/karpathy_evaluation/retrieval",
        help="Output directory for embeddings",
    )
    parser.add_argument("--layer", type=int, default=31, help="Layer index to extract from")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--device", type=str, default=get_device(), help="Device (default: cuda if available)"
    )

    args = parser.parse_args()

    log_banner(f"EXTRACTING EMBEDDINGS: {args.stage_name.upper()}")

    # Load model
    model, processor, tokenizer = load_model_checkpoint(args.checkpoint_path, args.device)

    # Load retrieval data
    logger.info(f"\nLoading retrieval data from: {args.retrieval_json}")
    retrieval_data = load_json(args.retrieval_json)
    images_data = retrieval_data["images"]
    captions_data = retrieval_data["captions"]
    logger.info(f"   Images: {len(images_data)}")
    logger.info(f"   Captions: {len(captions_data)}")

    # Extract image embeddings
    image_embeddings = extract_image_embeddings(
        model=model,
        processor=processor,
        images_data=images_data,
        image_base_dir=args.image_base_dir,
        layer_idx=args.layer,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Extract text embeddings
    text_embeddings = extract_text_embeddings(
        model=model,
        tokenizer=tokenizer,
        captions_data=captions_data,
        layer_idx=args.layer,
        batch_size=args.batch_size * 2,  # Text is faster, use larger batch
        device=args.device,
    )

    # Save embeddings
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_emb_path = output_dir / f"{args.stage_name}_image_embeddings.npy"
    text_emb_path = output_dir / f"{args.stage_name}_text_embeddings.npy"

    np.save(image_emb_path, image_embeddings)
    np.save(text_emb_path, text_embeddings)

    logger.info("\nSaved embeddings:")
    logger.info(f"   Images: {image_emb_path}")
    logger.info(f"   Text: {text_emb_path}")

    log_banner(f"{args.stage_name.upper()} EMBEDDING EXTRACTION COMPLETE")


if __name__ == "__main__":
    setup_logging()
    main()
