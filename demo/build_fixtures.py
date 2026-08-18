"""Build the tiny, fully synthetic fixtures the CPU demo trains against.

Everything here is generated locally — no network, no HuggingFace downloads, no
COCO. The point is to produce a *structurally real* miniature of the paper
setup so the unmodified training scripts can run against it:

- a tiny Mistral base model (2 layers, hidden 64) saved to disk
- a tiny CLIP vision tower + image processor
- a word-level tokenizer trained on the synthetic captions
- a valid COCO-format captions file plus matching JPEGs
- a LLaVA-Instruct-format conversation file over the same images
- a demo_config.yaml wired to all of the above

The models are randomly initialised: the demo demonstrates that the *pipeline*
runs and that routing behaves, not that a 2-layer model can caption images.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import (
    CLIPImageProcessor,
    CLIPVisionConfig,
    CLIPVisionModel,
    MistralConfig,
    MistralForCausalLM,
    PreTrainedTokenizerFast,
)

logger = logging.getLogger(__name__)

# Miniature dimensions. Small enough for a laptop, large enough that every
# shape-dependent code path (multi-head attention, patch embedding, the MoE
# expert split) is genuinely exercised.
IMAGE_SIZE = 32
PATCH_SIZE = 16
CLIP_HIDDEN = 32
LLM_HIDDEN = 64
NUM_LAYERS = 2
NUM_HEADS = 4
NUM_KV_HEADS = 2
MAX_POSITION = 256

# The synthetic world: coloured shapes on a plain background, described by a
# caption built from the same vocabulary. Deliberately learnable in principle,
# so a falling loss curve means something.
COLOURS = {
    "red": (220, 50, 50),
    "green": (50, 180, 80),
    "blue": (60, 90, 220),
    "yellow": (230, 200, 60),
}
SHAPES = ("circle", "square", "triangle")


def build_tokenizer(captions: list[str], output_dir: Path) -> PreTrainedTokenizerFast:
    """Train a word-level tokenizer on the synthetic captions and save it."""
    tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(
        vocab_size=256,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
    )
    tokenizer.train_from_iterator(captions, trainer)

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )
    fast.save_pretrained(str(output_dir))
    logger.info("Tokenizer: %d tokens → %s", fast.vocab_size, output_dir)
    return fast


def build_vision_tower(output_dir: Path) -> None:
    """Save a tiny randomly-initialised CLIP vision tower and its processor."""
    config = CLIPVisionConfig(
        hidden_size=CLIP_HIDDEN,
        intermediate_size=CLIP_HIDDEN * 2,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
    )
    CLIPVisionModel(config).save_pretrained(str(output_dir))

    processor = CLIPImageProcessor(
        size={"shortest_edge": IMAGE_SIZE},
        crop_size={"height": IMAGE_SIZE, "width": IMAGE_SIZE},
    )
    processor.save_pretrained(str(output_dir))

    num_patches = (IMAGE_SIZE // PATCH_SIZE) ** 2
    logger.info("Vision tower: %d patches + CLS → %s", num_patches, output_dir)


def build_base_llm(vocab_size: int, output_dir: Path) -> None:
    """Save a tiny randomly-initialised Mistral — the Stage 0 input."""
    config = MistralConfig(
        vocab_size=vocab_size,
        hidden_size=LLM_HIDDEN,
        intermediate_size=LLM_HIDDEN * 2,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        max_position_embeddings=MAX_POSITION,
    )
    MistralForCausalLM(config).save_pretrained(str(output_dir))
    logger.info("Base LLM: %d layers, hidden %d → %s", NUM_LAYERS, LLM_HIDDEN, output_dir)


def _draw(shape: str, colour: tuple[int, int, int], rng: np.random.Generator) -> Image.Image:
    """Render one shape on a noisy background, so images are not trivially identical."""
    background = rng.integers(200, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    image = Image.fromarray(background)
    draw = ImageDraw.Draw(image)
    box = (6, 6, IMAGE_SIZE - 6, IMAGE_SIZE - 6)

    if shape == "circle":
        draw.ellipse(box, fill=colour)
    elif shape == "square":
        draw.rectangle(box, fill=colour)
    else:
        draw.polygon(
            [(IMAGE_SIZE // 2, 6), (6, IMAGE_SIZE - 6), (IMAGE_SIZE - 6, IMAGE_SIZE - 6)],
            fill=colour,
        )
    return image


def build_dataset(num_images: int, output_root: Path, seed: int) -> list[str]:
    """Write synthetic images plus COCO-caption and LLaVA-instruct annotations.

    Returns the caption strings so the tokenizer can be trained on them.
    """
    rng = np.random.default_rng(seed)
    image_dir = output_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    coco_images, coco_annotations, llava_records, captions = [], [], [], []

    for image_id in range(num_images):
        colour_name = str(rng.choice(list(COLOURS)))
        shape = str(rng.choice(SHAPES))
        caption = f"a {colour_name} {shape} on a plain background"
        captions.append(caption)

        # COCO_Loader derives the filename from the image id as %012d.jpg.
        filename = f"{image_id:012d}.jpg"
        _draw(shape, COLOURS[colour_name], rng).save(image_dir / filename, quality=95)

        coco_images.append(
            {"id": image_id, "file_name": filename, "width": IMAGE_SIZE, "height": IMAGE_SIZE}
        )
        coco_annotations.append({"id": image_id, "image_id": image_id, "caption": caption})
        llava_records.append(
            {
                "id": str(image_id),
                "image": filename,
                "conversations": [
                    {"from": "human", "value": "<image>\nwhat shape is in this image ?"},
                    {"from": "gpt", "value": f"the image shows a {colour_name} {shape}"},
                ],
            }
        )
        captions.append(f"the image shows a {colour_name} {shape}")
        captions.append("what shape is in this image ?")

    coco = {
        "info": {"description": "Synthetic COCO-format fixtures for the CPU demo"},
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
    }
    (output_root / "coco_captions.json").write_text(json.dumps(coco))
    (output_root / "llava_instruct.json").write_text(json.dumps(llava_records))

    logger.info("Dataset: %d images + COCO/LLaVA annotations → %s", num_images, output_root)
    return captions


def write_config(root: Path, fixtures: Path, output_dir: Path) -> Path:
    """Write the demo training config the real scripts will read via MOE_CONFIG."""
    config = {
        "paths": {
            "mistral_local_path": str(fixtures / "base_llm"),
            "clip_local_path": str(fixtures / "clip"),
            "image_dir": str(fixtures / "data" / "images"),
            "annotations_file": str(fixtures / "data" / "coco_captions.json"),
            "llava_annotations_file": str(fixtures / "data" / "llava_instruct.json"),
            "llava_image_dir": str(fixtures / "data" / "images"),
            "moe_model_path": str(fixtures / "moe_model"),
            "output_dir": str(output_dir),
        },
        # A few short epochs, tiny batches: enough for the loss curves to have
        # a shape worth looking at, while keeping the whole run well under a
        # minute. The demo proves the pipeline runs and that routing metrics
        # are produced, not that the model learns to caption.
        "training_stage1": {
            "num_epochs": 3,
            "batch_size": 4,
            "learning_rate": 1e-3,
            "subset_fraction": 1.0,
            "weight_decay": 0.01,
            "gradient_accumulation_steps": 1,
        },
        "training_stage2": {
            "num_epochs": 3,
            "batch_size": 4,
            "learning_rate": 1e-3,
            "weight_decay": 0.01,
            "subset_fraction": 1.0,
            "gradient_accumulation_steps": 1,
        },
        "training_stage2.5": {
            "num_epochs": 3,
            "batch_size": 4,
            "learning_rate": 1e-3,
            "weight_decay": 0.01,
            "subset_fraction": 1.0,
            "gradient_accumulation_steps": 1,
            "load_balancing_coeff": 0.01,
        },
        "training_stage3": {
            "num_epochs": 3,
            "batch_size": 2,
            "learning_rate": 1e-3,
            "weight_decay": 0.01,
            "subset_fraction": 1.0,
            "val_subset_fraction": 1.0,
            "gradient_accumulation_steps": 1,
            "attention_dropout": 0.1,
            "expert_dropout": 0.1,
            "label_smoothing": 0.1,
            "dataset": "llava",
        },
        "dense_control": {
            "num_epochs": 3,
            "batch_size": 4,
            "learning_rate": 1e-3,
            "weight_decay": 0.01,
            "subset_fraction": 1.0,
            "gradient_accumulation_steps": 1,
        },
        # num_workers=0 keeps the demo single-process: faster for tiny datasets
        # and avoids worker start-up dominating the runtime.
        "dataloader": {"num_workers": 0, "num_workers_s1": 0, "data_seed": 42},
    }
    config_path = root / "demo_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    logger.info("Config → %s", config_path)
    return config_path


def build(output_root: Path, num_images: int = 24, seed: int = 42) -> Path:
    """Build every fixture and return the path to the demo config.

    Seeded so repeated builds produce byte-identical models: two demo runs are
    then directly comparable, and a change in the reported numbers means the
    code changed rather than the fixtures.
    """
    import torch

    torch.manual_seed(seed)

    fixtures = output_root / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)

    captions = build_dataset(num_images, fixtures / "data", seed)
    tokenizer = build_tokenizer(captions, fixtures / "tokenizer")
    build_vision_tower(fixtures / "clip")
    # The base LLM and the tokenizer live in the same directory because the
    # training scripts load both from paths.mistral_local_path.
    build_base_llm(tokenizer.vocab_size, fixtures / "base_llm")
    tokenizer.save_pretrained(str(fixtures / "base_llm"))

    return write_config(output_root, fixtures, output_root / "runs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="demo_output", help="Directory to write fixtures into")
    parser.add_argument("--num-images", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config_path = build(Path(args.output), args.num_images, args.seed)
    logger.info("Fixtures ready. Config: %s", config_path)


if __name__ == "__main__":
    main()
