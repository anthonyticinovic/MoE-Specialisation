import json
import logging
import os
import random
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class LLaVA_Loader(Dataset):
    """
    DataLoader for LLaVA-Instruct-150K dataset.

    Handles multi-turn visual instruction conversations by using ALL Q&A pairs.
    Questions are masked (label=-100), only answers contribute to loss.
    Supports train/val split and subset sampling.
    """

    def __init__(
        self,
        annotations_file: str,
        image_dir: str,
        clip_processor,
        tokenizer,
        split: str = "train",
        subset_fraction: float = 1.0,
        val_fraction: float = 0.2,
        seed: int = 42,
        debug: bool = False,
    ):
        """
        Args:
            annotations_file: Path to llava_instruct_150k.json
            image_dir: Path to image directory (COCO train2017 via symlink)
            clip_processor: CLIP image processor
            tokenizer: Text tokenizer
            split: 'train' or 'val'
            subset_fraction: Fraction of total data to use (0.0 to 1.0)
            val_fraction: Fraction of data reserved for validation (0.0 to 1.0)
            seed: Random seed for reproducibility
            debug: If True, print first 3 samples with decoded tokens and labels
        """
        self.annotations_file = annotations_file
        self.image_dir = image_dir
        self.clip_processor = clip_processor
        self.tokenizer = tokenizer
        self.split = split
        self.subset_fraction = subset_fraction
        self.val_fraction = val_fraction
        self.seed = seed
        self.debug = debug
        self.debug_counter = 0  # Track how many samples we've debugged

        # Load and process data
        self.data = self._load_data()

        # Calculate pool sizes for informative logging
        with open(annotations_file) as f:
            total_data = len(json.load(f))
        total_train_pool = int(total_data * (1 - val_fraction))
        total_val_pool = total_data - total_train_pool

        if split == "train":
            logger.info(
                "Using %d unique samples for training (LLaVA, ALL QA pairs) — %.0f%% of %d pool.",
                len(self.data),
                subset_fraction * 100,
                total_train_pool,
            )
        else:
            logger.info(
                "Using %d unique samples for validation (LLaVA, ALL QA pairs) — %.0f%% of %d pool.",
                len(self.data),
                subset_fraction * 100,
                total_val_pool,
            )

    def _load_data(self) -> list[dict[str, Any]]:
        """
        Load JSON, create train/val split, apply subset fraction.

        CRITICAL: Uses "Split First, Subsample Second" approach to prevent validation leakage.
        This ensures that when subset_fraction is increased for resumed training:
        1. The train/val split boundary remains fixed (based on full dataset)
        2. New training data is a superset of old training data
        3. New validation data is a superset of old validation data
        4. No validation samples ever appear in training data
        """
        # Load JSON
        with open(self.annotations_file) as f:
            all_data = json.load(f)

        # Set random seed for reproducibility
        random.seed(self.seed)

        # Shuffle FULL dataset with fixed seed (this order is permanent)
        shuffled_data = all_data.copy()
        random.shuffle(shuffled_data)

        # STEP 1: Split FULL dataset into train/val pools (this split is PERMANENT)
        total_samples = len(shuffled_data)
        val_size = int(total_samples * self.val_fraction)
        train_size = total_samples - val_size

        train_pool = shuffled_data[:train_size]
        val_pool = shuffled_data[train_size:]

        # STEP 2: Select which pool to use based on split
        if self.split == "train":
            split_pool = train_pool
        else:  # val
            split_pool = val_pool

        # STEP 3: NOW apply subset fraction to the selected pool (deterministic subsampling)
        if self.subset_fraction < 1.0:
            subset_size = int(len(split_pool) * self.subset_fraction)
            split_data = split_pool[:subset_size]
        else:
            split_data = split_pool

        # Filter out samples without conversations or images
        valid_data = []
        for item in split_data:
            if "conversations" in item and len(item["conversations"]) >= 2 and "image" in item:
                valid_data.append(item)

        return valid_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            pixel_values: Processed image tensor
            input_ids: Tokenized text (all QA pairs concatenated)
            attention_mask: Attention mask for text
            labels: Token IDs with questions masked as -100, answers unmasked
        """
        sample = self.data[idx]

        # Load and process image
        image_filename = sample["image"]
        image_path = os.path.join(self.image_dir, image_filename)

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            # Fallback: create a blank image if file not found
            logger.warning("Could not load image %s: %s", image_path, e)
            image = Image.new("RGB", (224, 224), color="black")

        # Process image with CLIP
        pixel_values = self.clip_processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)

        # Extract ALL Q&A pairs from conversations
        conversations = sample["conversations"]

        # Validate: conversations must have even length (question-answer pairs)
        if len(conversations) % 2 != 0:
            # If odd, truncate to last complete Q&A pair
            conversations = conversations[: len(conversations) - 1]

        # Skip samples with no valid Q&A pairs
        if len(conversations) < 2:
            # Fallback: return empty sequences (will be filtered in collate_fn if needed)
            logger.warning("Sample %d has no valid Q&A pairs — skipping.", idx)
            return (
                pixel_values,
                torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long),
                torch.tensor([1], dtype=torch.long),
                torch.tensor([-100], dtype=torch.long),
            )

        # Build combined sequence with proper masking
        # Start with BOS token (included in first question tokenization)
        combined_ids = []
        label_mask = []  # True = compute loss (answer), False = mask (question)

        for i in range(0, len(conversations), 2):
            question_text = conversations[i]["value"]
            answer_text = conversations[i + 1]["value"]

            # Remove <image> placeholder from question (only appears in first question)
            question_text = question_text.replace("<image>", "").replace("\n", " ").strip()
            answer_text = answer_text.strip()

            # Tokenize question
            if i == 0:
                # First question: include BOS token
                question_encoding = self.tokenizer(
                    question_text,
                    truncation=False,
                    add_special_tokens=True,  # Includes BOS
                    return_tensors="pt",
                )
            else:
                # Subsequent questions: no BOS (already have it)
                question_encoding = self.tokenizer(
                    question_text,
                    truncation=False,
                    add_special_tokens=False,
                    return_tensors="pt",
                )

            question_ids = question_encoding["input_ids"].squeeze(0).tolist()

            # Tokenize answer (never add special tokens for answers)
            answer_encoding = self.tokenizer(
                answer_text,
                truncation=False,
                add_special_tokens=False,
                return_tensors="pt",
            )
            answer_ids = answer_encoding["input_ids"].squeeze(0).tolist()

            # Add to combined sequence
            combined_ids.extend(question_ids)
            label_mask.extend([False] * len(question_ids))  # Questions masked

            combined_ids.extend(answer_ids)
            label_mask.extend([True] * len(answer_ids))  # Answers NOT masked

        # Add EOS token at the end
        combined_ids.append(self.tokenizer.eos_token_id)
        label_mask.append(True)  # EOS contributes to loss

        # Convert to tensors
        combined_ids = torch.tensor(combined_ids, dtype=torch.long)
        label_mask = torch.tensor(label_mask, dtype=torch.bool)

        # CRITICAL: Handle truncation if sequence too long
        max_length = 512
        if len(combined_ids) > max_length:
            # Truncate from the end
            combined_ids = combined_ids[:max_length]
            label_mask = label_mask[:max_length]

            # Check if we have at least some answer tokens left
            num_answer_tokens = label_mask.sum().item()
            if num_answer_tokens < 5:
                # Very few answer tokens remain - warn but continue
                if self.debug:
                    logger.debug(
                        "Sample %d truncated to only %d answer tokens.", idx, num_answer_tokens
                    )

        # Create labels: -100 for questions, actual token IDs for answers
        labels = torch.where(
            label_mask,
            combined_ids,  # Answer tokens: keep IDs
            torch.tensor(-100),  # Question tokens: mask with -100
        )

        # CRITICAL: Handle padding
        current_length = len(combined_ids)
        padding_length = max_length - current_length

        if padding_length > 0:
            # Pad input_ids with pad_token_id
            input_ids = torch.cat(
                [
                    combined_ids,
                    torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=torch.long),
                ]
            )

            # Pad labels with -100 (masked)
            labels = torch.cat([labels, torch.full((padding_length,), -100, dtype=torch.long)])

            # Attention mask: 1 for real tokens, 0 for padding
            attention_mask = torch.cat(
                [
                    torch.ones(current_length, dtype=torch.long),
                    torch.zeros(padding_length, dtype=torch.long),
                ]
            )
        else:
            # No padding needed
            input_ids = combined_ids
            attention_mask = torch.ones(len(input_ids), dtype=torch.long)

        # Debug logging for first 3 samples
        if self.debug and self.debug_counter < 3:
            self._debug_print_sample(
                idx, input_ids, labels, attention_mask, label_mask[:current_length]
            )
            self.debug_counter += 1

        # Final dtype validation
        input_ids = input_ids.long()
        labels = labels.long()
        attention_mask = attention_mask.long()

        return pixel_values, input_ids, attention_mask, labels

    def _debug_print_sample(self, idx, input_ids, labels, attention_mask, label_mask):
        """Log detailed debug information for a sample (only when debug=True)."""
        sep = "=" * 80
        logger.debug("\n%s\nDEBUG SAMPLE %d\n%s", sep, idx, sep)

        decoded_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        logger.debug("Full decoded text:\n%s", decoded_text)

        header = f"{'Idx':<5} {'Token':<30} {'InputID':<8} {'Label':<8} {'Mask':<6} {'AttnMask':<8}"
        rows = []
        for i in range(min(100, len(input_ids))):
            token = self.tokenizer.decode([input_ids[i].item()])
            is_answer = "ANSWER" if i < len(label_mask) and label_mask[i] else "QUEST"
            rows.append(
                f"{i:<5} {repr(token):<30} {input_ids[i].item():<8} "
                f"{labels[i].item():<8} {is_answer:<6} {attention_mask[i].item():<8}"
            )
        logger.debug("Token-by-token (first 100):\n%s\n%s\n%s", header, "-" * 80, "\n".join(rows))

        num_real_tokens = attention_mask.sum().item()
        num_answer_tokens = (labels != -100).sum().item()
        num_question_tokens = num_real_tokens - num_answer_tokens
        logger.debug(
            "Stats — total:%d real:%d pad:%d answer:%d question:%d ratio:%.1f%%",
            len(input_ids),
            num_real_tokens,
            (attention_mask == 0).sum().item(),
            num_answer_tokens,
            num_question_tokens,
            num_answer_tokens / max(num_real_tokens, 1) * 100,
        )

        all_question_masked = all(
            labels[i] == -100 for i in range(len(label_mask)) if not label_mask[i]
        )
        all_answer_unmasked = all(
            labels[i] == input_ids[i] for i in range(len(label_mask)) if label_mask[i]
        )
        all_padding_masked = all(labels[i] == -100 for i in range(num_real_tokens, len(labels)))
        logger.debug(
            "Masking — questions masked: %s | answers unmasked: %s | padding masked: %s",
            all_question_masked,
            all_answer_unmasked,
            all_padding_masked,
        )
