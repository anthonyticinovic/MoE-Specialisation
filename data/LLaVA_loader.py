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
        max_length: int = 512,
        verify_images: bool = True,
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
            max_length: Sequence length every sample is truncated or padded to
            verify_images: Drop samples whose image file is missing (one stat each)
        """
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        self.annotations_file = annotations_file
        self.image_dir = image_dir
        self.clip_processor = clip_processor
        self.tokenizer = tokenizer
        self.split = split
        self.subset_fraction = subset_fraction
        self.val_fraction = val_fraction
        self.seed = seed
        self.debug = debug
        self.max_length = max_length
        self.verify_images = verify_images
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

        # Filter out anything __getitem__ could not turn into a full sample.
        # It has no way to skip an index — the default collate stacks whatever
        # it returns — so every unusable sample must be dropped here. The two
        # conditions are exactly the invariants __getitem__ relies on:
        # at least one complete Q&A pair, and an image file that exists.
        valid_data = []
        missing_images = 0
        for item in split_data:
            if "conversations" not in item or "image" not in item:
                continue
            if len(item["conversations"]) < 2:
                continue
            if self.verify_images and not os.path.exists(
                os.path.join(self.image_dir, item["image"])
            ):
                missing_images += 1
                continue
            valid_data.append(item)

        if missing_images:
            logger.warning(
                "Dropped %d %s samples with no image file in %s.",
                missing_images,
                self.split,
                self.image_dir,
            )

        return valid_data

    def __len__(self):
        return len(self.data)

    def _load_pixel_values(self, sample: dict[str, Any]) -> torch.Tensor:
        """Read the sample's image and run it through the CLIP processor.

        Missing files are filtered out at construction. A blank-image fallback
        here would train the model on black pixels paired with a real caption
        and report nothing but a warning, so fail instead.
        """
        image_path = os.path.join(self.image_dir, sample["image"])
        try:
            image = Image.open(image_path).convert("RGB")
        except OSError as err:
            raise OSError(
                f"Could not read {image_path}, which existed when the dataset "
                "was built. Do not modify the image directory during a run."
            ) from err

        return self.clip_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

    def _tokenise_conversation(
        self, conversations: list[dict[str, str]], idx: int
    ) -> tuple[list[int], list[bool]]:
        """Flatten every Q&A turn into one token sequence plus an answer mask.

        Returns the token ids and, parallel to them, ``True`` wherever the token
        should contribute to the loss — the answers and the final EOS, never the
        questions. Both are plain lists; turning them into tensors is the next
        step's job, kept separate so neither name is ever rebound from list to
        tensor (which is what made this file untypeable).
        """
        if len(conversations) % 2 != 0:
            # Odd turn count: truncate to the last complete Q&A pair.
            conversations = conversations[: len(conversations) - 1]

        # _load_data admits only samples with >= 2 turns, and dropping one odd
        # turn from >= 2 still leaves >= 2, so this holds by construction.
        assert len(conversations) >= 2, f"sample {idx} reached __getitem__ with no Q&A pair"

        token_ids: list[int] = []
        answer_mask: list[bool] = []

        for turn in range(0, len(conversations), 2):
            question = conversations[turn]["value"]
            answer = conversations[turn + 1]["value"]

            # The <image> placeholder appears in the first question only.
            question = question.replace("<image>", "").replace("\n", " ").strip()

            question_ids = self._encode(question, add_special_tokens=turn == 0)
            answer_ids = self._encode(answer.strip(), add_special_tokens=False)

            token_ids.extend(question_ids)
            answer_mask.extend([False] * len(question_ids))
            token_ids.extend(answer_ids)
            answer_mask.extend([True] * len(answer_ids))

        token_ids.append(self.tokenizer.eos_token_id)
        answer_mask.append(True)  # EOS contributes to loss

        return token_ids, answer_mask

    def _encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        """Tokenise one turn. Only the first question carries the BOS token."""
        encoding = self.tokenizer(
            text,
            truncation=False,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
        )
        return encoding["input_ids"].squeeze(0).tolist()

    def _to_padded_tensors(
        self, token_ids: list[int], answer_mask: list[bool], idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Truncate or pad to ``max_length`` and build the three model inputs.

        Returns ``(input_ids, attention_mask, labels, answer_mask)``. The trailing
        answer mask is only needed by the debug dump, and covers the real tokens
        rather than the padding.
        """
        ids = torch.tensor(token_ids[: self.max_length], dtype=torch.long)
        mask = torch.tensor(answer_mask[: self.max_length], dtype=torch.bool)

        if len(token_ids) > self.max_length and self.debug:
            answers_left = int(mask.sum().item())
            if answers_left < 5:
                logger.debug("Sample %d truncated to only %d answer tokens.", idx, answers_left)

        # Questions are masked with -100 so only answers contribute to the loss.
        labels = torch.where(mask, ids, torch.tensor(-100))

        real_length = len(ids)
        padding = self.max_length - real_length
        if padding > 0:
            pad_id = self.tokenizer.pad_token_id
            ids = torch.cat([ids, torch.full((padding,), pad_id, dtype=torch.long)])
            labels = torch.cat([labels, torch.full((padding,), -100, dtype=torch.long)])
            attention_mask = torch.cat(
                [
                    torch.ones(real_length, dtype=torch.long),
                    torch.zeros(padding, dtype=torch.long),
                ]
            )
        else:
            attention_mask = torch.ones(real_length, dtype=torch.long)

        return ids.long(), attention_mask.long(), labels.long(), mask

    def __getitem__(self, idx):
        """
        Returns:
            pixel_values: Processed image tensor
            input_ids: Tokenized text (all QA pairs concatenated)
            attention_mask: Attention mask for text
            labels: Token IDs with questions masked as -100, answers unmasked
        """
        sample = self.data[idx]
        pixel_values = self._load_pixel_values(sample)
        token_ids, answer_mask = self._tokenise_conversation(sample["conversations"], idx)
        input_ids, attention_mask, labels, real_answer_mask = self._to_padded_tensors(
            token_ids, answer_mask, idx
        )

        if self.debug and self.debug_counter < 3:
            self._debug_print_sample(idx, input_ids, labels, attention_mask, real_answer_mask)
            self.debug_counter += 1

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
