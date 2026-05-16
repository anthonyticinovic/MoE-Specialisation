"""Tests for data loader label-masking and split logic.

All tests are CPU-only with synthetic in-memory data — no real COCO/LLaVA files,
no network, no GPU.
"""

import json
from typing import Any
from unittest.mock import patch

import pytest
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------


class _ProcessorOutput(dict):
    """Dict subclass that also supports attribute access (mirrors CLIPProcessor output)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as err:
            raise AttributeError(name) from err


class FakeProcessor:
    """Minimal CLIP processor stub returning a fixed-size pixel_values tensor."""

    def __call__(self, images, return_tensors="pt"):
        return _ProcessorOutput({"pixel_values": torch.zeros(1, 3, 224, 224)})


class FakeTokenizer:
    """Minimal tokenizer stub with deterministic token ids."""

    pad_token_id = 0
    eos_token_id = 2

    def __call__(
        self,
        text,
        return_tensors="pt",
        padding=None,
        truncation=None,
        max_length=None,
        add_special_tokens=True,
    ):
        # Encode each character as its ord % 100 + 3 (avoids 0,1,2 special ids)
        ids = [ord(c) % 100 + 3 for c in text[:10]]  # cap at 10 to keep tests tiny
        if add_special_tokens:
            ids = [1] + ids  # BOS = 1
        input_ids = torch.tensor([ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        if padding == "max_length" and max_length is not None:
            pad_len = max_length - input_ids.shape[1]
            if pad_len > 0:
                input_ids = torch.cat([input_ids, torch.zeros(1, pad_len, dtype=torch.long)], dim=1)
                attention_mask = torch.cat(
                    [attention_mask, torch.zeros(1, pad_len, dtype=torch.long)], dim=1
                )
            else:
                input_ids = input_ids[:, :max_length]
                attention_mask = attention_mask[:, :max_length]

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, ids, skip_special_tokens=False):
        return "decoded"


# ---------------------------------------------------------------------------
# LLaVA_Loader tests
# ---------------------------------------------------------------------------


def _make_llava_json(n: int = 20) -> list[dict[str, Any]]:
    """Generate synthetic LLaVA-style conversation data."""
    return [
        {
            "id": f"sample_{i}",
            "image": f"image_{i:06d}.jpg",
            "conversations": [
                {"from": "human", "value": f"<image>\nWhat is in image {i}?"},
                {"from": "gpt", "value": f"There is a cat in image {i}."},
            ],
        }
        for i in range(n)
    ]


@pytest.fixture
def llava_json_path(tmp_path):
    data = _make_llava_json(20)
    path = tmp_path / "llava_test.json"
    path.write_text(json.dumps(data))
    return str(path)


@pytest.fixture
def llava_image_dir(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(20):
        Image.new("RGB", (4, 4), color=(i % 256, 0, 0)).save(img_dir / f"image_{i:06d}.jpg")
    return str(img_dir)


class TestLLaVALabelMasking:
    def _load(self, llava_json_path, llava_image_dir, split="train", **kwargs):
        from data.LLaVA_loader import LLaVA_Loader

        return LLaVA_Loader(
            annotations_file=llava_json_path,
            image_dir=llava_image_dir,
            clip_processor=FakeProcessor(),
            tokenizer=FakeTokenizer(),
            split=split,
            subset_fraction=1.0,
            **kwargs,
        )

    def test_returns_four_tensors(self, llava_json_path, llava_image_dir):
        ds = self._load(llava_json_path, llava_image_dir)
        assert len(ds) > 0
        item = ds[0]
        assert len(item) == 4

    def test_question_tokens_masked_to_minus_100(self, llava_json_path, llava_image_dir):
        ds = self._load(llava_json_path, llava_image_dir)
        _, input_ids, attention_mask, labels = ds[0]

        # BOS and question tokens must all be -100
        real_mask = attention_mask.bool()
        real_labels = labels[real_mask]
        # At minimum the question portion (BOS + tokens) should be -100
        assert (real_labels == -100).any(), "Expected some masked (question) tokens"

    def test_answer_tokens_unmasked(self, llava_json_path, llava_image_dir):
        ds = self._load(llava_json_path, llava_image_dir)
        _, input_ids, _, labels = ds[0]
        assert (labels != -100).any(), "Expected some unmasked (answer) tokens"

    def test_padding_tokens_masked(self, llava_json_path, llava_image_dir):
        ds = self._load(llava_json_path, llava_image_dir)
        _, input_ids, attention_mask, labels = ds[0]
        padding_positions = attention_mask == 0
        if padding_positions.any():
            assert (labels[padding_positions] == -100).all(), (
                "Padding tokens must be -100 in labels"
            )

    def test_label_values_match_input_ids_for_answers(self, llava_json_path, llava_image_dir):
        ds = self._load(llava_json_path, llava_image_dir)
        _, input_ids, attention_mask, labels = ds[0]
        answer_positions = (labels != -100) & attention_mask.bool()
        if answer_positions.any():
            assert torch.equal(labels[answer_positions], input_ids[answer_positions]), (
                "Answer label IDs must match input_ids"
            )

    def test_output_length_is_max_length(self, llava_json_path, llava_image_dir):
        ds = self._load(llava_json_path, llava_image_dir)
        _, input_ids, attention_mask, labels = ds[0]
        assert input_ids.shape[0] == 512
        assert labels.shape[0] == 512
        assert attention_mask.shape[0] == 512

    def test_train_val_split_disjoint(self, llava_json_path, llava_image_dir):
        """Train and val splits must not share any sample IDs."""
        train_ds = self._load(llava_json_path, llava_image_dir, split="train")
        val_ds = self._load(llava_json_path, llava_image_dir, split="val")

        train_ids = {s["id"] for s in train_ds.data}
        val_ids = {s["id"] for s in val_ds.data}
        assert len(train_ids & val_ids) == 0, "Train and val sets must be disjoint"

    def test_split_covers_all_data(self, llava_json_path, llava_image_dir):
        train_ds = self._load(llava_json_path, llava_image_dir, split="train")
        val_ds = self._load(llava_json_path, llava_image_dir, split="val")
        assert len(train_ds) + len(val_ds) == 20

    def test_reproducible_with_same_seed(self, llava_json_path, llava_image_dir):
        ds1 = self._load(llava_json_path, llava_image_dir, seed=7)
        ds2 = self._load(llava_json_path, llava_image_dir, seed=7)
        assert [s["id"] for s in ds1.data] == [s["id"] for s in ds2.data]

    def test_different_seeds_give_different_splits(self, llava_json_path, llava_image_dir):
        ds1 = self._load(llava_json_path, llava_image_dir, seed=1)
        ds2 = self._load(llava_json_path, llava_image_dir, seed=2)
        assert [s["id"] for s in ds1.data] != [s["id"] for s in ds2.data]


# ---------------------------------------------------------------------------
# COCO_Loader tests
# ---------------------------------------------------------------------------


class FakeCOCO:
    """Minimal mock of pycocotools.coco.COCO."""

    def __init__(self, n_images=10):
        self.imgs = {i: {"id": i} for i in range(n_images)}
        self._anns = [
            {"id": i, "image_id": i, "caption": f"A caption for image {i}."}
            for i in range(n_images)
        ]

    def getAnnIds(self, imgIds=None):
        if imgIds is None:
            return [a["id"] for a in self._anns]
        return [a["id"] for a in self._anns if a["image_id"] in imgIds]

    def loadAnns(self, ids):
        return [a for a in self._anns if a["id"] in ids]


class TestCOCOLoader:
    def _make_loader(self, tmp_path, split="train", subset_fraction=1.0, **kwargs):
        from data.COCO_loader import COCO_Loader

        # Create fake images in tmp_path
        for i in range(10):
            Image.new("RGB", (4, 4)).save(tmp_path / f"{i:012d}.jpg")

        fake_coco = FakeCOCO(10)
        with patch("data.COCO_loader.COCO", return_value=fake_coco):
            return COCO_Loader(
                image_dir=str(tmp_path),
                annotations_file="fake_annotations.json",
                clip_processor=FakeProcessor(),
                tokenizer=FakeTokenizer(),
                split=split,
                subset_fraction=subset_fraction,
                **kwargs,
            )

    def test_returns_three_tensors(self, tmp_path):
        ds = self._make_loader(tmp_path)
        assert len(ds) > 0
        item = ds[0]
        assert item is not None
        assert len(item) == 3

    def test_output_shapes(self, tmp_path):
        ds = self._make_loader(tmp_path)
        pixel_values, input_ids, attention_mask = ds[0]
        assert pixel_values.shape == (3, 224, 224)
        assert input_ids.shape[0] == 128
        assert attention_mask.shape[0] == 128

    def test_train_val_partition(self, tmp_path):
        train_ds = self._make_loader(tmp_path, split="train")
        val_ds = self._make_loader(tmp_path, split="val")
        assert len(train_ds) + len(val_ds) == 10

    def test_subset_fraction_reduces_size(self, tmp_path):
        full_ds = self._make_loader(tmp_path, subset_fraction=1.0)
        half_ds = self._make_loader(tmp_path, subset_fraction=0.5)
        assert len(half_ds) < len(full_ds)

    def test_reproducible_with_fixed_seed(self, tmp_path):
        ds1 = self._make_loader(tmp_path, seed=42)
        ds2 = self._make_loader(tmp_path, seed=42)
        assert [a["id"] for a in ds1.annotations] == [a["id"] for a in ds2.annotations]
