import logging
import os
import random

from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class COCO_Loader(Dataset):
    def __init__(
        self,
        image_dir,
        annotations_file,
        clip_processor,
        tokenizer,
        subset_fraction=1.0,
        split="train",
        val_split_fraction=0.1,
        val_subset_fraction=1.0,  # Additional subsampling for validation set
        seed=42,  # Fixed seed for reproducible splits across all stages
        max_length=128,  # Caption token budget; must match the training config
        verify_images=True,  # Drop image IDs with no file on disk (one stat each)
    ):
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        self.image_dir = image_dir
        self.coco = COCO(annotations_file)
        self.clip_processor = clip_processor
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Subset based on unique image IDs with a fixed seed, so the train/val
        # boundary is identical across every stage and every run.
        all_img_ids = list(sorted(self.coco.imgs.keys()))

        # Use a separate Random instance with fixed seed for reproducibility
        rng = random.Random(seed)
        rng.shuffle(all_img_ids)

        # 1. Take a fraction of the unique image IDs
        subset_size = int(len(all_img_ids) * subset_fraction)
        subset_img_ids = all_img_ids[:subset_size]

        # 2. Split the subset of image IDs into train/val
        split_index = int(len(subset_img_ids) * (1 - val_split_fraction))
        if split == "train":
            final_img_ids = subset_img_ids[:split_index]
            logger.info("Using %d unique images for training.", len(final_img_ids))
        elif split == "val":
            val_img_ids = subset_img_ids[split_index:]
            # Apply additional subsampling to validation set if requested
            if val_subset_fraction < 1.0:
                val_subset_size = int(len(val_img_ids) * val_subset_fraction)
                final_img_ids = val_img_ids[:val_subset_size]
                logger.info(
                    "Using %d unique images for validation (subsampled from %d).",
                    len(final_img_ids),
                    len(val_img_ids),
                )
            else:
                final_img_ids = val_img_ids
                logger.info("Using %d unique images for validation.", len(final_img_ids))

        # 3. Drop image IDs whose file is missing, *after* the split so that a
        # gap in the image directory cannot move the train/val boundary.
        # __getitem__ has no way to skip a sample — the default collate stacks
        # whatever it returns — so an unusable sample must never enter the
        # dataset in the first place.
        if verify_images:
            present = [i for i in final_img_ids if os.path.exists(self._image_path(i))]
            if len(present) < len(final_img_ids):
                logger.warning(
                    "Dropped %d of %d %s images with no file in %s.",
                    len(final_img_ids) - len(present),
                    len(final_img_ids),
                    split,
                    image_dir,
                )
            final_img_ids = present

        # 4. Load annotations ONLY for the final set of image IDs
        ann_ids = self.coco.getAnnIds(imgIds=final_img_ids)
        self.annotations = self.coco.loadAnns(ann_ids)

    def _image_path(self, image_id):
        """COCO stores images as a zero-padded 12-digit id."""
        return os.path.join(self.image_dir, f"{image_id:012d}.jpg")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_path = self._image_path(annotation["image_id"])

        # Missing files are filtered out at construction, so reaching this is
        # a real fault (the directory changed under the run). Raise rather than
        # return a placeholder: the default collate would stack it silently.
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError as err:
            raise FileNotFoundError(
                f"{image_path} disappeared after the dataset was built. "
                "Construct COCO_Loader with verify_images=True (the default) "
                "and do not modify the image directory during a run."
            ) from err

        caption = annotation["caption"]

        image_processed = self.clip_processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        tokenized_caption = self.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = tokenized_caption["input_ids"].squeeze(0)
        attention_mask = tokenized_caption["attention_mask"].squeeze(0)

        return image_processed, input_ids, attention_mask
