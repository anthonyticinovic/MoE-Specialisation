import os
import random

from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


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
    ):
        self.image_dir = image_dir
        self.coco = COCO(annotations_file)
        self.clip_processor = clip_processor
        self.tokenizer = tokenizer

        # --- MODIFIED: Subset based on unique image IDs with fixed seed ---
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
            print(f"Using {len(final_img_ids)} unique images for training.")
        elif split == "val":
            val_img_ids = subset_img_ids[split_index:]
            # Apply additional subsampling to validation set if requested
            if val_subset_fraction < 1.0:
                val_subset_size = int(len(val_img_ids) * val_subset_fraction)
                final_img_ids = val_img_ids[:val_subset_size]
                print(
                    f"Using {len(final_img_ids)} unique images for validation (subsampled from {len(val_img_ids)})."
                )
            else:
                final_img_ids = val_img_ids
                print(f"Using {len(final_img_ids)} unique images for validation.")

        # 3. Load annotations ONLY for the final set of image IDs
        ann_ids = self.coco.getAnnIds(imgIds=final_img_ids)
        self.annotations = self.coco.loadAnns(ann_ids)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        # Construct image path safely, handling potential missing leading zeros in image_id
        image_filename = f"{annotation['image_id']:012d}.jpg"
        image_path = os.path.join(self.image_dir, image_filename)

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Image file not found at {image_path}. Skipping.")
            # Return None or a placeholder to be handled in the dataloader's collate_fn if needed
            return None

        caption = annotation["caption"]

        image_processed = self.clip_processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        tokenized_caption = self.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        input_ids = tokenized_caption["input_ids"].squeeze(0)
        attention_mask = tokenized_caption["attention_mask"].squeeze(0)

        return image_processed, input_ids, attention_mask
