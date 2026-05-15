"""
Dataset utilities and collate functions.
"""

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def get_ag_news_data():
    """Load AG News dataset."""
    dataset = load_dataset("ag_news")
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    class_names = ["World", "Sports", "Business", "Sci/Tech"]
    return train_dataset, test_dataset, class_names


def get_20_newsgroups_data():
    """Load 20 Newsgroups dataset."""
    dataset = load_dataset("SetFit/20_newsgroups", split="test")
    class_names = [
        "alt.atheism",
        "comp.graphics",
        "comp.os.ms-windows.misc",
        "comp.sys.ibm.pc.hardware",
        "comp.sys.mac.hardware",
        "comp.windows.x",
        "misc.forsale",
        "rec.autos",
        "rec.motorcycles",
        "rec.sport.baseball",
        "rec.sport.hockey",
        "sci.crypt",
        "sci.electronics",
        "sci.med",
        "sci.space",
        "soc.religion.christian",
        "talk.politics.guns",
        "talk.politics.mideast",
        "talk.politics.misc",
        "talk.religion.misc",
    ]
    return dataset, class_names


def collate_fn(batch, tokenizer=None, max_length=256):
    """Collate function for processing batches of data for the DataLoader."""
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    texts = [item["text"] for item in batch]
    labels = [item["label"] for item in batch]

    # The tokenizer handles padding, truncation, and tensor conversion
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",  # Pad to a fixed length
        truncation=True,
        max_length=max_length,
    )

    inputs["labels"] = torch.tensor(labels)
    return inputs


def create_data_loaders(dataset_name="ag_news", batch_size=32, tokenizer=None):
    """Create data loaders for a given dataset."""
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    if dataset_name == "ag_news":
        train_dataset, test_dataset, class_names = get_ag_news_data()

        # Create DataLoaders for training and testing
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=lambda batch: collate_fn(batch, tokenizer),
            shuffle=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=lambda batch: collate_fn(batch, tokenizer),
        )

        return train_loader, test_loader, class_names

    elif dataset_name == "20_newsgroups":
        test_dataset, class_names = get_20_newsgroups_data()

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=lambda batch: collate_fn(batch, tokenizer, max_length=512),
        )

        return None, test_loader, class_names

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
