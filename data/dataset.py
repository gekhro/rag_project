import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import random


class CodeQueryDataset(Dataset):
    """
    PyTorch Dataset for code query data.
    This dataset loads text-code pairs from a JSONL file and prepares them for training.
    """

    def __init__(
        self,
        data_path: str,
        max_text_length: int = 512,
        max_code_length: int = 1024,
        transform=None,
        augment: bool = False,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        random_seed: int = 42
    ):
        """
        Initialize the CodeQueryDataset.

        Args:
            data_path: Path to the JSONL file containing text-code pairs
            max_text_length: Maximum length for text inputs
            max_code_length: Maximum length for code inputs
            transform: Optional transform to apply to the data
            augment: Whether to use data augmentation
            split: Which split to use ('train', 'val', or 'test')
            train_ratio: Ratio of data to use for training
            val_ratio: Ratio of data to use for validation
            random_seed: Random seed for reproducibility
        """
        self.data_path = data_path
        self.max_text_length = max_text_length
        self.max_code_length = max_code_length
        self.transform = transform
        self.augment = augment

        # Load data from JSONL file
        self.data = self._load_data()

        # Split data into train, validation, and test sets
        random.seed(random_seed)
        indices = list(range(len(self.data)))
        random.shuffle(indices)

        train_size = int(train_ratio * len(indices))
        val_size = int(val_ratio * len(indices))

        if split == "train":
            self.indices = indices[:train_size]
        elif split == "val":
            self.indices = indices[train_size:train_size + val_size]
        elif split == "test":
            self.indices = indices[train_size + val_size:]
        else:
            raise ValueError(
                f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

        print(f"Loaded {len(self.indices)} examples for {split} split")

    def _load_data(self) -> List[Dict]:
        """
        Load data from JSONL file.

        Returns:
            List of dictionaries containing text-code pairs
        """
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if 'text' in item and 'code' in item:
                        data.append(item)
                except json.JSONDecodeError:
                    continue

        return data

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
        """
        Get an item from the dataset by index.

        Args:
            idx: Index of the item to get

        Returns:
            Dictionary containing text and code
        """
        data_idx = self.indices[idx]
        item = self.data[data_idx]

        text = item['text']
        code = item['code']

        # Truncate text and code if they're too long
        if len(text) > self.max_text_length:
            text = text[:self.max_text_length]

        if len(code) > self.max_code_length:
            code = code[:self.max_code_length]

        # Apply data augmentation if enabled
        if self.augment:
            text, code = self._augment_data(text, code)

        # Apply transformations if provided
        if self.transform:
            sample = self.transform({"text": text, "code": code})
            return sample

        return {"text": text, "code": code}

    def _augment_data(self, text: str, code: str) -> Tuple[str, str]:
        """
        Apply data augmentation techniques to the text and code.

        Args:
            text: The text to augment
            code: The code to augment

        Returns:
            Tuple of augmented text and code
        """
        # Simple augmentation example: randomly choose one augmentation technique
        aug_choice = random.random()

        if aug_choice < 0.25:
            # Add random noise to text by removing some words (20% chance for each word)
            text_words = text.split()
            augmented_text = []
            for word in text_words:
                if random.random() > 0.2:  # 80% chance to keep the word
                    augmented_text.append(word)
            text = ' '.join(augmented_text) if augmented_text else text

        elif aug_choice < 0.5:
            # Reorder parts of the code (function definitions or blocks) if possible
            code_lines = code.split('\n')
            if len(code_lines) > 4:  # Only if there are enough lines
                # Find function definitions or logical blocks
                func_starts = []
                for i, line in enumerate(code_lines):
                    if line.strip().startswith('def '):
                        func_starts.append(i)

                if len(func_starts) > 1:  # If there are multiple functions
                    # Sort functions in a different order
                    funcs = []
                    for i in range(len(func_starts)):
                        start = func_starts[i]
                        end = func_starts[i+1] if i + \
                            1 < len(func_starts) else len(code_lines)
                        funcs.append(code_lines[start:end])

                    random.shuffle(funcs)
                    new_code_lines = []
                    for func in funcs:
                        new_code_lines.extend(func)

                    code = '\n'.join(new_code_lines)

        return text, code


class TextCodeCollator:
    """
    Collator for batching text-code pairs.
    This collator handles the conversion of raw strings to tensors for model training.
    """

    def __init__(
        self,
        tokenizer,
        max_text_length: int = 512,
        max_code_length: int = 1024,
        return_tensors: str = "pt"
    ):
        """
        Initialize the TextCodeCollator.

        Args:
            tokenizer: The tokenizer to use for encoding text and code
            max_text_length: Maximum length for text inputs
            max_code_length: Maximum length for code inputs
            return_tensors: The type of tensors to return ('pt' for PyTorch)
        """
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.max_code_length = max_code_length
        self.return_tensors = return_tensors

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Process a batch of examples.

        Args:
            batch: List of dictionaries containing text and code

        Returns:
            Dictionary of tensors ready for model input
        """
        # Extract text and code from batch
        texts = [item["text"] for item in batch]
        codes = [item["code"] for item in batch]

        # Tokenize text and code
        text_encodings = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors=self.return_tensors
        )

        code_encodings = self.tokenizer(
            codes,
            padding="max_length",
            truncation=True,
            max_length=self.max_code_length,
            return_tensors=self.return_tensors
        )

        return {
            "text_input_ids": text_encodings.input_ids,
            "text_attention_mask": text_encodings.attention_mask,
            "code_input_ids": code_encodings.input_ids,
            "code_attention_mask": code_encodings.attention_mask,
        }


def get_dataloaders(
    data_path: str,
    tokenizer,
    batch_size: int = 16,
    max_text_length: int = 512,
    max_code_length: int = 1024,
    num_workers: int = 4,
    augment_train: bool = False
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for training, validation, and testing.

    Args:
        data_path: Path to the JSONL file containing text-code pairs
        tokenizer: The tokenizer to use for encoding text and code
        batch_size: Batch size for DataLoaders
        max_text_length: Maximum length for text inputs
        max_code_length: Maximum length for code inputs
        num_workers: Number of workers for DataLoader
        augment_train: Whether to use data augmentation for training

    Returns:
        Dictionary of DataLoaders for 'train', 'val', and 'test'
    """
    # Create datasets for each split
    train_dataset = CodeQueryDataset(
        data_path=data_path,
        max_text_length=max_text_length,
        max_code_length=max_code_length,
        augment=augment_train,
        split="train"
    )

    val_dataset = CodeQueryDataset(
        data_path=data_path,
        max_text_length=max_text_length,
        max_code_length=max_code_length,
        augment=False,
        split="val"
    )

    test_dataset = CodeQueryDataset(
        data_path=data_path,
        max_text_length=max_text_length,
        max_code_length=max_code_length,
        augment=False,
        split="test"
    )

    # Create collator for batching
    collator = TextCodeCollator(
        tokenizer=tokenizer,
        max_text_length=max_text_length,
        max_code_length=max_code_length
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }


# Example usage
if __name__ == "__main__":
    # Simple test to ensure the dataset loads correctly
    dataset_path = os.path.join("data", "combined_data.jsonl")

    # Test dataset without a tokenizer
    dataset = CodeQueryDataset(data_path=dataset_path, split="train")
    print(f"Dataset size: {len(dataset)}")

    # Get a sample
    sample = dataset[0]
    print(f"Sample text: {sample['text'][:100]}...")
    print(f"Sample code: {sample['code'][:100]}...")

    # To test with a tokenizer, you would need to add code like:
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    # dataloaders = get_dataloaders(dataset_path, tokenizer)
    # for batch in dataloaders["train"]:
    #     print(batch.keys())
    #     print(batch["text_input_ids"].shape)
    #     break
