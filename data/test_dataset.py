import os
import sys
from code.dataset import CodeQueryDataset


def test_dataset():
    """Test the CodeQueryDataset class."""
    print("Testing CodeQueryDataset...")

    # Path to the dataset
    dataset_path = "combined_data.jsonl"

    # Create dataset instances for each split
    train_dataset = CodeQueryDataset(
        data_path=dataset_path,
        max_text_length=512,
        max_code_length=1024,
        augment=False,
        split="train"
    )

    val_dataset = CodeQueryDataset(
        data_path=dataset_path,
        max_text_length=512,
        max_code_length=1024,
        augment=False,
        split="val"
    )

    test_dataset = CodeQueryDataset(
        data_path=dataset_path,
        max_text_length=512,
        max_code_length=1024,
        augment=False,
        split="test"
    )

    # Print dataset statistics
    print(
        f"Dataset splits: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    # Check a few samples
    print("\nSample from training set:")
    sample = train_dataset[0]
    print(f"Text: {sample['text'][:100]}...")
    # print(f"Code: {sample['code'][:100]}...")

    # Test data augmentation
    print("\nTesting data augmentation:")
    aug_dataset = CodeQueryDataset(
        data_path=dataset_path,
        max_text_length=512,
        max_code_length=1024,
        augment=True,
        split="train"
    )

    aug_sample = aug_dataset[0]
    print(f"Augmented text: {aug_sample['text'][:100]}...")
    # print(f"Augmented code: {aug_sample['code'][:100]}...")

    print("\nDataset test completed successfully!")


if __name__ == "__main__":
    # Change to the data directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test_dataset()
