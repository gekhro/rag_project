from datasets import Dataset
from data.dataloader import CodeTextPair
from torch.utils.data import DataLoader
import json


def load_jsonl(jsonl_path, tokeniser, tokeniser_args=None, dataloader_args=None, seed=42):
    with open(jsonl_path) as f:
        data = [json.loads(line) for line in f]
    
    dataset = Dataset.from_list(data)
    dataloader_args = dataloader_args or {
        "batch_size": 32,
        "shuffle": True,
        "num_workers": 2,
        "pin_memory": True,
        "drop_last": True
    }

    # Split
    split = dataset.train_test_split(test_size=0.2, seed=seed)
    val_test = split["test"].train_test_split(test_size=0.5, seed=seed)

    # Get raw list of dicts
    train_data = split["train"].to_list()
    val_data = val_test["train"].to_list()
    test_data = val_test["test"].to_list()

    # Wrap in CodeTextPair
    train_dataset = CodeTextPair(data=train_data, code_col="code", desc_col="text", tokeniser=tokeniser, tokeniser_args=tokeniser_args)
    val_dataset = CodeTextPair(data=val_data, code_col="code", desc_col="text", tokeniser=tokeniser, tokeniser_args=tokeniser_args)
    test_dataset = CodeTextPair(data=test_data, code_col="code", desc_col="text", tokeniser=tokeniser, tokeniser_args=tokeniser_args)

    # Turn into dataloader
    train_dataloader = DataLoader(train_dataset, **dataloader_args)
    val_dataloader = DataLoader(val_dataset, **dataloader_args)
    test_dataloader = DataLoader(test_dataset, **dataloader_args)

    return train_dataloader, val_dataloader, test_dataloader