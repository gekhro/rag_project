from torch.utils.data import Dataset
from sentence_transformers import InputExample
import json


class CodeTextDataset(Dataset):
    def __init__(self, data_path, use_transformed=True):
        with open(data_path) as f:
            self.data = [json.loads(line) for line in f]
        self.use_transformed = use_transformed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Choose between original or transformed versions
        if self.use_transformed:
            text = item["transformed_text"]
            code = item["transformed_code"]
        else:
            text = item["original_text"]
            code = item["original_code"]

        # Create a positive pair example
        return InputExample(texts=[text, code])
