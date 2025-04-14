from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer

class CodeTextPair(Dataset):
    """Dataset for Python code and text description pairs."""
    def __init__(self, data, code_col="Code", desc_col="Text", tokeniser=None, tokeniser_args=None):
        """
        Args:
            data_path (str): Path to a JSON or CSV file containing code-text pairs.
            tokenizer_name (str): Name of the BGE tokenizer.
        """

        # Load data (JSONL format)
        self.data = data
        self.code_col = code_col
        self.desc_col = desc_col
        self.tokeniser_args = tokeniser_args or {
            "padding": "max_length",
            "truncation": True,
            "return_tensors": "pt"
        }

        # Load tokeniser
        self.tokeniser = tokeniser

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns:
            code_tokens (tensor): Tokenized Python code.
            text_tokens (tensor): Tokenized natural language description.
        """
        code = str(self.data[index][self.code_col])
        description = str(self.data[index][self.desc_col])

        # Tokenize both code and description
        code_tokens = self.tokeniser(code, **self.tokeniser_args)
        text_tokens = self.tokeniser(description, **self.tokeniser_args)

        # Extract tensors from dict
        code_tensors = {key: val.squeeze(0) for key, val in code_tokens.items()}
        text_tensors = {key: val.squeeze(0) for key, val in text_tokens.items()}

        return code_tensors, text_tensors, index

