# Code Retrieval with Sentence Transformers

A project for semantic code retrieval using sentence transformers to match natural language queries with relevant code snippets.

## Overview

This project fine-tunes a sentence transformer model to improve the retrieval of code snippets based on natural language descriptions. It uses the MBPP (Mostly Basic Programming Problems) dataset augmented with transformed text-code pairs.

## Features

- Fine-tunes sentence transformer models for code retrieval
- Compares performance between baseline and fine-tuned models
- Provides evaluation metrics for retrieval accuracy
- Includes data augmentation capabilities

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/code-retrieval-transformer.git
cd code-retrieval-transformer
pip install -r requirements.txt
```

## Dataset

The project uses an augmented version of the MBPP dataset, with:

- Original text-code pairs
- Transformed/augmented text-code pairs

## Usage

### Training and Evaluation

Run the main script:

```bash
python -m code.main
```

This will:

1. Load and split the dataset
2. Create a baseline model
3. Train a fine-tuned model
4. Evaluate both models
5. Generate comparison visualizations

### Customization

You can modify hyperparameters in `code/main.py`:

- Model name
- Number of epochs
- Batch size
- Test/train split ratio

## Project Structure

```
├── code/                    # Main code
│   ├── main.py              # Entry point
│   ├── dataset.py           # Dataset loading
│   ├── train.py             # Training functions
│   └── eval.py              # Evaluation metrics
├── data/                    # Data files
│   └── mbpp_text_code_augmented.jsonl
├── output/                  # Results and saved models
└── requirements.txt         # Python dependencies
```

## License

MIT License - see the [LICENSE](LICENSE) file for details.
