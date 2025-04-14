import json
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, models, losses
from torch.utils.data import DataLoader, Dataset, random_split
from code.dataset import CodeTextDataset
from code.eval import evaluate_retrieval

# to run: python -m code.main

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)


def load_test_data(data_path, test_ratio=0.1, random_seed=42):
    """Load data and split into train/test sets"""
    with open(data_path) as f:
        data = [json.loads(line) for line in f]

    # Split into train and test sets
    test_size = int(len(data) * test_ratio)
    train_size = len(data) - test_size

    # Use random_split for the split
    torch.manual_seed(random_seed)
    train_data, test_data = random_split(data, [train_size, test_size])

    return list(train_data), list(test_data)


def train_sentence_transformer(train_dataset, model_name, epochs=3, batch_size=32, output_dir="output"):
    """Train a sentence transformer model"""
    print(f"\nTraining sentence transformer model: {model_name}")

    # Define model architecture
    word_embedding_model = models.Transformer(model_name, max_seq_length=512)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Create data loader
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size)

    # Set up loss function
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=100,
        output_path=f"{output_dir}/{model_name.replace('/', '-')}-finetuned"
    )

    return model


def create_baseline_model(model_name, output_dir="output"):
    """Create a baseline model (using the same architecture but without fine-tuning)"""
    print(f"\nPreparing baseline model: {model_name} (no fine-tuning)")

    # Use the same model architecture as the fine-tuned version
    word_embedding_model = models.Transformer(model_name, max_seq_length=512)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Save the model without training
    model.save(f"{output_dir}/{model_name.replace('/', '-')}-baseline")

    return model


def main():
    print("=== Code Retrieval with Sentence Transformers ===")

    # Use the same model architecture for both baseline and fine-tuned
    model_name = "BAAI/bge-base-en"

    # Path to the MBPP dataset
    data_path = "data/mbpp_text_code_augmented.jsonl"

    # Split data into train and test sets
    train_data, test_data = load_test_data(data_path)
    print(
        f"Loaded {len(train_data)} training examples and {len(test_data)} test examples")

    # Create dataset for training
    train_dataset = CodeTextDataset(data_path, use_transformed=True)

    # Create baseline model (same architecture, no fine-tuning)
    baseline_model = create_baseline_model(model_name)

    # Train our model (same architecture with fine-tuning)
    our_model = train_sentence_transformer(
        train_dataset,
        model_name=model_name,
        epochs=3
    )

    # Evaluate both models
    print("\n=== Evaluation Results ===")

    baseline_results = evaluate_retrieval(baseline_model, test_data)
    our_results = evaluate_retrieval(our_model, test_data)

    # Print results
    print(f"\nBaseline Model ({model_name} without fine-tuning):")
    for k, acc in baseline_results.items():
        print(f"  {k}: {acc:.4f}")

    print(f"\nOur Fine-tuned Model ({model_name} with fine-tuning):")
    for k, acc in our_results.items():
        print(f"  {k}: {acc:.4f}")

    # Create a comparison chart
    plot_comparison(baseline_results, our_results, model_name)


def plot_comparison(baseline_results, our_results, model_name):
    """Create a bar chart comparing the two models"""
    metrics = list(baseline_results.keys())
    baseline_scores = [baseline_results[m] for m in metrics]
    our_scores = [our_results[m] for m in metrics]

    x = range(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([i - width/2 for i in x], baseline_scores,
           width, label=f'Baseline ({model_name} without fine-tuning)')
    ax.bar([i + width/2 for i in x], our_scores,
           width, label=f'Fine-tuned ({model_name} with fine-tuning)')

    ax.set_ylabel('Accuracy')
    ax.set_title('Model Comparison: Effect of Fine-tuning')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    plt.tight_layout()
    plt.savefig('output/model_comparison.png')
    print("\nComparison chart saved to output/model_comparison.png")


if __name__ == "__main__":
    main()
