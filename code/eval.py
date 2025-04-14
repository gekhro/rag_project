import torch
from sentence_transformers import util


def evaluate_retrieval(model, test_data, top_k=(1, 5, 10)):
    # Extract descriptions and code separately
    descriptions = [item["transformed_text"] for item in test_data]
    codes = [item["transformed_code"] for item in test_data]

    # Encode all descriptions
    description_embeddings = model.encode(descriptions, convert_to_tensor=True)

    # Encode all code snippets
    code_embeddings = model.encode(codes, convert_to_tensor=True)

    # Compute similarity matrix
    cos_sim = util.pytorch_cos_sim(description_embeddings, code_embeddings)

    # Calculate retrieval metrics
    results = {}
    for k in top_k:
        correct = 0
        total = len(descriptions)

        # For each query, find top-k nearest neighbors
        top_k_indices = torch.topk(cos_sim, k=k, dim=1).indices

        # Check if the ground truth index is in top-k
        for i in range(total):
            if i in top_k_indices[i]:
                correct += 1

        results[f"top_{k}_accuracy"] = correct / total

    return results
