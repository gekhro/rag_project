import torch
from sentence_transformers import util
import numpy as np

def dcg(scores):
    return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(scores))

def evaluate_retrieval(model, test_data, top_k=(1, 5, 10), device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)

    # Extract inputs
    descriptions = [item["text"] for item in test_data]
    codes = [item["code"] for item in test_data]

    # Encode on device
    description_embeddings = model.encode(descriptions, convert_to_tensor=True, device=device, batch_size=32)
    code_embeddings = model.encode(codes, convert_to_tensor=True, device=device, batch_size=32)

    # Compute cosine similarity matrix
    cos_sim = util.pytorch_cos_sim(description_embeddings, code_embeddings).to(device)

    total = len(descriptions)
    targets = torch.arange(total, device=device)

    # Get ranks for each query
    sorted_indices = torch.argsort(cos_sim, dim=1, descending=True)

    # Store metric accumulators
    mrr_total = 0.0
    map_total = 0.0
    ndcg_total = {k: 0.0 for k in top_k}
    recall_total = {k: 0 for k in top_k}
    accuracy_total = {k: 0 for k in top_k}

    for i in range(total):
        ranking = sorted_indices[i].tolist()
        target = targets[i].item()

        # MRR
        rank_pos = ranking.index(target) + 1
        mrr_total += 1.0 / rank_pos

        # MAP@K (binary relevance — only 1 true target)
        for k in top_k:
            hits = [1 if ranking[j] == target else 0 for j in range(k)]
            precisions = [hits[j] / (j + 1) for j in range(k) if hits[j] == 1]
            map_total += sum(precisions) / 1 if precisions else 0

        # Recall@K and Accuracy@K
        for k in top_k:
            top_k_pred = ranking[:k]
            if target in top_k_pred:
                recall_total[k] += 1
                accuracy_total[k] += 1

        # nDCG@K
        for k in top_k:
            relevance = [1 if ranking[j] == target else 0 for j in range(k)]
            ideal = [1] + [0] * (k - 1)
            ndcg = dcg(relevance) / dcg(ideal)
            ndcg_total[k] += ndcg

    # Finalize metrics
    results = {}

    for k in top_k:
        results[f"top_{k}_accuracy"] = accuracy_total[k] / total
        results[f"recall@{k}"] = recall_total[k] / total
        results[f"ndcg@{k}"] = ndcg_total[k] / total
        results[f"map@{k}"] = map_total / total

    results["mrr"] = mrr_total / total

    return results
