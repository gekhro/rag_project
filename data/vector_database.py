from transformers import AutoModel, AutoTokenizer
import torch
import json
import os

data_dir = "./Processed_Datasets"

model_name = "BAAI/bge-base-en"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**inputs)
    return output.last_hidden_state[:, 0, :].squeeze().numpy()  # CLS token embedding


dimension = 768 # Embedding size (e.g., 768)
index = faiss.IndexFlatL2(dimension)
# Load JSON file
for data_file in os.listdir(data_dir):
    data_file = os.path.join(data_dir, data_file)

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract queries and code snippets
    queries = [item["text"] for item in data]
    code_snippets = [item["code"] for item in data]

    # Print examples
    print("Queries:", queries[0])
    print("Code Snippets:", code_snippets[0])
    query_embeddings = [get_embedding(query) for query in queries]
    code_embeddings = [get_embedding(code) for code in code_snippets]
    code_embeddings_np = np.array(code_embeddings, dtype="float32")
    index.add(code_embeddings_np)


faiss.write_index(index, "code_faiss_index.bin")
print("FAISS index saved successfully!")