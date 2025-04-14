from sentence_transformers import SentenceTransformer, models, losses
from torch.utils.data import DataLoader
from sentence_transformer_script.dataset import CodeTextDataset

# 1. Define model architecture
word_embedding_model = models.Transformer(
    'BAAI/bge-base-en', max_seq_length=512)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# 2. Create dataset
train_dataset = CodeTextDataset(
    "data/mbpp_text_code_augmented.jsonl", use_transformed=True)

# 3. Create data loader
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)

# 4. Set up loss function
train_loss = losses.MultipleNegativesRankingLoss(model=model)

# 5. Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    evaluation_steps=1000,
    output_path="output/code-text-embeddings"
)
