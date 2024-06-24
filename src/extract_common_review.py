from cfg import cfg
from lib import embedding
from sentence_transformers import SentenceTransformer
import pandas as pd

# Load the model
print("Loading the model...")
model = SentenceTransformer(cfg.embedding_model)

# Load data
print("Loading data...")
data = pd.read_csv(cfg.data_path)
print(len(data))

# Embed reviews
print("Embedding reviews...")
reviews_embedding = embedding(data["review"].tolist(), use_cache=True)

# Print the sentence embeddings
print(reviews_embedding.shape)