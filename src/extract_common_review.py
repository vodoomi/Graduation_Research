from cfg import cfg
from lib import embedding, load_data
import pandas as pd

# Load data
print("Loading data...")
reviews = load_data(cfg.data_path)

# Embed reviews
print("Embedding reviews...")
reviews_embedding = embedding(reviews, use_cache=True)

# Print the sentence embeddings
print(reviews_embedding.shape)