import pandas as pd
from cfg import cfg
from lib import embedding, load_data, extract_negative_reviews, summarize

# Load data
print("Loading data...")
reviews = load_data(cfg.data_path)

# Extract negative reviews
print("Extracting negative reviews...")
negative_reviews = extract_negative_reviews(reviews, use_cache=cfg.use_sentiment_cache)

# Embed reviews
print("Embedding reviews...")
reviews_embedding = embedding(negative_reviews, use_cache=cfg.use_embedding_cache)

# Extract common reviews using 
print("Extracting common reviews...")
common_reviews = summarize(negative_reviews, reviews_embedding, top_n=cfg.n_summarize_sentences)

# Save common reviews
print("Saving common reviews...")
common_reviews_path = f"../output/{cfg.data_type}/facility_{cfg.facility_id}/common_reviews.csv"
pd.DataFrame(common_reviews).to_csv(common_reviews_path, index=False, header=False)