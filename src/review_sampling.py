import polars as pl

from lib import load_data, extract_negative_reviews
from cfg import cfg

# Load data
print("Loading data...")
reviews = load_data(cfg.data_path)

# Extract negative reviews
print("Extracting negative reviews...")
negative_reviews = extract_negative_reviews(reviews, use_cache=cfg.use_sentiment_cache)

# Create a DataFrame
reviews_df = pl.DataFrame(
    {
        "review": negative_reviews
    }
)

# Sample reviews
print("Sampling reviews...")
sampled_reviews_df = reviews_df.sample(n=cfg.n_sample_reviews, shuffle=True, seed=cfg.random_state)

# Save sampled reviews
print("Saving sampled reviews...")
sampled_reviews_path = f"{cfg.output_dir}/sampled_reviews.csv"
sampled_reviews_df.write_csv(sampled_reviews_path)