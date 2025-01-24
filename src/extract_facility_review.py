import polars as pl

from lib import load_data, extract_negative_reviews
from cfg import cfg

# Load data
print("Loading data...")
reviews = load_data(cfg.data_path)

reviews_with_labels = pl.read_csv(f"{cfg.output_dir}/sampled_reviews_with_labels.csv", encoding="cp932")

# Extract negative reviews
print("Extracting negative reviews...")
negative_reviews = extract_negative_reviews(reviews, use_cache=cfg.use_sentiment_cache)

# Create a DataFrame
reviews_df = pl.DataFrame(
    {
        "review": negative_reviews
    }
)

# Join reviews with labels
reviews_df = reviews_df.join(reviews_with_labels, on="review", how="left")

# Save facility reviews
print("Saving facility reviews...")
facility_reviews_path = f"{cfg.output_dir}/facility_reviews.csv"
reviews_df.to_pandas().to_csv(facility_reviews_path, index=False, encoding="cp932")