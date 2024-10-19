import pandas as pd
from cfg import cfg
from lib import embedding, load_data, extract_negative_reviews, summarize, CustomBERTopic

# Load data
print("Loading data...")
reviews = load_data(cfg.data_path)

# Extract negative reviews
print("Extracting negative reviews...")
negative_reviews = extract_negative_reviews(reviews, use_cache=cfg.use_sentiment_cache)

# Embed reviews
print("Embedding reviews...")
reviews_embedding = embedding(negative_reviews, use_cache=cfg.use_embedding_cache)

# Topic modeling using BERTopic
if cfg.bertopic:
    print("Performing topic modeling using BERTopic...")
    assert cfg.n_summarize_sentences == 1, "n_summarize_sentences must be 1 when using BERTopic"
    model = CustomBERTopic()
    topics, _ = model.fit_transform(documents=negative_reviews, embeddings=reviews_embedding)

# Extract common reviews using LexRank
print("Extracting common reviews...")
if cfg.bertopic:
    unique_topics = set(topics)
    common_reviews = []
    for topic in unique_topics:
        topic_reviews = [review for i, review in enumerate(negative_reviews) if topics[i] == topic]
        topic_embedding = [embedding for i, embedding in enumerate(reviews_embedding) if topics[i] == topic]
        common_reviews.extend(summarize(topic_reviews, topic_embedding, top_n=cfg.n_summarize_sentences))

else:
    common_reviews = summarize(negative_reviews, reviews_embedding, topics, top_n=cfg.n_summarize_sentences)

# Save common reviews
print("Saving common reviews...")
if cfg.bertopic:
    common_reviews_path = f"{cfg.output_dir}/common_reviews_bertopic.csv"
    pd.DataFrame({"topic": list(unique_topics), "common_review": common_reviews}).to_csv(common_reviews_path, index=False)
else:
    common_reviews_path = f"{cfg.output_dir}/common_reviews.csv"
    pd.DataFrame(common_reviews).to_csv(common_reviews_path, index=False, header=False)