from cfg import cfg
from lib import embedding, load_data, CustomBERTopic

# Load data
print("Loading data...")
reviews = load_data(cfg.data_path)

# Embed reviews
print("Embedding reviews...")
reviews_embedding = embedding(reviews, use_cache=True)

# Extract common reviews using BERTopic
print("Extracting common reviews...")
model = CustomBERTopic()
topics, _ = model.fit_transform(documents=reviews, embeddings=reviews_embedding)
representative_docs = model.get_representative_docs()

# Save the representative documents
print("Saving representative documents...")
if cfg.sentence_split:
    path = f"../output/{cfg.data_type}/representative_docs_split.csv"
else:
    path = f"../output/{cfg.data_type}/representative_docs.csv"

with open(path, 'w', encoding='utf-8') as f:
    for topic, docs in representative_docs.items():
        f.write(f"Topic {topic}\n")
        for doc in docs:
            f.write(f"{doc}\n")
        f.write("\n")
