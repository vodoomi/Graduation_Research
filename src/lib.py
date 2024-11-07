import os
import re

import numpy as np
import pandas as pd
import polars as pl
from umap import UMAP
from bertopic import BERTopic
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import rustworkx as rx
from tqdm import tqdm

from cfg import cfg

# UMAPのseedを固定
class CustomBERTopic(BERTopic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.umap_model = UMAP(n_neighbors=15,
                               n_components=5,
                               min_dist=0.0,
                               metric='cosine',
                               low_memory=self.low_memory,
                               random_state=cfg.random_state)


def extract_specific_facility_reviews(facility_id):
    """
    Extract reviews of a specific facility.

    Args:
        facility_id (str): Facility ID.
    
    Returns:
        list: List of reviews of the specific facility.
    """
    all_reviews_df = pl.read_csv(cfg.data_path)
    facility_reviews_df = all_reviews_df.filter((pl.col("facility_id") == facility_id) & (pl.col("review").is_not_null()))
    return facility_reviews_df["review"].to_list()


def split_one_sentence(reviews):
    """
    Split one sentence into multiple sentences.

    Args:
        reviews (list): List of reviews.
    
    Returns:
        list: List of splitted sentences.
    """
    splitted_reviews = []
    for review in reviews:
        splitted_reviews.extend(re.split(r'[。｡]', review))
    return splitted_reviews


def preprocess_reviews(reviews):
    """
    Preprocess reviews. Remove short reviews and the part after "【ご利用の宿泊プラン】".

    Args:
        reviews (list): List of reviews.
    
    Returns:
        list: List of preprocessed reviews.
    """
    # 【ご利用の宿泊プラン】以降の文字列を削除し、全角・半角スペースを削除
    preprocessed_reviews = [x.split("【ご利用の宿泊プラン】")[0].strip().replace("\u3000", "").replace(" ", "") for x in reviews]
    # 短いレビューを削除
    preprocessed_reviews = [x for x in preprocessed_reviews if len(x) >= cfg.min_review_length]
    return preprocessed_reviews


def load_data(data_path):
    """
    Load data from the given path.

    Args:
        data_path (str): Path to the data.
    
    Returns:
        list: List of reviews.
    """
    assert cfg.data_type in ["jaran", "rakuten"], "cfg.data_type must be either 'jaran' or 'rakuten'."
    if cfg.data_type == "jaran":
        data = pd.read_csv(data_path)
        reviews = data["review"].tolist()
    elif cfg.data_type == "rakuten":
        reviews = extract_specific_facility_reviews(cfg.facility_id)
    if cfg.sentence_split:
        reviews = split_one_sentence(reviews)
    reviews = preprocess_reviews(reviews)
    print(f"Number of reviews: {len(reviews)}")
    return reviews


def get_after_slash(string):
    """
    Get the substring after the last slash.

    Args:
        string (str): Input string.
    
    Returns:
        str: Substring after the last slash.
    """
    if "/" in string:
        return string.split("/")[-1]
    else:
        return string
    

def get_before_slash(string):
    """
    Get the substring before the last slash.

    Args:
        string (str): Input string.
    
    Returns:
        str: Substring before the last slash.
    """
    if "/" in string:
        return string.split("/")[0]
    else:
        return string


def tokenize(example):
    # Import inside the function to avoid error
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.sentiment_model)

    tokenized = tokenizer(
        example["review"],
        max_length=cfg.max_length,
        truncation=True,
    ) 
        
    return tokenized


def sentiment_analysis(reviews, use_cache):
    """
    Perform sentiment analysis on the given reviews.

    Args:
        reviews (list): List of reviews.
    
    Returns:
        sentiments (np.array): Sentiment predictions.
        review_df (pd.DataFrame): Dataframe of reviews
    """
    # Get the model name and sentiment predictions path
    model_name = get_before_slash(cfg.sentiment_model)
    if cfg.sentence_split:
        model_name = "split_" + model_name
    
    # Remove empty reviews
    review_df = pd.DataFrame(reviews, columns=["review"])
    review_df = review_df.query("review != ''")

    # Load the predictions from cache
    if use_cache:
        try:
            predictions = np.load(f"{cfg.output_dir}/sentiment_analysis_predictions_{model_name}.npy")
            sentiments = np.argmax(predictions, axis=1)
            assert len(sentiments) == len(review_df), "Number of sentiments and reviews do not match. Set use_cache=False to recompute."
            return sentiments, review_df
        except FileNotFoundError:
            pass
    
    # Tokenize the reviews
    ds = Dataset.from_pandas(review_df)
    ds = ds.map(tokenize, batched=True)

    # Perform sentiment analysis
    tokenizer = AutoTokenizer.from_pretrained(cfg.sentiment_model)
    collator = DataCollatorWithPadding(tokenizer)
    args = TrainingArguments(
        ".", 
        per_device_eval_batch_size=cfg.sentiment_batch_size, 
        report_to="none",
    )

    model = AutoModelForSequenceClassification.from_pretrained(cfg.sentiment_model)
    trainer = Trainer(
        model=model, 
        args=args, 
        data_collator=collator, 
        tokenizer=tokenizer,
    )
    predictions = trainer.predict(ds).predictions
    os.makedirs(cfg.output_dir, exist_ok=True)
    np.save(f"{cfg.output_dir}/sentiment_analysis_predictions_{model_name}.npy", predictions)
    sentiments = np.argmax(predictions, axis=1)
    
    return sentiments, review_df


def extract_negative_reviews(reviews, use_cache=True):
    """
    Extract negative reviews from the given reviews.

    Args:
        reviews (list): List of reviews.
    
    Returns:
        list: List of negative reviews
    """
    # Perform sentiment analysis
    sentiments, review_df = sentiment_analysis(reviews, use_cache)

    # Extract negative reviews
    negative_reviews = review_df.loc[sentiments == 0, "review"]

    print(f"Number of negative reviews: {len(negative_reviews)}")
    
    return negative_reviews.tolist()


def embedding(reviews, use_cache=True):
    """
    Embed reviews using SentenceTransformer model.

    Args:
        reviews (list): List of reviews.
        use_cache (bool): Whether to use cache.

    Returns:
        sentence_embedding (np.array): Sentence embeddings.
    """
    # Get the model name and reviews embedding path
    model_name = get_after_slash(cfg.embedding_model)
    if cfg.sentence_split:
        model_name = "split_" + model_name
    emb_path = f"{cfg.output_dir}/reviews_emb_{model_name}.npy"

    # Load the embeddings from cache
    if use_cache:
        try:
            sentence_embedding = np.load(emb_path)
            assert len(sentence_embedding) == len(reviews), "Number of embeddings and reviews do not match. Set use_cache=False to recompute."
            return sentence_embedding
        except FileNotFoundError:
            pass

    # Load the model
    model = SentenceTransformer(cfg.embedding_model)
    
    # Embed reviews
    sentence_embedding = model.encode(reviews, convert_to_numpy=True)

    # Save the embeddings
    np.save(emb_path, sentence_embedding)
    
    return sentence_embedding


# 1. コサイン類似度行列の作成
def build_similarity_matrix(embeddings):
    # cosine_similarity関数を使って類似度行列を作成
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix

# 2. グラフの構築とPageRankの計算
def lexrank(similarity_matrix, threshold=0.1, damping=0.85):
    # 類似度行列を閾値でフィルタリングして隣接行列を作成
    adjacency_matrix = np.where(similarity_matrix > threshold, similarity_matrix, 0)
    adjacency_matrix = adjacency_matrix.astype(np.float64)

    # 隣接行列からグラフを作成
    graph = rx.PyDiGraph.from_adjacency_matrix(adjacency_matrix)
    
    # rustworkxのPageRank関数を使用してスコアを計算
    scores = rx.pagerank(graph, alpha=damping, weight_fn=lambda edge: edge)
    return scores

# 3. スコアに基づく文の選択
def summarize(sentences, embeddings, top_n=3):
    similarity_matrix = build_similarity_matrix(embeddings)
    scores = lexrank(similarity_matrix, threshold=cfg.lex_rank_threshold)
    # スコアが高い順に文を選択
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summary = [s for _, s in ranked_sentences[:top_n]]
    return summary