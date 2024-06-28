from cfg import cfg
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from umap import UMAP
from bertopic import BERTopic

# num_repsentative_docsをトピックで各1つに変更
class CustomBERTopic(BERTopic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.umap_model = UMAP(n_neighbors=15,
                               n_components=5,
                               min_dist=0.0,
                               metric='cosine',
                               low_memory=self.low_memory,
                               random_state=cfg.random_state)

    def _save_representative_docs(self, documents: pd.DataFrame):
        """ Save the 3 most representative docs per topic

        Arguments:
            documents: Dataframe with documents and their corresponding IDs

        Updates:
            self.representative_docs_: Populate each topic with 3 representative docs
        """
        repr_docs, _, _, _ = self._extract_representative_docs(
            self.c_tf_idf_,
            documents,
            self.topic_representations_,
            nr_samples=500,
            nr_repr_docs=1
        )
        self.representative_docs_ = repr_docs



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
        splitted_reviews.extend(review.split("。"))
    return splitted_reviews


def load_data(data_path):
    """
    Load data from the given path.

    Args:
        data_path (str): Path to the data.
    
    Returns:
        list: List of reviews.
    """
    data = pd.read_csv(data_path)
    reviews = data["review"].tolist()
    if cfg.sentence_split:
        reviews = split_one_sentence(reviews)
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
    emb_path = f"../output/{cfg.data_type}/reviews_emb_{model_name}.npy"

    # Load the embeddings from cache
    if use_cache:
        try:
            sentence_embedding = np.load(emb_path)
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