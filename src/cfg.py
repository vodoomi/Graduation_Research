class CFG:
    data_type = "rakuten"
    sentence_split = True
    bertopic = False
    use_sentiment_cache = True
    use_embedding_cache = True
    sentiment_batch_size = 16
    random_state = 88
    lex_rank_threshold = 0.1
    n_summarize_sentences = 5
    n_sample_reviews = 500
    max_length = 512
    facility_id = 19455
    data_path = f"../output/{data_type}/all_reviews.csv"
    output_dir = f"../output/{data_type}/facility_{facility_id}"
    embedding_model = "cl-nagoya/sup-simcse-ja-large" #"stsb-xlm-r-multilingual"
    sentiment_model = "jarvisx17/japanese-sentiment-analysis"

cfg = CFG()