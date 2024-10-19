class CFG:
    data_type = "rakuten"
    sentence_split = True
    use_sentiment_cache = True
    use_embedding_cache = True
    data_path = f"../output/{data_type}/all_reviews.csv"
    embedding_model = "cl-nagoya/sup-simcse-ja-large" #"stsb-xlm-r-multilingual"
    sentiment_model = "jarvisx17/japanese-sentiment-analysis"
    sentiment_batch_size = 16
    random_state = 88
    lex_rank_threshold = 0.1
    n_summarize_sentences = 5
    facility_id = 5553

cfg = CFG()