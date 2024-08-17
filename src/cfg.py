class CFG:
    data_type = "jaran"
    sentence_split = True
    data_path = f"../output/{data_type}/reviews.csv"
    embedding_model = "cl-nagoya/sup-simcse-ja-large" #"stsb-xlm-r-multilingual"
    sentiment_model = "jarvisx17/japanese-sentiment-analysis"
    sentiment_batch_size = 16
    random_state = 88
    lex_rank_threshold = 0.1
    n_summarize_sentences = 5

cfg = CFG()