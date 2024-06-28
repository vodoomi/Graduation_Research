class CFG:
    data_type = "jaran"
    sentence_split = False
    data_path = f"../output/{data_type}/reviews.csv"
    embedding_model = "cl-nagoya/sup-simcse-ja-large" #"stsb-xlm-r-multilingual"
    random_state = 88

cfg = CFG()