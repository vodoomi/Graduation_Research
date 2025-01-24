import random
import os
import gc
from glob import glob
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from scipy.special import softmax


TRAINING_MODEL_PATH = "globis-university/deberta-v3-japanese-base"
MAX_LENGTH = 512
N_FOLD = 5
SEED = 88

def seed_torch(seed=88):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_torch(SEED)

train_df = pd.read_csv(
    "../output/rakuten/facility_19455/facility_reviews.csv", 
    encoding="cp932", 
    usecols=["review", "labels"]
)
train_df = train_df.dropna().reset_index(drop=True)

# Split train and test
train_df, test_df = train_test_split(train_df, test_size=250, random_state=SEED, stratify=train_df["labels"])
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

# Cross Validation
def get_cv(df, y, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    return list(skf.split(df, y))

cv = get_cv(train_df, train_df["labels"], N_FOLD)

for i, (_, valid_idx) in enumerate(cv):
    train_df.loc[valid_idx, "fold"] = i

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(TRAINING_MODEL_PATH)

def tokenize(example):
    tokenized = tokenizer(
        example["review"], 
        return_offsets_mapping=True, 
        truncation=True,
        max_length=MAX_LENGTH
    )   
    length = len(tokenized.input_ids)
        
    return {
        **tokenized,
        "length": length
    }

# metrics
def compute_metrics(p):
    preds, labels = p
    score = roc_auc_score(labels, preds[:, 1])
    return { 'auc':score }

# Training
oof = np.zeros((len(train_df), ), dtype="float32")
for fold in range(N_FOLD):
    train_ds = Dataset.from_pandas(train_df[train_df["fold"]!=fold])
    valid_ds = Dataset.from_pandas(train_df[train_df["fold"]==fold])
    
    train_ds = train_ds.map(tokenize, num_proc=4).remove_columns(['review', 'fold', '__index_level_0__'])
    valid_ds = valid_ds.map(tokenize, num_proc=4).remove_columns(['review', 'fold', '__index_level_0__'])
    
    model = AutoModelForSequenceClassification.from_pretrained(TRAINING_MODEL_PATH, num_labels=2)

    collator = DataCollatorWithPadding(tokenizer)

    args = TrainingArguments(
        output_dir=f"../output/rakuten/facility_19455/models/fold_{fold}",  
        fp16=True,
        learning_rate=2e-5,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        report_to="none",
        eval_strategy="epoch",
        do_eval=True,
        eval_steps=1,
        save_total_limit=1,
        save_strategy="epoch",
        save_steps=1,
        logging_steps=1,
        lr_scheduler_type='linear',
        metric_for_best_model="auc",
        greater_is_better=True,
        warmup_ratio=0.1,
        weight_decay=0.01,
    )


    trainer = Trainer(
        model=model, 
        args=args, 
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator, 
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )


    trainer.train()
    
    pred_i = trainer.predict(valid_ds).predictions[:, 1]
    valid_idx = cv[fold][1]
    oof[valid_idx] = pred_i
        
    del model, train_ds, valid_ds
    gc.collect()

# Save OOF
pd.DataFrame(oof, columns=["pred"]).to_csv("../output/rakuten/facility_19455/oof.csv")
score = roc_auc_score(train_df["labels"].values, oof)
print("WHOLE_SCORE:", score)

# Inference
MODEL_PATHES = glob("../output/rakuten/facility_19455/models/*/*")

test_ds = Dataset.from_pandas(test_df)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHES[0])
test_ds = test_ds.map(tokenize, num_proc=4)

collator = DataCollatorWithPadding(tokenizer)
args = TrainingArguments(
    "../output/rakuten/facility_19455/inference", 
    per_device_eval_batch_size=1, 
    report_to="none",
)

predictions = np.zeros((len(test_df), 2), dtype="float32")
for model_path in MODEL_PATHES:
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    trainer = Trainer(
        model=model, 
        args=args, 
        data_collator=collator, 
        tokenizer=tokenizer,
    )
    predictions +=  trainer.predict(test_ds).predictions / len(MODEL_PATHES)
    
    del model, trainer
    gc.collect()

# Score of test data
print(roc_auc_score(test_df["labels"], predictions[:, 1]))

def find_optimal_threshold(y_true, y_scores):
    thresholds = np.linspace(0, 1, 101)
    best_threshold = 0
    best_score = 0

    for threshold in thresholds:
        y_pred = (y_scores > threshold).astype(int)
        score = f1_score(y_true, y_pred)

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


best_threshold, best_f1_score = find_optimal_threshold(test_df["labels"], softmax(predictions, axis=1)[:, 1])
print(f"Best threshold: {best_threshold}") # 最適な閾値
print(f"Best F1 Score: {best_f1_score}") # 最適な閾値でのF1スコア

hard_pred = (softmax(predictions, axis=1)[:, 1] > best_threshold).astype(int)
recall = recall_score(test_df["labels"], hard_pred)
prec = precision_score(test_df["labels"], hard_pred)

print(f"Recall:{recall:.4f}")
print(f"Precision:{prec:.4f}")