# ===========================================================================================================
    # DistilBERT training — Transformers 4.56.0 compatible.

    # Run this file and 
    # install all files in one folder and you are good to go for prediction by running app.py file.

    # https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification 
    #  "install from this link and replace DF_PATH"

    # =======================================================================================================

import os
os.environ["WANDB_DISABLED"] = "true"   # disable wandb logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from datasets import Dataset
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
# -----------------------
# 1) Load dataset
# -----------------------
print("[INFO] Loading dataset...")
DF_PATH = "/content/drive/MyDrive/cyberbullying_tweets.csv"
df = pd.read_csv(DF_PATH)
print("[INFO] Columns:", df.columns)
print("[INFO] Classes:", df["cyberbullying_type"].unique())
# encode labels
labels_cat = df["cyberbullying_type"].astype("category").cat
df["label"] = labels_cat.codes
id2label = dict(enumerate(labels_cat.categories))
label2id = {v: k for k, v in id2label.items()}
# -----------------------
# 2) Train/test split
# -----------------------
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
print(f"[INFO] Train size: {len(train_df)}, Test size: {len(test_df)}")
train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))
# -----------------------
# 3) Tokenizer
# -----------------------
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
def tokenize_fn(batch):
    return tokenizer(batch["tweet_text"], padding="max_length", truncation=True, max_length=128)
train_ds = train_ds.map(tokenize_fn, batched=True)
test_ds  = test_ds.map(tokenize_fn, batched=True)
train_ds = train_ds.rename_column("label", "labels")
test_ds  = test_ds.rename_column("label", "labels")
train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
# -----------------------
# 4) Model
# -----------------------
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id
)
# -----------------------
# 5) Training Arguments
# -----------------------
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",           # ✅ correct for 4.56.0
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./logs",
    logging_steps=50,
    report_to="none"
)
# -----------------------
# 6) Metrics function
# -----------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "precision_weighted": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall_weighted": recall_score(labels, preds, average="weighted", zero_division=0),
    }
# -----------------------
# 7) Trainer
# -----------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
# -----------------------
# 8) Train
# -----------------------
print("[INFO] Starting training...")
trainer.train()
# -----------------------
# 9) Evaluate
# -----------------------
print("[INFO] Evaluating on test set...")
pred_out = trainer.predict(test_ds)
y_true = pred_out.label_ids
y_pred = np.argmax(pred_out.predictions, axis=1)
print("✅ Test Accuracy:", accuracy_score(y_true, y_pred))
print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=list(id2label.values()), zero_division=0))
# -----------------------
# 10) Save
# -----------------------
save_dir = "/content/drive/MyDrive/distilbert_cyber_model"
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)

print("✅ Model saved to:", save_dir)
