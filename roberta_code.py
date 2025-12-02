# ==============================
# RoBERTa training — Transformers 4.56.0 compatible
# Run this code to install RoBerta Model for Cyberbullying Classification
# Dataset: https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification  "install from this link and replace DF_PATH"

# ==============================

import os
os.environ["WANDB_DISABLED"] = "true"   

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from datasets import Dataset
import torch
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments
)

# -----------------------
# 1) Load dataset
# -----------------------
print("[INFO] Loading dataset...")
DF_PATH = "/kaggle/input/cyberbullying-classification/cyberbullying_tweets.csv"
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
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

def tokenize_fn(batch):
    return tokenizer(batch["tweet_text"], padding="max_length", truncation=True, max_length=128)

train_ds = train_ds.map(tokenize_fn, batched=True)
test_ds = test_ds.map(tokenize_fn, batched=True)

train_ds = train_ds.rename_column("label", "labels")
test_ds = test_ds.rename_column("label", "labels")

train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# -----------------------
# 4) Model
# -----------------------
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id
)

# -----------------------
# 5) Training Arguments
# -----------------------
training_args = TrainingArguments(
    output_dir="./results_roberta",
    eval_strategy="epoch",          # for Transformers v4.56+
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./logs_roberta",
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
print("[INFO] Starting RoBERTa training...")
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
save_dir = "/kaggle/working/roberta_model"
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)
print("✅ Model saved to:", save_dir)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --- Confusion Matrix ---
print("[INFO] Plotting confusion matrix...")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=list(id2label.values()),
            yticklabels=list(id2label.values()))
plt.title("Confusion Matrix — DistilBERT")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.tight_layout()
plt.show()

# --- Training metrics history ---
print("[INFO] Plotting training loss and accuracy...")

# Trainer keeps logs in trainer.state.log_history
logs = pd.DataFrame(trainer.state.log_history)

# Some entries might not have loss/accuracy (like eval_only steps)
train_logs = logs.dropna(subset=["loss"])
eval_logs = logs.dropna(subset=["eval_loss"])

# Plot training loss
plt.figure(figsize=(8, 5))
plt.plot(train_logs["step"], train_logs["loss"], label="Training Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss over Steps")
plt.legend()
plt.grid(True)
plt.show()

# Plot eval accuracy and loss per epoch
if "eval_accuracy" in eval_logs.columns:
    plt.figure(figsize=(8, 5))
    plt.plot(eval_logs["epoch"], eval_logs["eval_accuracy"], marker="o", label="Eval Accuracy")
    plt.plot(eval_logs["epoch"], eval_logs["eval_loss"], marker="o", label="Eval Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Eval Accuracy & Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()

print("✅ Charts generated successfully!")

