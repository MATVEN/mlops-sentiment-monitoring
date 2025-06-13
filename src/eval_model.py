import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, default_data_collator
import evaluate as hf_evaluate
from pathlib import Path
import glob

# -----------------------------
# Se non viene passato model_dir, rileva automaticamente l'ultimo best_model
# -----------------------------
def get_latest_model_dir(base_dir="model_checkpoints"):
    pattern = os.path.join(base_dir, "best_model_*")
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        raise FileNotFoundError(f"Nessun modello trovato in {base_dir}")
    return dirs[-1]

# -----------------------------
# Parser linea di comando
# -----------------------------
import argparse
parser = argparse.ArgumentParser(description="Evaluate sentiment model")
parser.add_argument("--model_dir", type=str, default=None,
                    help="Directory del modello da valutare (default: ultimo best_model)")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--max_length", type=int, default=64)
parser.add_argument("--device", type=str, default=None)
args = parser.parse_args()

MODEL_DIR = args.model_dir or get_latest_model_dir()
DEVICE = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

# Carica tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
except Exception:
    print(f"⚠️  Tokenizer locale non trovato in {MODEL_DIR}, scarico dal hub")
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

# Carica modello
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    local_files_only=True
).to(DEVICE)

# Carica dataset test
from datasets import load_dataset
ds_test = load_dataset("tweet_eval", "sentiment", split="test")

# Preprocess

def preprocess(example):
    tokens = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=args.max_length
    )
    tokens["labels"] = example["label"]
    return tokens

ds_test = ds_test.map(preprocess, batched=True, remove_columns=ds_test.column_names)
loader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=default_data_collator)

# Inferenza
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        logits = model(**batch).logits
        all_preds.append(logits.cpu().numpy())
        all_labels.append(batch["labels"].cpu().numpy())

preds = np.concatenate(all_preds, axis=0)
labels = np.concatenate(all_labels, axis=0)

# Metriche
accuracy = hf_evaluate.load("accuracy")
f1       = hf_evaluate.load("f1")

preds_flat = np.argmax(preds, axis=1)
acc = accuracy.compute(predictions=preds_flat, references=labels)["accuracy"]
f1m = f1.compute(predictions=preds_flat, references=labels, average="macro")["f1"]

print(f"Test metrics on {MODEL_DIR}")
print(f" - Accuracy: {acc:.4f}")
print(f" - F1-macro: {f1m:.4f}")
