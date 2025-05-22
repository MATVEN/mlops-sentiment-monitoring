import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, default_data_collator
import evaluate as hf_evaluate

ORIGINAL_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

def compute_metrics(preds, labels):
    accuracy_metric = hf_evaluate.load("accuracy")
    f1_metric = hf_evaluate.load("f1")
    preds_flat = np.argmax(preds, axis=1)
    acc = accuracy_metric.compute(predictions=preds_flat, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=preds_flat, references=labels, average="macro")["f1"]
    return {"accuracy": acc, "f1_macro": f1}

def main(
    model_dir: str = "/content/model_checkpoints/test_model",
    batch_size: int = 16,
    max_length: int = 64,
    device: str = None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Carica tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except (OSError, TypeError):
        print(f"⚠️  Tokenizer non trovato in {model_dir}, uso il nome originale {ORIGINAL_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL)

    # Carica modello
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)

    # Carica test split e preprocess
    from datasets import load_dataset
    ds_test = load_dataset("tweet_eval", "sentiment", split="test")
    def preprocess(example):
        tokens = tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        tokens["labels"] = example["label"]
        return tokens
    ds_test = ds_test.map(preprocess, batched=True, remove_columns=ds_test.column_names)

    # DataLoader
    loader = DataLoader(ds_test, batch_size=batch_size, collate_fn=default_data_collator)

    # Inferenza
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            all_preds.append(outputs.logits.cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    metrics = compute_metrics(preds, labels)
    print(f"Test metrics: Accuracy: {metrics['accuracy']:.4f}, F1-macro: {metrics['f1_macro']:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="/content/model_checkpoints/test_model")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    main(
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device
    )
