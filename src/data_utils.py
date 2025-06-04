from transformers import AutoTokenizer, default_data_collator
from datasets import load_dataset
import os

def get_dataset(model_name: str, split: str = "train", max_length: int = 128):
    cache_dir = "/content/cache/hf_datasets"
    os.makedirs(cache_dir, exist_ok=True)
    ds = load_dataset("tweet_eval", "sentiment", split=split, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess(example):
        tokens = tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        tokens["labels"] = example["label"]
        return tokens

    return ds.map(preprocess, batched=True, remove_columns=ds.column_names)

def get_dataloaders(model_name: str, batch_size: int = 16, max_length: int = 128):
    from torch.utils.data import DataLoader
    train_ds = get_dataset(model_name, split="train", max_length=max_length)
    val_ds = get_dataset(model_name, split="validation", max_length=max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=default_data_collator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=default_data_collator)

    return train_loader, val_loader
