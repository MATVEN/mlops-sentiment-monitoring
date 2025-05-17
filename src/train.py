import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import evaluate

def parse_args():
    parser = argparse.ArgumentParser(description="Train sentiment model")
    parser.add_argument("--model-name", type=str,
                        default="cardiffnlp/twitter-roberta-base-sentiment-latest",
                        help="HuggingFace model identifier")
    parser.add_argument("--dataset", type=str, default="yelp_polarity",
                        help="Datasets load identifier")
    parser.add_argument("--subset", type=int, default=None,
                        help="Number of samples to use from train/test splits")
    parser.add_argument("--max-length", type=int, default=128,
                        help="Max token length for tokenizer")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size per device")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Directory to use for HuggingFace dataset cache")
    parser.add_argument("--output-dir", type=str, default="./results",
                        help="Output directory for checkpoints and logs")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model & tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2, ignore_mismatched_sizes=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Prepare cache kwargs
    load_kwargs = {}
    if args.cache_dir:
        load_kwargs["cache_dir"] = args.cache_dir

    # Load dataset
    train_split = f"train[:{args.subset}]" if args.subset else "train"
    eval_split = f"test[:{int(args.subset/10)}]" if args.subset else "test"
    train_ds = load_dataset(args.dataset, split=train_split, **load_kwargs)
    eval_ds = load_dataset(args.dataset, split=eval_split, **load_kwargs)

    # Tokenization
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)
    tokenized_train = train_ds.map(tokenize, batched=True)
    tokenized_eval = eval_ds.map(tokenize, batched=True)

    # Data collator & metrics
    data_collator = DataCollatorWithPadding(tokenizer)
    metric_acc = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = torch.argmax(torch.tensor(logits), dim=-1)
        return metric_acc.compute(predictions=preds, references=labels)

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=500,
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=1000,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_dir=os.path.join(args.output_dir, "logs"),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()

    # Save model & tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"✅ Training completo. Modello e tokenizer salvati in {args.output_dir}")

if __name__ == "__main__":
    main()
