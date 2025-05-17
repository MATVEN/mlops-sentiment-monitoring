import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import evaluate

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate sentiment model")
    parser.add_argument(
        "--model-dir", type=str, required=True,
        help="Path to the fine-tuned model directory"
    )
    parser.add_argument(
        "--dataset", type=str, default="yelp_polarity",
        help="Datasets load identifier"
    )
    parser.add_argument(
        "--subset", type=int, default=None,
        help="Number of samples to use from test split"
    )
    parser.add_argument(
        "--max-length", type=int, default=128,
        help="Max token length for tokenizer"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size per device for evaluation"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./evaluation",
        help="Directory where to save evaluation metrics"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load config and model from model-dir
    config = AutoConfig.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir,
        config=config,
        ignore_mismatched_sizes=True
    )
    # Load tokenizer from model-dir
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    # Load test dataset
    split = f"test[:{args.subset}]" if args.subset else "test"
    test_ds = load_dataset(args.dataset, split=split)

    # Tokenization
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)
    tokenized_test = test_ds.map(tokenize, batched=True)

    # Data collator and metric
    data_collator = DataCollatorWithPadding(tokenizer)
    accuracy_metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = torch.argmax(torch.tensor(logits), axis=-1)
        return accuracy_metric.compute(predictions=preds, references=labels)

    # Evaluation args
    eval_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.batch_size,
        do_train=False,
        do_eval=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Run evaluation
    metrics = trainer.evaluate()
    print("✅ Valutazione completata. Risultati:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

if __name__ == "__main__":
    main()
