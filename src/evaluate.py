import os
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from transformers import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate sentiment model and save reports")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to the trained model checkpoint")
    parser.add_argument("--dataset", type=str, default="yelp_polarity",
                        help="Datasets load identifier")
    parser.add_argument("--subset", type=int, default=None,
                        help="Number of samples to use from test split")
    parser.add_argument("--output-dir", type=str, default="./evaluation",
                        help="Directory where to save metrics and plots")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Recreate Trainer from checkpoint
    trainer = Trainer(model=args.model_dir)
    # Load eval dataset
    from datasets import load_dataset
    test_split = f"test[:{args.subset}]" if args.subset else "test"
    eval_ds = load_dataset(args.dataset, split=test_split)
    # Tokenize
    tokenizer = trainer.tokenizer
    def tokenize(batch): return tokenizer(batch['text'], truncation=True, max_length=128)
    tokenized_eval = eval_ds.map(tokenize, batched=True)

    # Predict
    pred_output = trainer.predict(tokenized_eval)
    logits = pred_output.predictions
    labels = pred_output.label_ids
    preds = np.argmax(logits, axis=1)

    # Accuracy
    acc = accuracy_score(labels, preds)
    with open(os.path.join(args.output_dir, "accuracy.txt"), "w") as f:
        f.write(f"Eval accuracy: {acc:.4f}\\n")

    # Classification report
    report = classification_report(labels, preds, target_names=['negative','positive'])
    with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = ['negative','positive']
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel='Predicted', ylabel='True',
        title='Confusion Matrix'
    )
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Results saved in {args.output_dir}")

if __name__ == "__main__":
    main()
