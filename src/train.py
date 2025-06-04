import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
)
from torch.optim import AdamW
import evaluate
from data_utils import get_dataloaders

# Configurazione parametri (modifica se necessario)
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BATCH_SIZE = 16
MAX_LENGTH = 64
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
CACHE_DIR = "/content/cache/hf_datasets"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carica dataset e dataloader
train_loader, val_loader = get_dataloaders(
    model_name=MODEL_NAME,
    batch_size=BATCH_SIZE,
    max_length=MAX_LENGTH,
)

# Carica modello e tokenizer
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3).to(DEVICE)

# Ottimizzatore e scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
num_training_steps = NUM_EPOCHS * len(train_loader)
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Metriche con il pacchetto evaluate
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(preds, labels):
    preds_flat = np.argmax(preds, axis=1)
    acc = accuracy_metric.compute(predictions=preds_flat, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=preds_flat, references=labels, average="macro")["f1"]
    return {"accuracy": acc, "f1_macro": f1}

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Training loss: {avg_train_loss:.4f}")

    # Validazione
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            all_preds.append(logits.cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    metrics = compute_metrics(all_preds, all_labels)
    print(f"Validation metrics: Accuracy: {metrics['accuracy']:.4f}, F1-macro: {metrics['f1_macro']:.4f}")

# Salvataggio modello (facoltativo)
output_dir = "/content/model_checkpoints/test_model"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
print(f"✅ Modello salvato in: {output_dir}")
