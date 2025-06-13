from transformers import AutoTokenizer, default_data_collator
from datasets import load_dataset
import os

def get_dataset(model_name: str, split: str = "train", max_length: int = 128):
    """
    Funzione per caricare e preprocessare il dataset 'tweet_eval' per sentiment analysis.

    Args:
        model_name (str): Nome del modello pre-addestrato per il tokenizer.
        split (str): Suddivisione del dataset da caricare ('train', 'validation', 'test').
        max_length (int): Lunghezza massima per il padding e il truncamento del testo.

    Returns:
        Dataset preprocessato con tokenizzazione e label associata.
    """
    # Creazione della cartella cache per Hugging Face datasets (se non esiste)
    cache_dir = "/content/cache/hf_datasets"
    os.makedirs(cache_dir, exist_ok=True)

    # Caricamento del dataset tweet_eval, split specificato, con cache locale
    ds = load_dataset("tweet_eval", "sentiment", split=split, cache_dir=cache_dir)

    # Inizializzazione tokenizer dal modello pre-addestrato specificato
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess(example):
        # Tokenizzazione del testo con padding e truncamento a max_length
        tokens = tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        # Aggiunta del label originale nel dizionario tokenizzato
        tokens["labels"] = example["label"]
        return tokens

    # Applicazione del preprocessamento a tutto il dataset rimuovendo le colonne originali
    return ds.map(preprocess, batched=True, remove_columns=ds.column_names)

def get_dataloaders(model_name: str, batch_size: int = 16, max_length: int = 128):
    """
    Funzione per ottenere DataLoader di training e validation.

    Args:
        model_name (str): Nome del modello pre-addestrato per il tokenizer.
        batch_size (int): Dimensione del batch per i DataLoader.
        max_length (int): Lunghezza massima del token.

    Returns:
        Tuple contenente DataLoader per training e validation.
    """
    from torch.utils.data import DataLoader

    # Caricamento dataset train e validation preprocessati
    train_ds = get_dataset(model_name, split="train", max_length=max_length)
    val_ds = get_dataset(model_name, split="validation", max_length=max_length)

    # Creazione DataLoader con shuffling per il training, senza per validation
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=default_data_collator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=default_data_collator)

    return train_loader, val_loader
