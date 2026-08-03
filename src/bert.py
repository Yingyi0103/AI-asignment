from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

try:
    from src.evaluation import evaluate_model
except ImportError:
    from evaluation import evaluate_model


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
SAVED_MODELS_DIR = ROOT_DIR / "saved_models"
CLEANED_DATA_PATH = DATA_DIR / "cleaned_amazon_reviews.csv"
BERT_MODEL_DIR = SAVED_MODELS_DIR / "bert_sentiment_model"

SAVED_MODELS_DIR.mkdir(exist_ok=True)


class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, index):
        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item

    def __len__(self):
        return len(self.labels)


def train_bert(
    model_name="distilbert-base-uncased",
    num_train_epochs=2,
    train_batch_size=8,
    eval_batch_size=16,
):
    """Fine-tune a binary sentiment BERT model on the cleaned dataset."""
    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise ImportError("transformers is required to train the BERT model.") from exc

    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(
            "Cleaned dataset not found. Run `src/data_preprocessing.py` first."
        )

    print("Loading data for BERT...")
    df = pd.read_csv(CLEANED_DATA_PATH)
    df = df.dropna(subset=["cleaned_text", "sentiment"]).copy()

    texts = df["cleaned_text"].astype(str).tolist()
    labels = df["sentiment"].astype(int).tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels if len(set(labels)) > 1 else None,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Tokenizing data...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

    train_dataset = ReviewDataset(train_encodings, train_labels)
    val_dataset = ReviewDataset(val_encodings, val_labels)

    print("Loading pre-trained model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir=str(ROOT_DIR / "results"),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        weight_decay=0.01,
        logging_dir=str(ROOT_DIR / "logs"),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("Starting BERT training...")
    trainer.train()

    prediction_output = trainer.predict(val_dataset)
    y_pred = np.argmax(prediction_output.predictions, axis=1)
    metrics = evaluate_model("BERT", val_labels, y_pred)

    print("Training complete! Saving model and tokenizer...")
    model.save_pretrained(BERT_MODEL_DIR)
    tokenizer.save_pretrained(BERT_MODEL_DIR)

    print("Success! BERT model has been saved to 'saved_models/bert_sentiment_model'.")
    return model, tokenizer, metrics


if __name__ == "__main__":
    train_bert()
