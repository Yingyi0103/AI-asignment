from pathlib import Path
import pickle

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

try:
    from src.evaluation import evaluate_model
except ImportError:
    from evaluation import evaluate_model


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
SAVED_MODELS_DIR = ROOT_DIR / "saved_models"

DATASET_SPLIT_PATH = DATA_DIR / "dataset_split.pkl"
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
    model_name="bert-base-uncased",
    num_train_epochs=3,
    train_batch_size=8,
    eval_batch_size=16,
):
    """Fine-tune a negative/neutral/positive BERT model on the cleaned dataset."""
    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise ImportError("transformers is required to train the BERT model.") from exc

    if not DATASET_SPLIT_PATH.exists():
        raise FileNotFoundError(
            "Shared dataset not found. Run `feature_setup.py` first."
        )

    print("Loading data for BERT...")
    with DATASET_SPLIT_PATH.open("rb") as file_handle:
        split_data = pickle.load(file_handle)

    train_df = split_data["train_df"].copy()
    test_df = split_data["test_df"].copy()

    print(f"Shared training samples: {len(train_df)}")
    print(f"Shared test samples: {len(test_df)}")

    train_texts = train_df["raw_text"].astype(str).tolist()
    train_labels = train_df["sentiment"].astype(int).tolist()

    test_texts = test_df["raw_text"].astype(str).tolist()
    test_labels = test_df["sentiment"].astype(int).tolist()

    print("Creating BERT training/validation split...")
    (
        bert_train_texts,
        val_texts,
        bert_train_labels,
        val_labels,
    ) = train_test_split(
        train_texts,
        train_labels,
        test_size=0.2,
        random_state=42,
        stratify=train_labels if len(set(train_labels)) > 1 else None,
    )

    print(f"BERT training samples: {len(bert_train_texts)}")
    print(f"BERT validation samples: {len(val_texts)}")
    print(f"Final test samples: {len(test_texts)}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Tokenizing data...")
    train_encodings = tokenizer(
        bert_train_texts, 
        truncation=True, 
        padding=True, 
        max_length=256,
    )

    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=256)

    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding=True,
        max_length=256,
    )

    train_dataset = ReviewDataset(train_encodings, bert_train_labels)
    val_dataset = ReviewDataset(val_encodings, val_labels)
    test_dataset = ReviewDataset(test_encodings, test_labels)

    print("Loading pre-trained model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label={0: "Negative", 1: "Neutral", 2: "Positive"},
        label2id={"Negative": 0, "Neutral": 1, "Positive": 2},
    )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(bert_train_labels),
        y=bert_train_labels,
    )

    class_weights_tensor = torch.tensor(
        class_weights,
        dtype=torch.float,
    )

    training_args = TrainingArguments(
        output_dir=str(ROOT_DIR / "results"),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=2e-5,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
    )

    class WeightedTrainer(Trainer):
        """Use class weights so rare neutral reviews influence BERT training."""

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss_function = torch.nn.CrossEntropyLoss(
                weight=class_weights_tensor.to(outputs.logits.device)
            )
            loss = loss_function(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(prediction):
        predicted_labels = np.argmax(prediction.predictions, axis=1)
        return {"f1_macro": f1_score(prediction.label_ids, predicted_labels, average="macro")}

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting BERT training...")
    trainer.train()

    print("\nEvaluating BERT on the shared final test set...")
    prediction_output = trainer.predict(test_dataset)
    y_pred = np.argmax(prediction_output.predictions, axis=1)
    metrics = evaluate_model("BERT", test_labels, y_pred)

    print("Training complete! Saving model and tokenizer...")
    model.save_pretrained(BERT_MODEL_DIR)
    tokenizer.save_pretrained(BERT_MODEL_DIR)

    print("Success! BERT model has been saved to 'saved_models/bert_sentiment_model'.")
    return model, tokenizer, metrics


if __name__ == "__main__":
    train_bert()
