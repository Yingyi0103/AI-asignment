import os

import pandas as pd
from sklearn.model_selection import train_test_split

from evaluation import evaluate_model


CLEANED_DATA_PATH = "data/cleaned_amazon_reviews.csv"
BERT_MODEL_DIR = "saved_models/bert_sentiment_model"


def get_bert_metrics():
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError("transformers is required to evaluate the BERT model.") from exc

    if not os.path.exists(CLEANED_DATA_PATH):
        raise FileNotFoundError(
            "Cleaned dataset not found. Run `src/data_preprocessing.py` first."
        )
    if not os.path.exists(BERT_MODEL_DIR):
        raise FileNotFoundError(
            "BERT model folder not found. Run `src/bert.py` first."
        )

    print("Loading test data...")
    df = pd.read_csv(CLEANED_DATA_PATH)
    df = df.dropna(subset=["cleaned_text", "sentiment"]).copy()

    _, test_texts, _, y_test = train_test_split(
        df["cleaned_text"].astype(str).tolist(),
        df["sentiment"].astype(int).tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df["sentiment"] if df["sentiment"].nunique() > 1 else None,
    )

    print("Loading BERT pipeline...")
    bert_pipeline = pipeline(
        "text-classification",
        model=BERT_MODEL_DIR,
        tokenizer=BERT_MODEL_DIR,
    )

    predictions = bert_pipeline(test_texts, truncation=True, max_length=512)
    y_pred = [1 if prediction["label"] == "LABEL_1" else 0 for prediction in predictions]
    return evaluate_model("BERT", y_test, y_pred)


if __name__ == "__main__":
    get_bert_metrics()
