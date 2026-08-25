import pickle
import json

from pathlib import Path

try:
    from src.evaluation import evaluate_model
except ImportError:
    from evaluation import evaluate_model


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
SAVED_MODELS_DIR = ROOT_DIR / "saved_models"

DATASET_SPLIT_PATH = DATA_DIR / "dataset_split.pkl"
BERT_MODEL_DIR = SAVED_MODELS_DIR / "bert_sentiment_model"

def get_bert_metrics():
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError("Transformers is required to evaluate the BERT model.") from exc

    if not DATASET_SPLIT_PATH.exists():
        raise FileNotFoundError(
            "Dataset split not found. Run `feature_setup.py` first."
        )
    
    if not BERT_MODEL_DIR.exists():
        raise FileNotFoundError(
            "BERT model folder not found. Run `src/bert.py` first."
        )

    print("Loading test data...")

    with DATASET_SPLIT_PATH.open("rb") as file_handle:
        split_data = pickle.load(file_handle)

    test_df = split_data["test_df"].copy()
    
    required_columns = {"model_text", "sentiment"}
    missing_columns = required_columns - set(test_df.columns)

    if missing_columns:
        raise KeyError(
            f"Test data is missing columns: {missing_columns}. "
            f"Available columns: {test_df.columns.tolist()}"
        )
    
    test_texts = test_df["model_text"].astype(str).tolist()
    y_test = test_df["sentiment"].astype(int).tolist()

    print(f"Test samples: {len(test_texts)}")

    print("Loading BERT pipeline...")
    bert_pipeline = pipeline(
        "text-classification",
        model=str(BERT_MODEL_DIR),
        tokenizer=str(BERT_MODEL_DIR),
    )

    print("Generating BERT predictions...")

    predictions = bert_pipeline(test_texts, truncation=True, max_length=256)

    label_to_id = {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2,
                   "Negative": 0, "Neutral": 1, "Positive": 2}
    y_pred = [label_to_id[prediction["label"]] for prediction in predictions]
    metrics = evaluate_model(
            "BERT",
            y_test,
            y_pred,
        )

    metrics_path = SAVED_MODELS_DIR / "bert_metrics.json"

    with metrics_path.open("w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=4)

    print(f"\nBERT metrics saved to: {metrics_path}")

    return metrics

if __name__ == "__main__":
    get_bert_metrics()
