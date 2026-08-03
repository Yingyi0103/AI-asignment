import json
import os
import re

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def _slugify_model_name(model_name):
    return re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")


def evaluate_model(model_name, y_true, y_pred, save_metrics=True):
    """Print and optionally save the evaluation metrics for a binary model."""
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "Confusion Matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    print(f"\n--- {model_name} Evaluation ---")
    print(f"Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall:    {metrics['Recall']:.4f}")
    print(f"F1-Score:  {metrics['F1-Score']:.4f}")
    print("Confusion Matrix:\n", metrics["Confusion Matrix"])
    print("-" * 30)

    if save_metrics:
        os.makedirs("saved_models", exist_ok=True)
        metrics_path = f"saved_models/{_slugify_model_name(model_name)}_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as file_handle:
            json.dump(metrics, file_handle, indent=2)

    return metrics
