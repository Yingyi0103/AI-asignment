from pathlib import Path
import pickle

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight

try:
    from src.evaluation import evaluate_model
except ImportError:
    from evaluation import evaluate_model

# Paths
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
SAVED_MODELS_DIR = ROOT_DIR / "saved_models"

TRAIN_TEST_DATA_PATH = DATA_DIR / "train_test_data.pkl"
MODEL_PATH = SAVED_MODELS_DIR / "svm_model.pkl"

SAVED_MODELS_DIR.mkdir(exist_ok=True)

def train_svm():
    """Train the SVM model on the prepared TF-IDF features."""
    if not TRAIN_TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            "Training split file not found. Run `feature_setup.py` first."
        )

    print("Loading train/test data splits...")
    with TRAIN_TEST_DATA_PATH.open("rb") as file_handle:
        x_train_vec, x_test_vec, y_train, y_test = pickle.load(file_handle)

    print(f"Training samples: {x_train_vec.shape[0]:,}")
    print(f"Testing samples : {x_test_vec.shape[0]:,}")
    print(f"TF-IDF features : {x_train_vec.shape[1]:,}")

    print("\nTraining class distribution:")
    print(y_train.value_counts().sort_index())

    # Balance Sample Weight
    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )

    print("Selecting the best SVM regularisation setting with cross-validation...")
    search = GridSearchCV(
        estimator=LinearSVC(class_weight="balanced", max_iter=5000),
        param_grid={"C": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]},
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    search.fit(x_train_vec, y_train, sample_weight=sample_weights)

    print("\nBest parameters:")
    print(search.best_params_)

    print(
        f"Best cross-validation Macro F1: "
        f"{search.best_score_:.4f}"
    )

    # Fit the selected model with calibrated probabilities for the Streamlit app.
    best_c = search.best_params_["C"]
    print(f"Training final SVM model with C={best_c}...")
    model = LinearSVC(
        C=best_c,
        class_weight="balanced",
        max_iter=5000
    )
    model.fit(x_train_vec, y_train, sample_weight=sample_weights)

    print("\nEvaluating SVM model...")

    y_pred = model.predict(x_test_vec)
    metrics = evaluate_model("SVM", y_test, y_pred)

    with MODEL_PATH.open("wb") as file_handle:
        pickle.dump(model, file_handle)

    print("\nSuccess! SVM model has been saved to 'saved_models/'.")
    return model, metrics


if __name__ == "__main__":
    train_svm()
