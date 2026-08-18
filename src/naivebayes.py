from pathlib import Path
import pickle

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight

try:
    from src.evaluation import evaluate_model
except ImportError:
    from evaluation import evaluate_model


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
SAVED_MODELS_DIR = ROOT_DIR / "saved_models"
TRAIN_TEST_DATA_PATH = DATA_DIR / "train_test_data.pkl"
MODEL_PATH = SAVED_MODELS_DIR / "naive_bayes_model.pkl"
VECTORIZER_PATH = SAVED_MODELS_DIR / "tfidf_vectorizer.pkl"

SAVED_MODELS_DIR.mkdir(exist_ok=True)


def train_naive_bayes():
    """Train the Naive Bayes model on the prepared TF-IDF features."""
    if not TRAIN_TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            "Training split file not found. Run `feature_setup.py` first."
        )

    print("Loading train/test data splits...")
    with TRAIN_TEST_DATA_PATH.open("rb") as file_handle:
        x_train_vec, x_test_vec, y_train, y_test = pickle.load(file_handle)

    # Balanced weights stop the majority positive class from dominating training.
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    print("Selecting the best Naive Bayes smoothing setting with cross-validation...")
    search = GridSearchCV(
        estimator=MultinomialNB(),
        param_grid={"alpha": [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0]},
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
    )
    search.fit(x_train_vec, y_train, sample_weight=sample_weights)

    print(f"Training final Naive Bayes model with alpha={search.best_params_['alpha']}...")
    model = MultinomialNB(alpha=search.best_params_["alpha"])
    model.fit(x_train_vec, y_train, sample_weight=sample_weights)

    y_pred = model.predict(x_test_vec)
    metrics = evaluate_model("Naive Bayes", y_test, y_pred)

    with MODEL_PATH.open("wb") as file_handle:
        pickle.dump(model, file_handle)

    print("\nSuccess! Naive Bayes model has been saved to 'saved_models/'.")
    return model, metrics

if __name__ == "__main__":
    train_naive_bayes()
