from pathlib import Path
import pickle

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

try:
    from src.evaluation import evaluate_model
except ImportError:
    from evaluation import evaluate_model


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

    print("Selecting the best SVM regularisation setting with cross-validation...")
    search = GridSearchCV(
        estimator=SVC(kernel="linear", class_weight="balanced", random_state=42),
        param_grid={"C": [0.1, 1, 3]},
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
    )
    search.fit(x_train_vec, y_train)

    # Fit the selected model with calibrated probabilities for the Streamlit app.
    print(f"Training final SVM model with C={search.best_params_['C']}...")
    model = SVC(
        kernel="linear",
        C=search.best_params_["C"],
        class_weight="balanced",
        probability=True,
        random_state=42,
    )
    model.fit(x_train_vec, y_train)

    y_pred = model.predict(x_test_vec)
    metrics = evaluate_model("SVM", y_test, y_pred)

    with MODEL_PATH.open("wb") as file_handle:
        pickle.dump(model, file_handle)

    print("\nSuccess! SVM model has been saved to 'saved_models/'.")
    return model, metrics


if __name__ == "__main__":
    train_svm()
