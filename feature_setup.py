from pathlib import Path
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# Project paths
ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
SAVED_MODELS_DIR = ROOT_DIR / "saved_models"

CLEANED_DATA_PATH = DATA_DIR / "cleaned_amazon_reviews.csv"
TRAIN_TEST_DATA_PATH = DATA_DIR / "train_test_data.pkl"
DATASET_SPLIT_PATH = DATA_DIR / "dataset_split.pkl"
VECTORIZER_PATH = SAVED_MODELS_DIR / "tfidf_vectorizer.pkl"


def build_feature_dataset(
    cleaned_csv_path=CLEANED_DATA_PATH,
    max_features=20_000,
    test_size=0.2,
    random_state=42,
):
    """
    Create one shared stratified train/test split for all models.

    Naive Bayes and SVM use TF-IDF features from cleaned_text.
    BERT uses raw_text from the same train/test split.
    """

    cleaned_csv_path = Path(cleaned_csv_path)

    if not cleaned_csv_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {cleaned_csv_path}. "
            "Run `src/data_preprocessing.py` first."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading cleaned data...")
    df = pd.read_csv(cleaned_csv_path)

    # Ensure all columns required by the three-model pipeline exist.
    required_columns = {"raw_text", "cleaned_text", "sentiment"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Cleaned dataset is missing required column(s): {missing}."
        )

    # Remove incomplete rows and standardize data types.
    df = df.dropna(
        subset=["raw_text", "cleaned_text", "sentiment"]
    ).copy()

    df["raw_text"] = df["raw_text"].astype(str).str.strip()
    df["cleaned_text"] = df["cleaned_text"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].astype(int)

    # Remove empty text rows.
    df = df[
        (df["raw_text"].str.len() > 0)
        & (df["cleaned_text"].str.len() > 0)
    ].copy()

    if df["sentiment"].nunique() < 2:
        raise ValueError(
            "At least two sentiment classes are required for training."
        )

    # ---------------------------------------------------------
    # STEP 1: ONE SHARED TRAIN/TEST SPLIT
    # ---------------------------------------------------------
    print("Creating shared stratified train/test split...")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["sentiment"],
    )

    # Reset indices for cleaner saved datasets.
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Save the original text split for BERT and reproducibility.
    with DATASET_SPLIT_PATH.open("wb") as file_handle:
        pickle.dump(
            {
                "train_df": train_df,
                "test_df": test_df,
            },
            file_handle,
        )

    # ---------------------------------------------------------
    # STEP 2: TF-IDF FEATURES FOR NAIVE BAYES AND SVM
    # ---------------------------------------------------------
    print("Building TF-IDF Vectorizer...")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    # Fit ONLY on training data to prevent data leakage.
    x_train_vec = vectorizer.fit_transform(train_df["cleaned_text"])
    x_test_vec = vectorizer.transform(test_df["cleaned_text"])

    y_train = train_df["sentiment"]
    y_test = test_df["sentiment"]

    # Save TF-IDF vectorizer for Streamlit predictions.
    with VECTORIZER_PATH.open("wb") as file_handle:
        pickle.dump(vectorizer, file_handle)

    # Save vectorized data for Naive Bayes and SVM.
    with TRAIN_TEST_DATA_PATH.open("wb") as file_handle:
        pickle.dump(
            (
                x_train_vec,
                x_test_vec,
                y_train,
                y_test,
            ),
            file_handle,
        )

    # Print split information.
    print("\nShared dataset split:")
    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(train_df)}")
    print(f"Testing samples: {len(test_df)}")

    print("\nTraining class distribution:")
    print(train_df["sentiment"].value_counts().sort_index())

    print("\nTesting class distribution:")
    print(test_df["sentiment"].value_counts().sort_index())

    print(
        "\nSuccess! Shared dataset split, TF-IDF vectorizer, "
        "and TF-IDF train/test data have been saved."
    )

    return {
        "dataset_split_path": str(DATASET_SPLIT_PATH),
        "vectorizer_path": str(VECTORIZER_PATH),
        "train_test_data_path": str(TRAIN_TEST_DATA_PATH),
        "train_size": len(train_df),
        "test_size": len(test_df),
    }


if __name__ == "__main__":
    build_feature_dataset()