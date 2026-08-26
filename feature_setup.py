from pathlib import Path
import pickle
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Project paths
ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
SAVED_MODELS_DIR = ROOT_DIR / "saved_models"

FINAL_DATASET_PATH = DATA_DIR / "final_sentiment_dataset.csv"

TRAIN_TEST_DATA_PATH = DATA_DIR / "train_test_data.pkl"
DATASET_SPLIT_PATH = DATA_DIR / "dataset_split.pkl"
VECTORIZER_PATH = SAVED_MODELS_DIR / "tfidf_vectorizer.pkl"

SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Text Cleaning
def clean_for_tfidf(text):
    """
    Light text cleaning for TF-IDF.

    Important:
    - Keep negation words such as 'not'
    - Keep sentiment words
    - Remove URLs and HTML
    - Convert text to lowercase
    - Keep natural wording
    """

    text = str(text).lower()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Preserve common contractions involving negation
    text = re.sub(r"\bcan't\b", "cannot", text)
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"\bdon't\b", "do not", text)
    text = re.sub(r"\bdoesn't\b", "does not", text)
    text = re.sub(r"\bdidn't\b", "did not", text)
    text = re.sub(r"\bisn't\b", "is not", text)
    text = re.sub(r"\baren't\b", "are not", text)
    text = re.sub(r"\bwasn't\b", "was not", text)
    text = re.sub(r"\bweren't\b", "were not", text)
    text = re.sub(r"\bwouldn't\b", "would not", text)
    text = re.sub(r"\bcouldn't\b", "could not", text)
    text = re.sub(r"\bshouldn't\b", "should not", text)
    text = re.sub(r"\bhasn't\b", "has not", text)
    text = re.sub(r"\bhaven't\b", "have not", text)
    text = re.sub(r"\bhadn't\b", "had not", text)

    # Keep letters and apostrophes only
    text = re.sub(r"[^a-zA-Z\s']", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

# Build Features
def build_feature_dataset(
    dataset_path=FINAL_DATASET_PATH,
    max_features=50_000,
    test_size=0.2,
    random_state=42,
):
    """
    Create one shared stratified train/test split for all models.

    Naive Bayes and SVM use TF-IDF features from cleaned_text.
    BERT uses raw_text from the same train/test split.
    """

    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. "
            "Run `src/data_preprocessing.py` first."
        )

    print("Loading cleaned data...")
    df = pd.read_csv(dataset_path)

    # Ensure all columns required by the three-model pipeline exist.
    required_columns = {"text", "sentiment"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Misisng Column(s): {missing_columns}."
        )

    # Remove incomplete rows and standardize data types.
    print("\nCLeaning Text...")

    df = df.dropna(
        subset=["text", "sentiment"]
    ).copy()

    df["text"] = df["text"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].astype(int)

    # Remove empty text rows.
    df = df[df["text"].str.len() > 0].copy()

    # Light text cleaning
    df["model_text"] = df["text"].apply(clean_for_tfidf)

    # Remove empty cleaned reviews
    df = df[df["model_text"].str.len() > 0].copy()

    # Remove exact duplicate review text
    before_duplicates = len(df)

    df = df.drop_duplicates(subset=["model_text"]).reset_index(drop=True)

    duplicates_removed = before_duplicates - len(df)

    print(f"Duplicate reviews removed: {duplicates_removed:,}")
    print(f"Final rows available: {len(df):,}")

    # ---------------------------------------------------------
    # CHECK CLASS DISTRIBUTION
    # ---------------------------------------------------------
    print("\nClass distribution:")

    class_counts = df["sentiment"].value_counts().sort_index()

    print(class_counts)

    # ---------------------------------------------------------
    # STEP 1: SHARED TRAIN/TEST SPLIT
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
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
        stop_words=None,
    )

    print("Fitting TF-IDF on training data only...")

    # Fit ONLY on training data to prevent data leakage.
    x_train_vec = vectorizer.fit_transform(train_df["model_text"])
    x_test_vec = vectorizer.transform(test_df["model_text"])

    y_train = train_df["sentiment"]
    y_test = test_df["sentiment"]

    print(f"Training TF-IDF shape: {x_train_vec.shape}")
    print(f"Testing TF-IDF shape:  {x_test_vec.shape}")

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
    print(f"Total samples    : {len(df):,}")
    print(f"Training samples : {len(train_df):,}")
    print(f"Testing samples  : {len(test_df):,}")
    print(f"TF-IDF features  : {x_train_vec.shape[1]:,}")

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
        "features": x_train_vec.shape[1],
    }


if __name__ == "__main__":
    build_feature_dataset()