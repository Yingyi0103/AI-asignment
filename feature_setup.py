import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


CLEANED_DATA_PATH = "data/cleaned_amazon_reviews.csv"
TRAIN_TEST_DATA_PATH = "data/train_test_data.pkl"
VECTORIZER_PATH = "saved_models/tfidf_vectorizer.pkl"


def build_feature_dataset(
    cleaned_csv_path=CLEANED_DATA_PATH,
    max_features=20_000,
    test_size=0.2,
    random_state=42,
):
    """Build TF-IDF features and save the vectorizer plus train/test splits."""
    if not os.path.exists(cleaned_csv_path):
        raise FileNotFoundError(
            f"Cleaned dataset not found: {cleaned_csv_path}. "
            "Run `src/data_preprocessing.py` first."
        )

    os.makedirs("saved_models", exist_ok=True)

    print("Loading cleaned data...")
    df = pd.read_csv(cleaned_csv_path)
    df = df.dropna(subset=["cleaned_text", "sentiment"]).copy()
    df["cleaned_text"] = df["cleaned_text"].astype(str)
    df["sentiment"] = df["sentiment"].astype(int)

    stratify_target = df["sentiment"] if df["sentiment"].nunique() > 1 else None
    x_train_text, x_test_text, y_train, y_test = train_test_split(
        df["cleaned_text"],
        df["sentiment"],
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    print("Building TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    x_train_vec = vectorizer.fit_transform(x_train_text)
    x_test_vec = vectorizer.transform(x_test_text)

    with open(VECTORIZER_PATH, "wb") as file_handle:
        pickle.dump(vectorizer, file_handle)

    with open(TRAIN_TEST_DATA_PATH, "wb") as file_handle:
        pickle.dump((x_train_vec, x_test_vec, y_train, y_test), file_handle)

    print("Success! Vectorizer saved to 'saved_models/' and data splits saved to 'data/'.")
    return {
        "vectorizer_path": VECTORIZER_PATH,
        "train_test_data_path": TRAIN_TEST_DATA_PATH,
        "train_size": int(x_train_vec.shape[0]),
        "test_size": int(x_test_vec.shape[0]),
    }


if __name__ == "__main__":
    build_feature_dataset()
