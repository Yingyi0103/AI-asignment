import os
import re
import unicodedata

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

try:
    from nltk.stem import PorterStemmer, WordNetLemmatizer
except ImportError:
    PorterStemmer = None
    WordNetLemmatizer = None


RAW_DATA_PATH = "data/test.ft.txt"
CLEANED_DATA_PATH = "data/cleaned_amazon_reviews.csv"

FASTTEXT_PATTERN = re.compile(r"^__label__(?P<label>\S+)\s+(?P<text>.+)$")
NON_LETTER_PATTERN = re.compile(r"[^a-zA-Z\s']")
WHITESPACE_PATTERN = re.compile(r"\s+")
STOP_WORDS = set(ENGLISH_STOP_WORDS)
STEMMER = PorterStemmer() if PorterStemmer is not None else None
LEMMATIZER = WordNetLemmatizer() if WordNetLemmatizer is not None else None


def load_fasttext_dataset(file_path=RAW_DATA_PATH):
    """Load the local fastText-style review file from `data/test.ft.txt`."""
    rows = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file_handle:
        for line in file_handle:
            match = FASTTEXT_PATTERN.match(line.strip())
            if not match:
                continue

            raw_label = match.group("label").strip()
            sentiment = 0 if raw_label == "1" else 1 if raw_label == "2" else None
            review_text = match.group("text").strip()

            if sentiment is None or not review_text:
                continue

            rows.append({"review_text": review_text, "sentiment": sentiment})

    if not rows:
        raise ValueError(f"No valid fastText records were found in {file_path}.")

    return pd.DataFrame(rows)


def clean_text(text):
    """Clean text into the same format used for training and app prediction."""
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = NON_LETTER_PATTERN.sub(" ", text.lower())
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    cleaned_tokens = []
    for token in text.split():
        if len(token) <= 1 or token in STOP_WORDS:
            continue

        if LEMMATIZER is not None:
            try:
                token = LEMMATIZER.lemmatize(token)
            except LookupError:
                pass

        if STEMMER is not None:
            token = STEMMER.stem(token)

        cleaned_tokens.append(token)

    return " ".join(cleaned_tokens)


def preprocess_dataset(input_path=RAW_DATA_PATH, output_path=CLEANED_DATA_PATH, balance=True):
    """Load `test.ft.txt`, clean it, and save a new cleaned CSV file."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Raw dataset not found: {input_path}")

    print("Loading dataset...")
    df = load_fasttext_dataset(input_path)
    df = df.dropna(subset=["review_text", "sentiment"]).copy()
    df["review_text"] = df["review_text"].astype(str)
    df["sentiment"] = df["sentiment"].astype(int)

    if balance and df["sentiment"].nunique() == 2:
        print("Balancing dataset to 50/50 (undersampling)...")
        positive_df = df[df["sentiment"] == 1]
        negative_df = df[df["sentiment"] == 0]
        min_class_size = min(len(positive_df), len(negative_df))

        positive_df = positive_df.sample(n=min_class_size, random_state=42)
        negative_df = negative_df.sample(n=min_class_size, random_state=42)
        df = pd.concat([positive_df, negative_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    print("Cleaning text data. Please wait...")
    df["cleaned_text"] = df["review_text"].apply(clean_text)
    final_df = df[["review_text", "cleaned_text", "sentiment"]]
    final_df = final_df[final_df["cleaned_text"].str.len() > 0].reset_index(drop=True)
    final_df.to_csv(output_path, index=False)

    print(f"Success! {len(final_df)} cleaned reviews saved as '{output_path}'.")
    return final_df


if __name__ == "__main__":
    preprocess_dataset()
