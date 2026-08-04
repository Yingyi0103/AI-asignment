import bz2
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

try:
    from nltk.stem import PorterStemmer, WordNetLemmatizer
except ImportError:
    PorterStemmer = None
    WordNetLemmatizer = None


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_DIR / "data" / "train.ft.txt.bz2"
RATINGS_DATA_PATH = PROJECT_DIR / "data" / "data_amazon.xlsx - Sheet1.csv"
CLEANED_DATA_PATH = PROJECT_DIR / "data" / "cleaned_amazon_reviews.csv"
TARGET_SAMPLES_PER_CLASS = 25_000
RANDOM_STATE = 42

# A common three-class rating convention.  The fastText source has no neutral
# class; it is supplied by the ratings dataset (three-star reviews).
NEGATIVE = 0
NEUTRAL = 1
POSITIVE = 2

FASTTEXT_PATTERN = re.compile(r"^__label__(?P<label>\S+)\s+(?P<text>.+)$")
NON_LETTER_PATTERN = re.compile(r"[^a-zA-Z\s']")
WHITESPACE_PATTERN = re.compile(r"\s+")
STOP_WORDS = set(ENGLISH_STOP_WORDS)
STEMMER = PorterStemmer() if PorterStemmer is not None else None
LEMMATIZER = WordNetLemmatizer() if WordNetLemmatizer is not None else None
WORDNET_AVAILABLE = False

if LEMMATIZER is not None:
    try:
        LEMMATIZER.lemmatize("reviews")
        WORDNET_AVAILABLE = True
    except LookupError:
        # WordNet is optional; avoid raising an exception for every token.
        pass


def load_fasttext_dataset(file_path=RAW_DATA_PATH, samples_per_class=TARGET_SAMPLES_PER_CLASS):
    """Load a balanced negative/positive sample from the fastText dataset."""
    rows = []
    class_counts = {NEGATIVE: 0, POSITIVE: 0}

    with bz2.open(file_path, "rt", encoding="utf-8", errors="ignore") as file_handle:
        for line in file_handle:
            match = FASTTEXT_PATTERN.match(line.strip())
            if not match:
                continue

            raw_label = match.group("label").strip()
            sentiment = NEGATIVE if raw_label == "1" else POSITIVE if raw_label == "2" else None
            raw_text = match.group("text").strip()

            if sentiment is None or not raw_text or class_counts[sentiment] >= samples_per_class:
                continue

            rows.append({"raw_text": raw_text, "sentiment": sentiment})
            class_counts[sentiment] += 1

            if all(count >= samples_per_class for count in class_counts.values()):
                break

    if not rows:
        raise ValueError(f"No valid fastText records were found in {file_path}.")

    return pd.DataFrame(rows)


def rating_to_sentiment(rating):
    """Convert a one-to-five star rating to negative, neutral, or positive."""
    try:
        rating = float(rating)
    except (TypeError, ValueError):
        return None

    if rating <= 2:
        return NEGATIVE
    if rating == 3:
        return NEUTRAL
    if rating >= 4:
        return POSITIVE
    return None


def load_ratings_dataset(file_path=RATINGS_DATA_PATH):
    """Load the CSV reviews and derive three sentiment classes from star ratings."""
    ratings_df = pd.read_csv(file_path)
    required_columns = {"Review", "Cons_rating"}
    missing_columns = required_columns.difference(ratings_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Ratings dataset is missing required column(s): {missing}.")

    reviews = ratings_df["Review"].fillna("").astype(str).str.strip()
    titles = ratings_df.get("Title", pd.Series("", index=ratings_df.index)).fillna("").astype(str).str.strip()
    # Titles add useful context, but do not add an empty title or duplicate it.
    raw_text = (titles + ". " + reviews).str.strip(". ").where(titles.ne(""), reviews)
    result = pd.DataFrame(
        {
            "raw_text": raw_text,
            "sentiment": ratings_df["Cons_rating"].apply(rating_to_sentiment),
        }
    )
    return result.dropna(subset=["raw_text", "sentiment"])


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

        if WORDNET_AVAILABLE:
            token = LEMMATIZER.lemmatize(token)

        if STEMMER is not None:
            token = STEMMER.stem(token)

        cleaned_tokens.append(token)

    return " ".join(cleaned_tokens)


def preprocess_dataset(
    fasttext_path=RAW_DATA_PATH,
    ratings_path=RATINGS_DATA_PATH,
    output_path=CLEANED_DATA_PATH,
):
    """Combine both raw datasets, clean them, and save one three-class CSV."""
    fasttext_path = Path(fasttext_path)
    ratings_path = Path(ratings_path)
    output_path = Path(output_path)

    if not fasttext_path.exists():
        raise FileNotFoundError(f"fastText dataset not found: {fasttext_path}")
    if not ratings_path.exists():
        raise FileNotFoundError(f"Ratings dataset not found: {ratings_path}")

    print("Loading fastText and ratings datasets...")
    df = pd.concat(
        [load_fasttext_dataset(fasttext_path), load_ratings_dataset(ratings_path)],
        ignore_index=True,
    )
    df = df.dropna(subset=["raw_text", "sentiment"]).copy()
    df["raw_text"] = df["raw_text"].astype(str)
    df["sentiment"] = df["sentiment"].astype(int)

    print("Cleaning text data. Please wait...")
    df["cleaned_text"] = df["raw_text"].apply(clean_text)
    final_df = df[["cleaned_text", "sentiment"]]
    final_df = final_df[final_df["cleaned_text"].str.len() > 0]
    final_df = final_df.drop_duplicates(subset=["cleaned_text", "sentiment"])
    final_df = final_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        final_df.to_csv(output_path, index=False)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot write '{output_path}'. Close the file in Excel or another program, then run this script again."
        ) from exc

    class_names = {NEGATIVE: "negative", NEUTRAL: "neutral", POSITIVE: "positive"}
    counts = final_df["sentiment"].value_counts().sort_index()
    count_summary = ", ".join(
        f"{class_names[label]}={counts.get(label, 0)}" for label in class_names
    )
    print(f"Success! {len(final_df)} cleaned reviews saved as '{output_path}' ({count_summary}).")
    return final_df


if __name__ == "__main__":
    preprocess_dataset()
