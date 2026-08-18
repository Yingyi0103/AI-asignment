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
STEMMER = PorterStemmer() if PorterStemmer is not None else None
LEMMATIZER = WordNetLemmatizer() if WordNetLemmatizer is not None else None
WORDNET_AVAILABLE = False

STOP_WORDS = set(ENGLISH_STOP_WORDS)

# Keep important negation words for sentiment analysis.
NEGATION_WORDS = {
    "no", "nor", "not", "never", "none", "nothing", "neither", "nowhere", "cannot", "cant",
    "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't", "won't", "wouldn't",
    "couldn't", "shouldn't", "haven't", "hasn't", "hadn't",
}

STOP_WORDS = STOP_WORDS - NEGATION_WORDS

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

    result = pd.DataFrame(
        {
            "raw_text": reviews,
            "sentiment": ratings_df["Cons_rating"].apply(rating_to_sentiment),
        }
    )

    # Remove rows with empty reviews or invalid sentiment labels.
    result = result[result["raw_text"].str.len() > 0]
    result = result.dropna(subset=["sentiment"])

    return result


def clean_text(text):
    """Clean review text while preserving sentiment information."""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower()

    # Remove URLs.
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Preserve important negation meaning.
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

    # Remove non-letter characters.
    text = NON_LETTER_PATTERN.sub(" ", text)

    # Normalize whitespace.
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    cleaned_tokens = []

    for token in text.split():
        if len(token) <= 1:
            continue

        if token in STOP_WORDS and token not in NEGATION_WORDS:
            continue

        # IMPORTANT:
        # Do NOT stem or lemmatize sentiment words.
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
    # Keep original wording as well: BERT learns better from natural sentences
    # than from stemmed, stop-word-removed text used by the classical models.
    final_df = df[["raw_text", "cleaned_text", "sentiment"]]
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
