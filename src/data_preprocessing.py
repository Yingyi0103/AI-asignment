from pathlib import Path
import pandas as pd
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

OLD_DATA = DATA_DIR / "data_amazon.xlsx - Sheet1.csv"
NEW_BINARY_DATA = DATA_DIR / "amazon_new.csv"
NEW_RATING_DATA = DATA_DIR / "Reviews_New.csv"

OUTPUT_FILE = DATA_DIR / "final_sentiment_dataset.csv"

TARGET_PER_CLASS = 25_000
RANDOM_STATE = 42


# ============================================================
# SENTIMENT LABELS
# ============================================================

NEGATIVE = 0
NEUTRAL = 1
POSITIVE = 2


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_review(text):
    """
    Basic cleaning for dataset construction.
    We keep the actual wording because TF-IDF will
    perform the final vectorization later.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def rating_to_sentiment(rating):
    """
    Convert rating into sentiment:

    1-2 -> Negative
    3   -> Neutral candidate
    4-5 -> Positive
    """

    try:
        rating = float(rating)
    except (TypeError, ValueError):
        return None

    if rating <= 2:
        return NEGATIVE

    elif rating == 3:
        return NEUTRAL

    elif rating >= 4:
        return POSITIVE

    return None


# ============================================================
# LOAD OLD AMAZON DATASET
# ============================================================

print("=" * 70)
print("LOADING OLD AMAZON DATASET")
print("=" * 70)

df_old = pd.read_csv(OLD_DATA)

print("Columns:")
print(df_old.columns.tolist())

print(f"Rows: {len(df_old):,}")


required_old = {"Review", "Cons_rating"}

if not required_old.issubset(df_old.columns):
    raise ValueError(
        f"Old dataset must contain {required_old}. "
        f"Found: {df_old.columns.tolist()}"
    )

df_old = df_old[["Review", "Cons_rating"]].copy()

df_old.rename(
    columns={"Review": "text"},
    inplace=True
)

df_old["text"] = df_old["text"].apply(clean_review)

df_old["sentiment"] = df_old["Cons_rating"].apply(
    rating_to_sentiment
)

df_old["source"] = "old_amazon"


# ============================================================
# LOAD NEW BINARY DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING NEW BINARY DATASET")
print("=" * 70)

df_binary = pd.read_csv(NEW_BINARY_DATA)

print("Columns:")
print(df_binary.columns.tolist())

print(f"Rows: {len(df_binary):,}")


required_binary = {"Text", "label"}

if not required_binary.issubset(df_binary.columns):
    raise ValueError(
        f"Binary dataset must contain {required_binary}. "
        f"Found: {df_binary.columns.tolist()}"
    )

df_binary = df_binary[["Text", "label"]].copy()

df_binary.rename(
    columns={"Text": "text"},
    inplace=True
)

df_binary["text"] = df_binary["text"].apply(clean_review)

# Dataset definition:
# 0 = Negative
# 1 = Positive
df_binary["sentiment"] = df_binary["label"].map({
    0: NEGATIVE,
    1: POSITIVE
})

df_binary["source"] = "new_binary"


# ============================================================
# LOAD NEW RATING DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING NEW RATING DATASET")
print("=" * 70)

df_rating = pd.read_csv(
    NEW_RATING_DATA,
    encoding="utf-8",
    usecols=["Score", "Text"]
)

print("Required columns loaded successfully.")
print(f"Rows: {len(df_rating):,}")


# Rename Text -> text
df_rating.rename(
    columns={"Text": "text"},
    inplace=True
)

# Clean review text
df_rating["text"] = df_rating["text"].apply(clean_review)

# Convert:
# 1-2 -> Negative
# 3   -> Neutral candidate
# 4-5 -> Positive
df_rating["sentiment"] = df_rating["Score"].apply(
    rating_to_sentiment
)

df_rating["source"] = "new_rating"


# ============================================================
# DISPLAY SOURCE DISTRIBUTIONS
# ============================================================

print("\n" + "=" * 70)
print("SOURCE SENTIMENT DISTRIBUTIONS")
print("=" * 70)

print("\nOld Amazon:")
print(
    df_old["sentiment"]
    .value_counts()
    .sort_index()
)

print("\nNew Binary:")
print(
    df_binary["sentiment"]
    .value_counts()
    .sort_index()
)

print("\nNew Rating:")
print(
    df_rating["sentiment"]
    .value_counts()
    .sort_index()
)


# ============================================================
# COMBINE DATASETS
# ============================================================

print("\n" + "=" * 70)
print("COMBINING DATASETS")
print("=" * 70)

combined = pd.concat(
    [
        df_old[["text", "sentiment", "source"]],
        df_binary[["text", "sentiment", "source"]],
        df_rating[["text", "sentiment", "source"]],
    ],
    ignore_index=True
)

print(
    f"Total rows before cleaning: "
    f"{len(combined):,}"
)


# ============================================================
# REMOVE INVALID REVIEWS
# ============================================================

print("\nRemoving empty reviews...")

combined = combined[
    combined["text"].str.len() >= 10
].copy()

print(
    f"Rows after removing very short reviews: "
    f"{len(combined):,}"
)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

print("\nRemoving duplicate reviews...")

before_duplicates = len(combined)

combined["text_normalized"] = (
    combined["text"]
    .str.lower()
    .str.strip()
)

combined = combined.drop_duplicates(
    subset=["text_normalized"]
).copy()

duplicates_removed = (
    before_duplicates - len(combined)
)

print(
    f"Duplicates removed: "
    f"{duplicates_removed:,}"
)

print(
    f"Rows remaining: "
    f"{len(combined):,}"
)


# ============================================================
# IMPORTANT:
# SCORE-3 REVIEWS ARE ONLY NEUTRAL CANDIDATES
# ============================================================

print("\n" + "=" * 70)
print("FILTERING NEUTRAL CANDIDATES")
print("=" * 70)

# Only use 3-star reviews from the NEW rating dataset
# as Neutral candidates.
#
# We deliberately DO NOT use all 3-star reviews from
# the old dataset automatically.

neutral_candidates = combined[
    (combined["source"] == "new_rating")
    & (combined["sentiment"] == NEUTRAL)
].copy()

print(
    f"Initial Neutral candidates: "
    f"{len(neutral_candidates):,}"
)


# ============================================================
# REMOVE OBVIOUSLY EXTREME SENTIMENT FROM NEUTRAL
# ============================================================

strong_positive_words = [
    "excellent",
    "amazing",
    "fantastic",
    "wonderful",
    "perfect",
    "awesome",
    "outstanding",
    "love it",
    "love this",
    "highly recommend",
    "best product",
    "great product",
]

strong_negative_words = [
    "terrible",
    "horrible",
    "awful",
    "worst",
    "useless",
    "waste of money",
    "do not buy",
    "don't buy",
    "never buy",
    "highly disappointed",
    "completely disappointed",
]


def is_extreme_neutral_candidate(text):

    text_lower = text.lower()

    positive_hits = sum(
        phrase in text_lower
        for phrase in strong_positive_words
    )

    negative_hits = sum(
        phrase in text_lower
        for phrase in strong_negative_words
    )

    # Remove reviews containing strong sentiment signals.
    if positive_hits > 0:
        return False

    if negative_hits > 0:
        return False

    return True


neutral_candidates = neutral_candidates[
    neutral_candidates["text"].apply(
        is_extreme_neutral_candidate
    )
].copy()

print(
    f"Neutral candidates after basic filtering: "
    f"{len(neutral_candidates):,}"
)


# ============================================================
# CREATE NEGATIVE CANDIDATES
# ============================================================

negative_candidates = combined[
    combined["sentiment"] == NEGATIVE
].copy()

positive_candidates = combined[
    combined["sentiment"] == POSITIVE
].copy()


print("\nAvailable candidates:")

print(
    f"Negative: {len(negative_candidates):,}"
)

print(
    f"Neutral:  {len(neutral_candidates):,}"
)

print(
    f"Positive: {len(positive_candidates):,}"
)


# ============================================================
# CHECK WHETHER ENOUGH DATA EXISTS
# ============================================================

if len(negative_candidates) < TARGET_PER_CLASS:
    raise ValueError(
        "Not enough Negative reviews."
    )

if len(neutral_candidates) < TARGET_PER_CLASS:
    raise ValueError(
        "Not enough Neutral reviews after filtering."
    )

if len(positive_candidates) < TARGET_PER_CLASS:
    raise ValueError(
        "Not enough Positive reviews."
    )


# ============================================================
# SAMPLE 25,000 FROM EACH CLASS
# ============================================================

print("\n" + "=" * 70)
print("BALANCING DATASET")
print("=" * 70)

negative_final = negative_candidates.sample(
    n=TARGET_PER_CLASS,
    random_state=RANDOM_STATE
)

neutral_final = neutral_candidates.sample(
    n=TARGET_PER_CLASS,
    random_state=RANDOM_STATE
)

positive_final = positive_candidates.sample(
    n=TARGET_PER_CLASS,
    random_state=RANDOM_STATE
)


# ============================================================
# COMBINE FINAL DATASET
# ============================================================

final_df = pd.concat(
    [
        negative_final,
        neutral_final,
        positive_final
    ],
    ignore_index=True
)


# ============================================================
# FINAL CLEANUP
# ============================================================

final_df = final_df[
    ["text", "sentiment", "source"]
].copy()

final_df = final_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET")
print("=" * 70)

print(
    f"Total reviews: {len(final_df):,}"
)

print("\nClass distribution:")

class_names = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

counts = final_df["sentiment"].value_counts().sort_index()

for label, count in counts.items():

    print(
        f"{class_names[label]}: "
        f"{count:,}"
    )


print("\nSource distribution:")

print(
    final_df["source"].value_counts()
)


# ============================================================
# SAVE
# ============================================================

final_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print(
    f"Final dataset saved to:\n"
    f"{OUTPUT_FILE}"
)