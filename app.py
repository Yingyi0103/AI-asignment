"""Streamlit interface for product-review sentiment analysis."""

import json
import pickle
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

try:
    from nltk.stem import PorterStemmer, WordNetLemmatizer
except ImportError:
    PorterStemmer = None
    WordNetLemmatizer = None


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
SAVED_MODELS_DIR = ROOT_DIR / "saved_models"
CLEANED_DATA_PATH = DATA_DIR / "cleaned_amazon_reviews.csv"
TRAIN_TEST_DATA_PATH = DATA_DIR / "train_test_data.pkl"
VECTORIZER_PATH = SAVED_MODELS_DIR / "tfidf_vectorizer.pkl"
MODEL_PATHS = {
    "Naive Bayes": SAVED_MODELS_DIR / "naive_bayes_model.pkl",
    "SVM": SAVED_MODELS_DIR / "svm_model.pkl",
}
BERT_MODEL_DIR = SAVED_MODELS_DIR / "bert_sentiment_model"
SENTIMENT_LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}

STOP_WORDS = set(ENGLISH_STOP_WORDS)
STEMMER = PorterStemmer() if PorterStemmer is not None else None
LEMMATIZER = WordNetLemmatizer() if WordNetLemmatizer is not None else None
WORDNET_AVAILABLE = False

if LEMMATIZER is not None:
    try:
        LEMMATIZER.lemmatize("reviews")
        WORDNET_AVAILABLE = True
    except LookupError:
        pass

st.set_page_config(page_title="Review Sentiment Analyzer", page_icon="💬", layout="wide")


def clean_input(text: str) -> str:
    """Apply the same text preparation used for the TF-IDF classifiers."""
    text = re.sub(r"https?://\S+|www\.\S+", " ", str(text))
    text = re.sub(r"[^a-zA-Z\s']", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()

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


@st.cache_resource
def load_models():
    """Load available trained models once per Streamlit server session."""
    vectorizer = None
    models = {"Naive Bayes": None, "SVM": None, "BERT": None}

    if VECTORIZER_PATH.exists():
        with VECTORIZER_PATH.open("rb") as file_handle:
            vectorizer = pickle.load(file_handle)

    for model_name, path in MODEL_PATHS.items():
        if path.exists():
            with path.open("rb") as file_handle:
                models[model_name] = pickle.load(file_handle)

    if BERT_MODEL_DIR.exists():
        try:
            from transformers import pipeline

            models["BERT"] = pipeline(
                "text-classification", model=str(BERT_MODEL_DIR), tokenizer=str(BERT_MODEL_DIR)
            )
        except Exception:
            # A clear model-specific message is shown when the user selects BERT.
            models["BERT"] = None
    return vectorizer, models


@st.cache_data
def load_cleaned_dataset():
    return pd.read_csv(CLEANED_DATA_PATH) if CLEANED_DATA_PATH.exists() else None


@st.cache_data
def most_common_words(texts: tuple[str, ...], limit: int = 15) -> pd.DataFrame:
    """Create a frequency table from the already-cleaned review text."""
    word_counts = Counter(word for text in texts for word in text.split())
    return pd.DataFrame(word_counts.most_common(limit), columns=["Word", "Frequency"])


@st.cache_data
def load_saved_metrics():
    metrics = {}
    for model_key in ("naive_bayes", "svm", "bert"):
        path = SAVED_MODELS_DIR / f"{model_key}_metrics.json"
        if path.exists():
            with path.open(encoding="utf-8") as file_handle:
                metrics[model_key] = json.load(file_handle)
    return metrics


@st.cache_data
def compute_classical_metrics():
    if not TRAIN_TEST_DATA_PATH.exists():
        return {}
    _, models = load_models()
    with TRAIN_TEST_DATA_PATH.open("rb") as file_handle:
        _, x_test, _, y_test = pickle.load(file_handle)

    metrics = {}
    for display_name, model_key in (("Naive Bayes", "naive_bayes"), ("SVM", "svm")):
        model = models[display_name]
        if model is not None:
            predictions = model.predict(x_test)
            metrics[model_key] = {
                "Accuracy": float(accuracy_score(y_test, predictions)),
                "Precision": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
                "Recall": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
                "F1-Score": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
            }
    return metrics


def analyse_review(review: str, model_name: str):
    """Return a sentiment label, confidence, and prepared text for one review."""
    vectorizer, models = load_models()

    if model_name == "BERT":
        classifier = models["BERT"]
        if classifier is None:
            raise FileNotFoundError(
                "BERT is unavailable. Train it with `python src/bert.py` first."
            )
        if classifier.model.config.num_labels != 3:
            raise ValueError("This BERT model uses the old two-class dataset. Retrain it for neutral reviews.")
        result = classifier(review, truncation=True, max_length=512)[0]
        label = result["label"].title()
        return (label, float(result["score"]), clean_input(review))

    if vectorizer is None:
        raise FileNotFoundError("TF-IDF vectorizer not found. Run `python feature_setup.py` first.")
    model = models[model_name]
    if model is None:
        script = "naivebayes" if model_name == "Naive Bayes" else "svm"
        raise FileNotFoundError(f"{model_name} is unavailable. Train it with `python src/{script}.py` first.")

    prepared_text = clean_input(review)
    features = vectorizer.transform([prepared_text])
    if set(model.classes_) != set(SENTIMENT_LABELS):
        raise ValueError(f"{model_name} uses the old two-class dataset. Retrain it for neutral reviews.")
    predicted_label = int(model.predict(features)[0])
    probability_index = list(model.classes_).index(predicted_label)
    confidence = float(model.predict_proba(features)[0][probability_index])
    return (SENTIMENT_LABELS[predicted_label], confidence, prepared_text)


def categorise_issue(review: str) -> str:
    """Assign the review to the issue category with the most matching keywords."""
    text = review.lower()
    category_keywords = {
        "Quality issue": (
            "quality", "broken", "defective", "damaged", "faulty", "poor",
            "doesn't work", "does not work", "stopped working", "durability", "scratch",
        ),
        "Delivery issue": (
            "delivery", "deliver", "shipping", "shipped", "arrived", "arrival", "courier",
            "package", "parcel", "late", "delay", "tracking",
        ),
        "Price": (
            "price", "cost", "expensive", "cheap", "overpriced", "discount", "value for money",
        ),
        "Seller service": (
            "seller", "customer service", "customer support", "support", "response", "respond",
            "return", "replacement", "communication", "contact",
        ),
    }
    scores = {
        category: sum(text.count(keyword) for keyword in keywords)
        for category, keywords in category_keywords.items()
    }
    category, score = max(scores.items(), key=lambda item: item[1])
    return category if score else "Other"


@st.cache_data
def issue_category_summary(reviews: tuple[str, ...], sentiments: tuple[int, ...]) -> pd.DataFrame:
    """Count negative, neutral, and positive reviews for every issue category."""
    categories = ("Quality issue", "Delivery issue", "Price", "Seller service", "Other")
    counts = {
        category: {"Negative reviews": 0, "Neutral reviews": 0, "Positive reviews": 0}
        for category in categories
    }

    for review, sentiment in zip(reviews, sentiments):
        sentiment_column = f"{SENTIMENT_LABELS[int(sentiment)]} reviews"
        counts[categorise_issue(review)][sentiment_column] += 1

    return pd.DataFrame(
        [
            {"Issue category": category, **category_counts}
            for category, category_counts in counts.items()
        ]
    )


def add_history_item(
    review: str, model: str, sentiment: str, confidence: float, issue_category: str
) -> None:
    st.session_state.history.insert(
        0,
        {
            "Time": datetime.now().strftime("%d %b %Y, %I:%M %p"),
            "Model": model,
            "Sentiment": sentiment,
            "Issue category": issue_category,
            "Confidence": f"{confidence:.2%}",
            "Review": review,
        },
    )


if "history" not in st.session_state:
    st.session_state.history = []

st.title("Product Review Sentiment Analyzer")
st.caption("Classify product reviews with Naive Bayes, SVM, or BERT.")

analyzer_tab, dataset_tab, metrics_tab = st.tabs(["Analyzer", "Dataset", "Model metrics"])

with analyzer_tab:
    st.subheader("Analyze a review")
    st.write("Enter a review, choose a trained AI model, then confirm the analysis.")
    review = st.text_area(
        "Product review",
        placeholder="Example: The product arrived quickly and works exactly as described.",
        height=140,
    )
    selected_model = st.selectbox("AI model", ["Naive Bayes", "SVM", "BERT"])

    if st.button("Confirm analysis", type="primary"):
        if not review.strip():
            st.warning("Please enter a product review before confirming.")
        else:
            try:
                sentiment, confidence, prepared_text = analyse_review(review, selected_model)
                issue_category = categorise_issue(review)
                add_history_item(review, selected_model, sentiment, confidence, issue_category)
                st.session_state.latest_result = (
                    sentiment,
                    confidence,
                    prepared_text,
                    selected_model,
                    issue_category,
                )
            except Exception as exc:
                st.error(str(exc))

    latest_result = st.session_state.get("latest_result")
    if latest_result:
        sentiment, confidence, prepared_text, result_model, issue_category = latest_result
        st.divider()
        st.subheader("Analysis result")
        result_col, category_col, confidence_col = st.columns(3)
        with result_col:
            if sentiment == "Positive":
                st.success("Positive review")
            elif sentiment == "Neutral":
                st.info("Neutral review")
            else:
                st.error("Negative review")
        with category_col:
            st.metric("Issue category", issue_category)
        with confidence_col:
            st.metric("Confidence", f"{confidence:.2%}")
        with st.expander("View prepared text"):
            st.code(prepared_text or "No terms remained after text preparation.", language="text")
        st.caption(f"Analysed with {result_model}")

    st.divider()
    history_heading, clear_column = st.columns([4, 1])
    with history_heading:
        st.subheader("Review history")
    with clear_column:
        if st.button("Clear history"):
            st.session_state.history = []
            st.session_state.pop("latest_result", None)
            st.rerun()

    if st.session_state.history:
        st.dataframe(
            pd.DataFrame(st.session_state.history),
            use_container_width=True,
            hide_index=True,
            column_config={"Review": st.column_config.TextColumn("Review", width="large")},
        )
    else:
        st.info("Your completed analyses will appear here during this session.")

with dataset_tab:
    dataset = load_cleaned_dataset()
    if dataset is None:
        st.info("No cleaned dataset found. Run `python src/data_preprocessing.py` first.")
    else:
        st.subheader("Cleaned Amazon reviews")
        summary_left, summary_middle, summary_right, summary_last = st.columns(4)
        summary_left.metric("Total reviews", len(dataset))
        if "sentiment" in dataset.columns:
            positive_reviews = dataset.loc[dataset["sentiment"] == 2, "cleaned_text"].dropna()
            neutral_reviews = dataset.loc[dataset["sentiment"] == 1, "cleaned_text"].dropna()
            negative_reviews = dataset.loc[dataset["sentiment"] == 0, "cleaned_text"].dropna()
            summary_middle.metric("Negative reviews", len(negative_reviews))
            summary_right.metric("Neutral reviews", len(neutral_reviews))
            summary_last.metric("Positive reviews", len(positive_reviews))

            st.subheader("Issue categories by sentiment")
            st.dataframe(
                issue_category_summary(
                    tuple(dataset["cleaned_text"].fillna("").astype(str)),
                    tuple(dataset["sentiment"].astype(int)),
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Most common words by sentiment")
            negative_column, neutral_column, positive_column = st.columns(3)
            with negative_column:
                st.markdown("**Negative reviews**")
                st.dataframe(
                    most_common_words(tuple(negative_reviews.astype(str))),
                    use_container_width=True,
                    hide_index=True,
                )
            with neutral_column:
                st.markdown("**Neutral reviews**")
                st.dataframe(
                    most_common_words(tuple(neutral_reviews.astype(str))),
                    use_container_width=True,
                    hide_index=True,
                )
            with positive_column:
                st.markdown("**Positive reviews**")
                st.dataframe(
                    most_common_words(tuple(positive_reviews.astype(str))),
                    use_container_width=True,
                    hide_index=True,
                )

with metrics_tab:
    metrics = load_saved_metrics() or compute_classical_metrics()
    if not metrics:
        st.info("No metrics are available yet. Train at least one model first.")
    else:
        model_keys = ("naive_bayes", "svm", "bert")
        table = pd.DataFrame(
            [
                {
                    "Model": key.replace("_", " ").title(),
                    "Accuracy": values.get("Accuracy") * 100 if values.get("Accuracy") is not None else None,
                    "Precision": values.get("Precision") * 100 if values.get("Precision") is not None else None,
                    "Recall": values.get("Recall") * 100 if values.get("Recall") is not None else None,
                    "F1 Score": values.get("F1-Score") * 100 if values.get("F1-Score") is not None else None,
                }
                for key in model_keys
                for values in [metrics.get(key, {})]
            ]
        )
        st.subheader("Saved model performance")
        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            column_config={
                name: st.column_config.NumberColumn(name, format="%.2%%")
                for name in ("Accuracy", "Precision", "Recall", "F1 Score")
            },
        )

        st.subheader("Model comparison")
        comparison_data = table.set_index("Model").T.dropna(axis=1, how="all")
        st.bar_chart(comparison_data, use_container_width=True, horizontal=True)

        missing_models = table.loc[table["Accuracy"].isna(), "Model"].tolist()
        if missing_models:
            st.caption(
                "No metrics are available yet for: "
                + ", ".join(missing_models)
                + ". Train and evaluate these models to include them in the chart."
            )
