import json
import os
import pickle
import re

import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

try:
    from nltk.stem import PorterStemmer, WordNetLemmatizer
except ImportError:
    PorterStemmer = None
    WordNetLemmatizer = None

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


CLEANED_DATA_PATH = "data/cleaned_amazon_reviews.csv"
TRAIN_TEST_DATA_PATH = "data/train_test_data.pkl"
VECTORIZER_PATH = "saved_models/tfidf_vectorizer.pkl"
NAIVE_BAYES_MODEL_PATH = "saved_models/naive_bayes_model.pkl"
SVM_MODEL_PATH = "saved_models/svm_model.pkl"
BERT_MODEL_DIR = "saved_models/bert_sentiment_model"

STOP_WORDS = set(ENGLISH_STOP_WORDS)
STEMMER = PorterStemmer() if PorterStemmer is not None else None
LEMMATIZER = WordNetLemmatizer() if WordNetLemmatizer is not None else None


st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="📦",
    layout="wide",
)


def clean_input(text):
    text = re.sub(r"https?://\S+|www\.\S+", " ", str(text))
    text = re.sub(r"[^a-zA-Z\s']", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()

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


@st.cache_resource
def load_models():
    vectorizer = None
    nb_model = None
    svm_model = None
    bert_pipeline = None

    if os.path.exists(VECTORIZER_PATH):
        with open(VECTORIZER_PATH, "rb") as file_handle:
            vectorizer = pickle.load(file_handle)
    if os.path.exists(NAIVE_BAYES_MODEL_PATH):
        with open(NAIVE_BAYES_MODEL_PATH, "rb") as file_handle:
            nb_model = pickle.load(file_handle)
    if os.path.exists(SVM_MODEL_PATH):
        with open(SVM_MODEL_PATH, "rb") as file_handle:
            svm_model = pickle.load(file_handle)
    if os.path.exists(BERT_MODEL_DIR):
        try:
            from transformers import pipeline

            bert_pipeline = pipeline(
                "text-classification",
                model=BERT_MODEL_DIR,
                tokenizer=BERT_MODEL_DIR,
            )
        except Exception:
            bert_pipeline = None

    return vectorizer, nb_model, svm_model, bert_pipeline


@st.cache_data
def load_cleaned_dataset():
    if not os.path.exists(CLEANED_DATA_PATH):
        return None
    return pd.read_csv(CLEANED_DATA_PATH)


@st.cache_data
def load_test_splits():
    if not os.path.exists(TRAIN_TEST_DATA_PATH):
        return None, None
    with open(TRAIN_TEST_DATA_PATH, "rb") as file_handle:
        _, x_test_vec, _, y_test = pickle.load(file_handle)
    return x_test_vec, y_test


@st.cache_data
def load_saved_metrics():
    metrics = {}
    for model_name in ("naive_bayes", "svm", "bert"):
        metrics_path = f"saved_models/{model_name}_metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as file_handle:
                metrics[model_name] = json.load(file_handle)
    return metrics


@st.cache_data
def compute_classical_metrics():
    vectorizer, nb_model, svm_model, _ = load_models()
    x_test_vec, y_test = load_test_splits()
    metrics = {}

    if x_test_vec is None or y_test is None:
        return metrics

    if nb_model is not None:
        nb_pred = nb_model.predict(x_test_vec)
        metrics["naive_bayes"] = {
            "Accuracy": float(accuracy_score(y_test, nb_pred)),
            "Precision": float(precision_score(y_test, nb_pred, zero_division=0)),
            "Recall": float(recall_score(y_test, nb_pred, zero_division=0)),
            "F1-Score": float(f1_score(y_test, nb_pred, zero_division=0)),
        }

    if svm_model is not None:
        svm_pred = svm_model.predict(x_test_vec)
        metrics["svm"] = {
            "Accuracy": float(accuracy_score(y_test, svm_pred)),
            "Precision": float(precision_score(y_test, svm_pred, zero_division=0)),
            "Recall": float(recall_score(y_test, svm_pred, zero_division=0)),
            "F1-Score": float(f1_score(y_test, svm_pred, zero_division=0)),
        }

    return metrics


st.title("📦 Product Review Sentiment Analyzer")
st.markdown(
    "Workflow: **data preprocessing → feature setup → Naive Bayes / SVM / BERT → evaluation**"
)
st.caption(
    "Run `src/data_preprocessing.py`, `feature_setup.py`, `src/naivebayes.py`, "
    "`src/svm.py`, and optionally `src/bert.py` first."
)

tab1, tab2, tab3 = st.tabs(["🔍 Analyzer", "📊 Dataset", "🏆 Metrics"])

with tab1:
    user_input = st.text_area(
        "Enter a product review:",
        "This product is amazing and works really well.",
        height=150,
    )
    model_name = st.selectbox("Choose model", ["Naive Bayes", "SVM", "BERT"])

    if st.button("Analyze Sentiment", type="primary"):
        try:
            vectorizer, nb_model, svm_model, bert_pipeline = load_models()

            if model_name == "BERT":
                if bert_pipeline is None:
                    raise FileNotFoundError("BERT model not found. Train `src/bert.py` first.")

                bert_result = bert_pipeline(user_input, truncation=True, max_length=512)[0]
                prediction = "Positive" if bert_result["label"] == "LABEL_1" else "Negative"
                confidence = float(bert_result["score"])
                cleaned_text = clean_input(user_input)
            else:
                if vectorizer is None:
                    raise FileNotFoundError("TF-IDF vectorizer not found. Run `feature_setup.py` first.")

                cleaned_text = clean_input(user_input)
                transformed_text = vectorizer.transform([cleaned_text])

                if model_name == "Naive Bayes":
                    if nb_model is None:
                        raise FileNotFoundError("Naive Bayes model not found. Train `src/naivebayes.py` first.")
                    model = nb_model
                else:
                    if svm_model is None:
                        raise FileNotFoundError("SVM model not found. Train `src/svm.py` first.")
                    model = svm_model

                predicted_index = int(model.predict(transformed_text)[0])
                prediction = "Positive" if predicted_index == 1 else "Negative"
                confidence = None
                if hasattr(model, "predict_proba"):
                    confidence = float(model.predict_proba(transformed_text)[0][predicted_index])

            if prediction == "Positive":
                st.success(f"Prediction: {prediction}")
            else:
                st.error(f"Prediction: {prediction}")

            if confidence is not None:
                st.metric("Confidence", f"{confidence * 100:.2f}%")
            st.code(cleaned_text, language="text")
        except Exception as exc:
            st.error(str(exc))

with tab2:
    dataset = load_cleaned_dataset()
    if dataset is None:
        st.info("No cleaned dataset found yet. Run `src/data_preprocessing.py` first.")
    else:
        st.metric("Total Cleaned Reviews", len(dataset))
        if "sentiment" in dataset.columns:
            positive_count = int((dataset["sentiment"] == 1).sum())
            negative_count = int((dataset["sentiment"] == 0).sum())
            st.write(
                pd.DataFrame(
                    {
                        "Sentiment": ["Positive", "Negative"],
                        "Count": [positive_count, negative_count],
                    }
                )
            )
        st.dataframe(dataset.head(20), use_container_width=True)

with tab3:
    metrics = load_saved_metrics()
    if not metrics:
        metrics = compute_classical_metrics()

    if not metrics:
        st.info("No saved metrics found yet. Train the models first.")
    else:
        metrics_rows = []
        for model_key, metric_values in metrics.items():
            metrics_rows.append(
                {
                    "Model": model_key.replace("_", " ").title(),
                    "Accuracy": metric_values.get("Accuracy"),
                    "Precision": metric_values.get("Precision"),
                    "Recall": metric_values.get("Recall"),
                    "F1-Score": metric_values.get("F1-Score"),
                }
            )
        st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True, hide_index=True)
