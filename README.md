# Product Review Sentiment Analyzer

<<<<<<< HEAD
A Streamlit web application for classifying Amazon product reviews as **negative**, **neutral**, or **positive**. The project compares three sentiment-analysis approaches:
=======
A Streamlit web application for classifying Amazon product reviews as **positive** or **negative**. The project compares three sentiment-analysis approaches:
>>>>>>> c9e4ea23807d8ac93774f687cde20ca737a6f5be

- Multinomial Naive Bayes
- Linear Support Vector Machine (SVM)
- Fine-tuned BERT (optional)

Alongside a prediction, the app shows model confidence, assigns a simple issue category, retains an in-session review history, and provides dataset and model-performance views.

## Features

<<<<<<< HEAD
- Classify a custom product review as negative, neutral, or positive.
=======
- Classify a custom product review as positive or negative.
>>>>>>> c9e4ea23807d8ac93774f687cde20ca737a6f5be
- Choose between Naive Bayes, SVM, and BERT models when they are trained.
- See the prediction confidence and the prepared text used by classical models.
- Categorise review content into product quality/performance, product accuracy/expectation,
  delivery/packaging, seller/customer service, price/value, or other issues.
- Explore class counts, issue categories, and common words in the cleaned dataset.
- Compare accuracy, precision, recall, and F1 score for trained models.

## Project structure

```text
.
├── app.py                       # Streamlit application
├── feature_setup.py             # Builds TF-IDF features and train/test split
├── data/
│   ├── train.ft.txt.bz2         # Raw fastText-format Amazon review dataset
│   ├── cleaned_amazon_reviews.csv
│   └── train_test_data.pkl
├── saved_models/
│   ├── tfidf_vectorizer.pkl
│   ├── naive_bayes_model.pkl
│   └── *_metrics.json
└── src/
    ├── data_preprocessing.py    # Cleans and samples the raw dataset
    ├── naivebayes.py            # Trains Naive Bayes
    ├── svm.py                   # Trains linear SVM
    ├── bert.py                  # Fine-tunes BERT
    ├── get_bert_metrics.py      # Evaluates a saved BERT model
    └── evaluation.py            # Shared evaluation utilities
```

## Requirements

- Python 3.10 or later
- `pip`

Install the dependencies:

```bash
pip install streamlit pandas scikit-learn nltk
```

To train or run the BERT model, also install:

```bash
pip install torch transformers
```

> On its first run, the BERT training script downloads the `bert-base-uncased` checkpoint from Hugging Face.

## Run the app

From the project root:

```bash
streamlit run app.py
```

Open the local address displayed by Streamlit, usually `http://localhost:8501`.

<<<<<<< HEAD
After regenerating the three-class dataset, rebuild the TF-IDF features and retrain any classical model. Previously saved two-class models cannot predict neutral reviews.
=======
The repository includes a trained Naive Bayes model and TF-IDF vectorizer, so the app can be started immediately. Train the SVM or BERT model if you would like those options to be available too.
>>>>>>> c9e4ea23807d8ac93774f687cde20ca737a6f5be

## Train the models

Run commands from the project root in this order.

### 1. Preprocess the dataset

<<<<<<< HEAD
This combines both source files into `data/cleaned_amazon_reviews.csv`: up to 25,000 negative and 25,000 positive reviews from `train.ft.txt.bz2`, plus every usable review from `data_amazon.xlsx - Sheet1.csv`. Ratings of 1-2 become negative, 3 becomes neutral, and 4-5 become positive.
=======
This creates a balanced sample of up to 25,000 reviews per class, cleans the text, and saves `data/cleaned_amazon_reviews.csv`.
>>>>>>> c9e4ea23807d8ac93774f687cde20ca737a6f5be

```bash
python src/data_preprocessing.py
```

### 2. Build TF-IDF features

This creates the train/test split and saves the vectorizer.

```bash
python feature_setup.py
```

### 3. Train a classical model

```bash
python src/naivebayes.py
python src/svm.py
```

Each command saves its trained model and evaluation metrics to `saved_models/`.

### 4. Train BERT (optional)

Fine-tuning is more resource-intensive than the classical models and is best run with a compatible GPU when available.

```bash
python src/bert.py
```

The trained model and tokenizer are saved to `saved_models/bert_sentiment_model/`. To recalculate its evaluation metrics later, run:

```bash
python src/get_bert_metrics.py
```

## Data preparation

<<<<<<< HEAD
The cleaned dataset combines both raw files. Its labels are:

- `0` = negative (fastText `__label__1` and ratings 1-2)
- `1` = neutral (ratings 3)
- `2` = positive (fastText `__label__2` and ratings 4-5)
=======
The raw dataset uses fastText labels:

- `__label__1` → negative (`0`)
- `__label__2` → positive (`1`)
>>>>>>> c9e4ea23807d8ac93774f687cde20ca737a6f5be

Preprocessing normalises text, removes URLs and non-letter characters, removes English stop words, and applies optional lemmatisation and stemming. The classical models use TF-IDF unigrams and bigrams with up to 5,000 features.

## Current result

The saved Naive Bayes model was evaluated on the held-out test split:

| Model | Accuracy | Precision | Recall | F1 score |
| --- | ---: | ---: | ---: | ---: |
| Naive Bayes | 84.84% | 84.45% | 85.40% | 84.92% |

## Notes

- Selecting a model that has not yet been trained shows a helpful message explaining which training command to run.
- The review history is stored only for the current browser session.
- NLTK lemmatisation is optional; the project still runs if the WordNet corpus is not installed.

## License

Add a license file before publishing if you would like to specify how others may use this project.
