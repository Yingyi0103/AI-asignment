# Product Review Sentiment Analyzer

A Streamlit web application for classifying products reviews into **negative**, **neutral**, or **positive** sentiment.

The project compares three sentiment-analysis approaches:

- **Multinomial Naive Bayes**
- **Linear Support Vector Machine (SVM)**
- **Fine-tuned BERT**

The application provides sentiment predictions, model confidence, issue-category detection, review history, dataset statistics, and model-performance comparisons.

## Features

- Classify a custom product review as **negative**, **neutral**, or **positive**.
- Choose between Naive Bayes, SVM, and BERT models.
- Display the predicted sentiment and model confidence.
- Display the processed text used by the classical machine-learning models.
- Categorise reviews into:
  - Product Quality / Performance
  - Product Accuracy / Expectation
  - Delivery / Packaging
  - Seller / Customer Service
  - Price / Value
  - Other
- Explore sentiment class distributions.
- Explore issue categories and common words in the dataset.
- Compare model performance using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- Maintain an in-session review history.
- Use previously trained models without retraining.

---

## Project Structure

```text
.
├── app.py                              # Streamlit application
├── feature_setup.py                    # Builds TF-IDF features and train/test split
├── README.md
├── .gitattributes                      # Git LFS configuration
│
├── data/
│   ├── amazon_new.csv                  # Binary sentiment source dataset
│   ├── data_amazon.xlsx - Sheet1.csv   # Original Amazon review dataset
│   ├── Reviews_New.csv                 # Rating-based review dataset
│   ├── final_sentiment_dataset.csv     # Final balanced dataset
│   ├── dataset_split.pkl               # Shared train/test split
│   └── train_test_data.pkl             # TF-IDF train/test data
│
├── saved_models/
│   ├── bert_metrics.json               # BERT evaluation metrics
│   ├── naive_bayes_metrics.json        # Naive Bayes evaluation metrics
│   ├── naive_bayes_model.pkl           # Trained Naive Bayes model
│   ├── svm_metrics.json                # SVM evaluation metrics
│   ├── svm_model.pkl                   # Trained SVM model
│   ├── tfidf_vectorizer.pkl            # Trained TF-IDF vectorizer
│   │
│   └── bert_sentiment_model/
│       ├── config.json
│       ├── model.safetensors           # Fine-tuned BERT weights
│       ├── tokenizer.json
│       └── tokenizer_config.json
│
└── src/
    ├── data_preprocessing.py           # Creates the final balanced dataset
    ├── evaluation.py                   # Shared evaluation utilities
    ├── naivebayes.py                   # Trains Naive Bayes
    ├── svm.py                          # Trains SVM
    ├── bert.py                         # Fine-tunes BERT
    └── get_bert_metrics.py             # Evaluates saved BERT model


## Requirements
Before running the project, install Python.
Recommended version:
- Python 3.10 or later
You can check your Python version using:
```bash
python --version
```

# Installation
## 1. Clone the Repository

Clone the repository from GitHub:

```bash
git clone https://github.com/Yingyi0103/AI-asignment.git
```

Move into the project directory:

```bash
cd AI-asignment
```
## 2. Install Git LFS

The trained BERT model is stored using Git Large File Storage (Git LFS).

Install Git LFS if it is not already installed:

```bash
git lfs install
```

Then download the LFS files:

```bash
git lfs pull
```

This downloads the BERT model file:

```text
saved_models/bert_sentiment_model/model.safetensors
```

---

## 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Alternatively, install the required packages manually:

```bash
pip install streamlit pandas numpy scikit-learn nltk
```

For the BERT model:

```bash
pip install torch transformers
```

Depending on the version of the project, the following packages may also be required:

```bash
pip install scipy
```

---

# Running the Application

From the project root directory, run:

```bash
streamlit run app.py
```

Streamlit will start a local web server.

Open the address displayed in the terminal, usually:

```text
http://localhost:8501
```

The application should open in your web browser.

---

# Using the Application

## Single Review Prediction

1. Open the Streamlit application.
2. Select a sentiment analysis model.
3. Enter a product review.
4. Run the prediction.
5. The system will display:
   - Predicted sentiment
   - Model confidence
   - Detected issue category

Example review:

```text
The product arrived quickly and works exactly as expected.
```

---

## Available Models

The application supports three sentiment analysis models.

### Multinomial Naive Bayes

A classical machine learning model that uses TF-IDF features.

Advantages:

- Fast training
- Fast predictions
- Lower computational requirements

---

### Linear Support Vector Machine

A classical machine learning model trained using TF-IDF features.

Advantages:

- Strong performance on text classification
- Efficient for large datasets
- Usually provides good generalisation

---

### Fine-Tuned BERT

A transformer-based deep learning model fine-tuned for sentiment classification.

Advantages:

- Understands contextual relationships between words
- Uses natural review text
- Can capture more complex language patterns

Disadvantages:

- Requires significantly more computational resources
- Training is recommended on a GPU

---

# Dataset Preparation

The project uses multiple review datasets to create a balanced three-class sentiment dataset.

The final dataset is saved as:

```text
data/final_sentiment_dataset.csv
```

The sentiment labels are:

| Label | Sentiment |
|---|---|
| 0 | Negative |
| 1 | Neutral |
| 2 | Positive |

The final dataset is balanced using:

- 25,000 Negative reviews
- 25,000 Neutral reviews
- 25,000 Positive reviews

Total:

```text
75,000 reviews
```

The dataset is created from multiple review sources.

---

# Rebuilding the Dataset

If you want to regenerate the final sentiment dataset, run:

```bash
python src/data_preprocessing.py
```

The script creates:

```text
data/final_sentiment_dataset.csv
```

The preprocessing process:

1. Loads the available review datasets.
2. Removes HTML tags and URLs.
3. Removes invalid reviews.
4. Removes very short reviews.
5. Removes duplicate reviews.
6. Converts ratings into sentiment labels.
7. Filters neutral candidates.
8. Removes strongly positive or negative reviews from neutral candidates.
9. Balances the dataset.
10. Saves the final dataset.

---

#
```

### 2. Build TF-IDF features

This creates the train/test split and saves the vectorizer.

```bash
python feature_setup.py
```
 Building TF-IDF Features

Before training Naive Bayes or SVM, create the shared train/test split and TF-IDF features.

Run:

```bash
python feature_setup.py
```

This process:

1. Loads `final_sentiment_dataset.csv`.
2. Removes invalid reviews.
3. Cleans review text.
4. Removes duplicate reviews.
5. Creates a stratified train/test split.
6. Fits the TF-IDF vectorizer using training data only.
7. Saves the vectorizer.
8. Saves the train/test data.

The following files are created:

```text
data/dataset_split.pkl
data/train_test_data.pkl
saved_models/tfidf_vectorizer.pkl
```

The same train/test split is used to support fair comparison between models.

---

# Training Naive Bayes

Run:

```bash
python src/naivebayes.py
```

The trained model and evaluation metrics are saved in:

```text
saved_models/naive_bayes_model.pkl
saved_models/naive_bayes_metrics.json
```

---

# Training SVM

Run:

```bash
python src/svm.py
```

The trained model and evaluation metrics are saved in:

```text
saved_models/svm_model.pkl
saved_models/svm_metrics.json
```

---

# Training BERT

BERT training requires significantly more computational resources than Naive Bayes and SVM.

For best performance, training should be performed using a GPU.

Run:

```bash
python src/bert.py
```

The fine-tuned BERT model is saved in:

```text
saved_models/bert_sentiment_model/
```

The folder contains:

```text
config.json
model.safetensors
tokenizer.json
tokenizer_config.json
```

---

# Evaluating BERT

To evaluate the saved BERT model, run:

```bash
python src/get_bert_metrics.py
```

The evaluation metrics are saved as:

```text
saved_models/bert_metrics.json
```

---

```
# Recommended Training Order

If you are starting from the raw datasets, run the commands in the following order.

## Step 1: Create the Final Dataset

```bash
python src/data_preprocessing.py
```

## Step 2: Build TF-IDF Features

```bash
python feature_setup.py
```

## Step 3: Train Naive Bayes

```bash
python src/naivebayes.py
```

## Step 4: Train SVM

```bash
python src/svm.py
```

## Step 5: Train BERT

```bash
python src/bert.py
```

## Step 6: Evaluate BERT

```bash
python src/get_bert_metrics.py
```

---

# Using Pre-Trained Models

The repository may already contain trained models.

If the following files are available:

```text
saved_models/naive_bayes_model.pkl
saved_models/svm_model.pkl
saved_models/tfidf_vectorizer.pkl
saved_models/bert_sentiment_model/
```

you can run the Streamlit application without retraining the models.

Simply run:

```bash
streamlit run app.py
```

---

# Git LFS

The BERT model file is large:

```text
model.safetensors
```

Therefore, it is stored using Git Large File Storage (Git LFS).

After cloning the repository, make sure to run:

```bash
git lfs install
git lfs pull
```

Without Git LFS, the full BERT model may not be downloaded correctly.

---

## Current result

The saved Naive Bayes model was evaluated on the held-out test split:

| Model | Accuracy | Precision | Recall | F1 score |
| --- | ---: | ---: | ---: | ---: |
| Naive Bayes | 84.84% | 84.45% | 85.40% | 84.92% |

# Notes

- Naive Bayes and SVM use TF-IDF features.
- BERT uses natural review text.
- The same train/test split is used for model comparison.
- The review history is stored only during the current Streamlit session.
- BERT training is recommended on a GPU.
- If the final dataset is regenerated, the TF-IDF features and classical models should also be rebuilt and retrained.

---

# Technologies Used

The project uses:

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- NLTK
- PyTorch
- Hugging Face Transformers
- Git
- Git LFS

---

# Running the Complete System

For most users who simply want to run the application:

```bash
git clone https://github.com/Yingyi0103/AI-asignment.git
cd AI-asignment
git checkout Final_Testing
git lfs install
git lfs pull
pip install -r requirements.txt
streamlit run app.py
```

---

# License

This project is developed for educational and academic purposes.
