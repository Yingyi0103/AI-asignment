from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Test2.py is inside src/
# parents[1] takes us to the project root
ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT_DIR / "saved_models" / "bert_sentiment_model"

print("Checking BERT model...")
print("Model path:", MODEL_PATH)


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"BERT model folder not found: {MODEL_PATH}"
    )


print("\nFiles found:")

for file in MODEL_PATH.iterdir():
    print(" -", file.name)


print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

print("Tokenizer loaded successfully!")


print("\nLoading BERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

print("BERT model loaded successfully!")


print("\nModel labels:")
print(model.config.id2label)


print("\nBERT model is ready!")