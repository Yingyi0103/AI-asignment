import pandas as pd

df = pd.read_csv("data/final_sentiment_dataset.csv")

print("=" * 60)
print("COLUMNS")
print("=" * 60)
print(df.columns.tolist())

print("\n" + "=" * 60)
print("DATASET SIZE")
print("=" * 60)
print(df.shape)

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)
print(df["sentiment"].value_counts().sort_index())

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(df.isnull().sum())

print("\n" + "=" * 60)
print("DUPLICATES")
print("=" * 60)
print("Duplicate rows:", df.duplicated().sum())

print("\n" + "=" * 60)
print("SAMPLE")
print("=" * 60)
print(df.sample(5, random_state=42).to_string(index=False))