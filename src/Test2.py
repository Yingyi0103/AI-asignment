import pandas as pd

df = pd.read_csv("data/final_sentiment_dataset.csv")

print(df.columns.tolist())
print(df.head())
print(df.shape)
print(df["sentiment"].value_counts().sort_index())