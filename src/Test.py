import pandas as pd

df = pd.read_csv("data/cleaned_amazon_reviews.csv")

print(df["sentiment"].value_counts().sort_index())