import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv("product_reviews.csv")
df["date"] = pd.to_datetime(df["date"])

# -------------------------
# 1) Data cleaning
# -------------------------
df = df.drop_duplicates()
df["review"] = df["review"].fillna("")
df["category"] = df["category"].fillna("Unknown")
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df = df.dropna(subset=["rating"])

# -------------------------
# 2) Sentiment Analysis
# Simple lexicon-based approach
# -------------------------
positive_words = {
    "excellent", "smooth", "fast", "great", "good", "comfortable", "perfect",
    "amazing", "quality", "nice", "strong", "soft", "improved", "helpful",
    "beautiful", "useful", "fair", "lasting", "compliments", "easy"
}
negative_words = {
    "cheap", "hot", "stopped", "weak", "late", "terrible", "noisy", "regret",
    "bad", "dry", "leaks", "slow", "hard", "uncomfortable", "not", "poor"
}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

def sentiment_score(text):
    words = clean_text(text).split()
    pos = sum(1 for w in words if w in positive_words)
    neg = sum(1 for w in words if w in negative_words)
    return pos - neg

def classify_sentiment(row):
    score = sentiment_score(row["review"])
    rating = row["rating"]

    # Use both text words and rating to make classification more realistic.
    if score > 0 or rating >= 4:
        return "Positive"
    elif score < 0 or rating <= 2:
        return "Negative"
    else:
        return "Neutral"

df["sentiment_score"] = df["review"].apply(sentiment_score)
df["sentiment"] = df.apply(classify_sentiment, axis=1)

# -------------------------
# 3) EDA Questions
# -------------------------
print("\nDATASET OVERVIEW")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

print("\nAVERAGE RATING BY CATEGORY")
print(df.groupby("category")["rating"].mean().sort_values(ascending=False))

print("\nSENTIMENT DISTRIBUTION")
print(df["sentiment"].value_counts())

print("\nTOP PRODUCTS BY RATING")
print(df.sort_values("rating", ascending=False)[["product", "category", "rating", "sentiment"]].head(10))

# Save processed data
df.to_csv(OUTPUT_DIR / "processed_reviews_with_sentiment.csv", index=False)

# Save summary
summary = []
summary.append("CodeAlpha Data Analytics Project Summary\n")
summary.append("Project: Product Reviews EDA, Visualization, and Sentiment Analysis\n")
summary.append(f"Total reviews analyzed: {len(df)}\n")
summary.append(f"Average rating: {df['rating'].mean():.2f}\n")
summary.append("\nAverage rating by category:\n")
summary.append(df.groupby("category")["rating"].mean().sort_values(ascending=False).to_string())
summary.append("\n\nSentiment distribution:\n")
summary.append(df["sentiment"].value_counts().to_string())
summary.append("\n\nMain insights:\n")
summary.append("- Positive reviews are mainly connected to comfort, quality, design, and usefulness.\n")
summary.append("- Negative reviews are mainly connected to poor quality, late delivery, weak material, leakage, or bad performance.\n")
summary.append("- Categories with higher average ratings show stronger customer satisfaction.\n")
summary.append("- Products with low ratings should be investigated to improve quality and customer experience.\n")

with open(OUTPUT_DIR / "analysis_summary.txt", "w", encoding="utf-8") as f:
    f.write("".join(summary))

# -------------------------
# 4) Data Visualization
# -------------------------

# Chart 1: Sentiment count
sentiment_counts = df["sentiment"].value_counts()
plt.figure(figsize=(7, 5))
plt.bar(sentiment_counts.index, sentiment_counts.values)
plt.title("Sentiment Distribution of Product Reviews")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sentiment_distribution.png", dpi=300)
plt.close()

# Chart 2: Average rating by category
avg_rating = df.groupby("category")["rating"].mean().sort_values()
plt.figure(figsize=(8, 5))
plt.barh(avg_rating.index, avg_rating.values)
plt.title("Average Rating by Product Category")
plt.xlabel("Average Rating")
plt.ylabel("Category")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "average_rating_by_category.png", dpi=300)
plt.close()

# Chart 3: Number of reviews by category
category_counts = df["category"].value_counts()
plt.figure(figsize=(8, 5))
plt.bar(category_counts.index, category_counts.values)
plt.title("Number of Reviews by Category")
plt.xlabel("Category")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "reviews_by_category.png", dpi=300)
plt.close()

# Chart 4: Ratings over time
daily_rating = df.groupby("date")["rating"].mean()
plt.figure(figsize=(9, 5))
plt.plot(daily_rating.index, daily_rating.values, marker="o")
plt.title("Average Rating Over Time")
plt.xlabel("Date")
plt.ylabel("Average Rating")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "average_rating_over_time.png", dpi=300)
plt.close()

print("\nProject completed successfully.")
print("Check the outputs folder for charts, processed data, and analysis summary.")