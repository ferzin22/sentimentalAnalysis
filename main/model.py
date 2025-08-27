import nltk
from nltk.corpus import movie_reviews
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# i will downalod the dataset from reviews to just train the data
nltk.download('movie_reviews')

# Load IMDB movie reviews dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

# Convert to DataFrame
df = pd.DataFrame(documents, columns=["text", "sentiment"])

# Convert words list to string sentences
df["text"] = df["text"].apply(lambda words: " ".join(words))

# Map sentiment labels (positive = 1, negative = 0)
df["sentiment"] = df["sentiment"].map({"pos": 1, "neg": 0})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["sentiment"], test_size=0.2, random_state=42)

# Vectorizing text data
vectorizer = TfidfVectorizer(max_features=5000)  # Use 5000 words for better accuracy
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)


joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model trained with real dataset and saved successfully!")
