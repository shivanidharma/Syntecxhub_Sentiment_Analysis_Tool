import pandas as pd
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report

# -----------------------------
# Text Preprocessing Function
# -----------------------------
def preprocess_text(text):
    text = text.lower()                          # Lowercase
    text = re.sub(r"http\S+", "", text)          # Remove URLs
    text = re.sub(r"\d+", "", text)              # Remove Numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove Punctuation
    text = " ".join(text.split())                # Remove Extra Spaces
    return text

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("IMDB Dataset.csv")

# Keep only required columns
df = df[["review", "sentiment"]]


# Preprocess Reviews
df["review"] = df["review"].apply(preprocess_text)

# Features and Labels
X = df["review"]
y = df["sentiment"]

# -----------------------------
# Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Feature Extraction
# -----------------------------
vectorizer = TfidfVectorizer(stop_words="english")

# If internship asks CountVectorizer, replace above with:
# vectorizer = CountVectorizer(stop_words="english")

X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)

# -----------------------------
# Train Model
# -----------------------------
model = MultinomialNB()
model.fit(X_train_vector, y_train)

# -----------------------------
# Prediction
# -----------------------------
predictions = model.predict(X_test_vector)

# -----------------------------
# Evaluation
# -----------------------------
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="weighted")

print("=" * 50)
print("Sentiment Analysis using Naive Bayes")
print("=" * 50)

print(f"Accuracy : {accuracy:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

# -----------------------------
# CLI Prediction
# -----------------------------
print("\n====================================")
print("Sentiment Prediction CLI")
print("Type 'exit' to quit")
print("====================================")

while True:
    user_input = input("\nEnter Review: ")

    if user_input.lower() == "exit":
        print("Program Closed.")
        break

    cleaned = preprocess_text(user_input)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)

    print("Predicted Sentiment:", prediction[0])