import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


FAKE_PATH = "data/Fake.csv"
TRUE_PATH = "data/True.csv"

MODEL_DIR = "models"
MODEL_PATH = "models/fake_news_model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"


def clean_text(text):
    if not isinstance(text, str):
        return ""
    return text.lower().strip()


def load_dataset():
    fake_df = pd.read_csv(FAKE_PATH)
    true_df = pd.read_csv(TRUE_PATH)

    fake_df["label"] = "FAKE"
    true_df["label"] = "REAL"

    # combine title + text
    fake_df["content"] = (
        fake_df["title"].fillna("") + " " +
        fake_df["text"].fillna("")
    )

    true_df["content"] = (
        true_df["title"].fillna("") + " " +
        true_df["text"].fillna("")
    )

    df = pd.concat([
        fake_df[["content", "label"]],
        true_df[["content", "label"]]
    ])

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


def train_model():
    if not os.path.exists(FAKE_PATH):
        raise FileNotFoundError("Fake.csv not found")

    if not os.path.exists(TRUE_PATH):
        raise FileNotFoundError("True.csv not found")

    df = load_dataset()

    df["content"] = df["content"].apply(clean_text)

    X = df["content"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2)
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        solver="liblinear"
    )

    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, predictions)

    print("\nModel trained successfully.")
    print("Accuracy:", round(accuracy, 4))

    print("\nClassification Report:\n")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Vectorizer saved to {VECTORIZER_PATH}")


if __name__ == "__main__":
    train_model()
