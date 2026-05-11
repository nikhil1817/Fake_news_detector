import os
import joblib
import math


MODEL_PATH = "models/fake_news_model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"


def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError("Model files not found. Run train_model.py first.")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def predict_news(text):
    model, vectorizer = load_model()

    text_vec = vectorizer.transform([text])

    prediction = model.predict(text_vec)[0]

    # Avoid predict_proba() because your sklearn version is causing multi_class error
    try:
        score = model.decision_function(text_vec)[0]
        confidence = sigmoid(abs(score))
    except Exception:
        confidence = 0.5

    return {
        "prediction": prediction,
        "confidence": round(float(confidence), 4)
    }
