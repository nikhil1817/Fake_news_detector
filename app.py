from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict_news


app = FastAPI(title="Fake News Detector API")


class NewsRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {"message": "Fake News Detector API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: NewsRequest):
    if not request.text.strip():
        return {
            "error": "Text cannot be empty"
        }

    result = predict_news(request.text)

    return {
        "text": request.text,
        "prediction": result["prediction"],
        "confidence": result["confidence"]
    }
