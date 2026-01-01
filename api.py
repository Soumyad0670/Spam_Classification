from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib  # Or pickle, depending on your model serialization
import config  # Your config.py for parameters
import numpy as np

app = FastAPI()

# Load the model and vectorizer at startup
model = joblib.load('spam_classifier_model.pkl')  # Name could differ
vectorizer = joblib.load('tfidf_vectorizer.pkl')  # If TF-IDF is used

class MessageRequest(BaseModel):
    message: str

@app.post("/predict")
def predict_spam(request: MessageRequest):
    X = vectorizer.transform([request.message])
    pred = model.predict(X)[0]
    return {"spam": bool(pred)}
