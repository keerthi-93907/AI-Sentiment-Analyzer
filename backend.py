from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import re
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Read API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize API
app = FastAPI(title="Sentiment Analysis API", version="1.0")

# --- LOAD MODEL & TOKENIZER ---
print("Loading model and tokenizer...")

try:
    model = tf.keras.models.load_model("sentiment_model.h5")

    with open("tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)

except Exception as e:
    print(f"Error loading files: {e}")
    print("Make sure you ran train.py first!")
    exit()

# Constants (Must match training config)
MAX_LEN = 200

# --- PREPROCESSING FUNCTION ---
def preprocess_input(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()

    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    return padded

# --- API MODELS ---
class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    message: str

# --- ROUTES ---
@app.get("/")
def home():
    return {
        "message": "Sentiment Analysis API is running!",
        "api_key_loaded": True if GOOGLE_API_KEY else False
    }

@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(request: SentimentRequest):

    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # Preprocess input
    processed_text = preprocess_input(request.text)

    # Predict sentiment
    prediction_prob = model.predict(processed_text)[0][0]

    if prediction_prob > 0.5:
        sentiment = "Positive"
        confidence = prediction_prob
    else:
        sentiment = "Negative"
        confidence = 1 - prediction_prob

    return SentimentResponse(
        sentiment=sentiment,
        confidence=float(confidence),
        message="Prediction successful"
    )

# Run using:
# uvicorn backend:app --reload