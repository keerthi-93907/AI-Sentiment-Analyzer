import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, SpatialDropout1D
import numpy as np
import pandas as pd
import pickle
import re

# --- CONFIGURATION ---
VOCAB_SIZE = 10000  # Only keep the top 10,000 most frequent words
MAX_LEN = 200       # Max length of a review (in words)
EMBEDDING_DIM = 128 # Size of the vector representation for each word
EPOCHS = 5          # Number of training rounds
BATCH_SIZE = 32

print("Downloading dataset...")
# We use the official IMDB dataset from Keras
# This version gives us integers, but we decode it to text to simulate real-world preprocessing
dataset = tf.keras.utils.get_file(
    fname="aclImdb_v1.tar.gz", 
    origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", 
    untar=True
)

import os
import glob

def load_data(data_dir):
    data = {}
    for split in ["train", "test"]:
        data[split] = []
        for sentiment in ["neg", "pos"]:
            score = 1 if sentiment == "pos" else 0
            path = os.path.join(data_dir, "aclImdb", split, sentiment, "*.txt")
            files = glob.glob(path)
            # Load a subset for speed in this demo (use all files for full training)
            for f in files[:2000]: 
                with open(f, "r", encoding="utf-8") as file:
                    data[split].append({"text": file.read(), "sentiment": score})
    return pd.DataFrame(data["train"]), pd.DataFrame(data["test"])

# Load data (using the downloaded path)
data_dir = dataset
print(f"Data directory: {data_dir}")
train_df, test_df = load_data(data_dir)

print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")

# --- PREPROCESSING ---
def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphabetic characters and lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    return text

print("Cleaning text...")
train_df['text'] = train_df['text'].apply(clean_text)
test_df['text'] = test_df['text'].apply(clean_text)

print("Tokenizing...")
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df['text'])

# Convert text to sequences of integers
X_train = tokenizer.texts_to_sequences(train_df['text'])
X_test = tokenizer.texts_to_sequences(test_df['text'])

# Pad sequences
X_train = pad_sequences(X_train, maxlen=MAX_LEN, padding='post', truncating='post')
X_test = pad_sequences(X_test, maxlen=MAX_LEN, padding='post', truncating='post')

y_train = np.array(train_df['sentiment'])
y_test = np.array(test_df['sentiment'])

# --- MODEL ARCHITECTURE ---
print("Building Model...")
model = Sequential([
    # 1. Embedding Layer: Converts integer word IDs to dense vectors of size 128
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
    
    # 2. Spatial Dropout: Prevents overfitting by dropping 1D feature maps
    SpatialDropout1D(0.2),
    
    # 3. Bidirectional LSTM: Reads text forwards AND backwards for better context
    Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
    
    # 4. Dense Layer: High-level feature learning
    Dense(32, activation='relu'),
    
    # 5. Output Layer: Single neuron, 0 to 1 probability (Sigmoid)
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
)

model.summary()

# --- TRAINING ---
print("Training Model...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    verbose=1
)

# --- EVALUATION ---
print("Evaluating Model...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# --- SAVING ---
print("Saving artifacts...")
model.save("sentiment_model.h5")

with open("tokenizer.pkl", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Done! Model and Tokenizer saved.")