#RNN


import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')

MAX_WORDS = 10000      # Top words to keep in vocabulary
MAX_LEN = 100          # Max number of words per sentence
EMBEDDING_DIM = 128
BATCH_SIZE = 64
EPOCHS = 5

# =========================================================
# 2. Prepare DataFrame (Assuming a CSV with 'text' and 'label')
# =========================================================
# Using dummy data creation to illustrate the structure
data = {
    'text': ["I loved this movie", "It was a terrible film", "Great acting", "Boring plot"],
    'label': [1, 0, 1, 0] # 1 for Positive, 0 for Negative
}
df = pd.DataFrame(data)

train_df, validate_df = train_test_split(df, test_size=0.2, random_state=42)

# =========================================================
# 3. Text Preprocessing (The "ImageGenerator" equivalent for Text)
# =========================================================
tokenizer = Tokenizer(num_words=MAX_WORDS, lower=True)
tokenizer.fit_on_texts(train_df['text'].values)

def preprocess_text(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    # Padds shorter sentences with zeros so they are all the same length
    return pad_sequences(sequences, maxlen=MAX_LEN)

X_train = preprocess_text(train_df['text'].values)
y_train = train_df['label'].values

X_val = preprocess_text(validate_df['text'].values)
y_val = validate_df['label'].values

# =========================================================
# 4. LSTM Model Architecture
# =========================================================
model = Sequential([
    # Embedding: Turns word indices into dense vectors
    Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),

    # Dropout specific to sequences
    SpatialDropout1D(0.2),

    # The LSTM layer: captures the "memory" of the sentence
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),

    Dense(64, activation='relu'),
    Dropout(0.5),

    # Output layer (float32 for stability)
    Dense(1, activation='sigmoid', dtype='float32')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# =========================================================
# 5. Training
# =========================================================
model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val)
)

# =========================================================
# 6. Prediction
# =========================================================
new_texts = ["This movie was absolutely fantastic!", "I hated every minute of it."]
new_seq = preprocess_text(new_texts)
predictions = model.predict(new_seq)

for text, prob in zip(new_texts, predictions):
    sentiment = "Positive" if prob > 0.5 else "Negative"
    print(f"Text: {text} | Sentiment: {sentiment} ({prob[0]:.4f})")

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, Input

# 1. Configuration
MAX_WORDS = 10000 # Only keep the top 10k most frequent words
MAX_LEN = 200     # Cut off reviews after 200 words
BATCH_SIZE = 128
EPOCHS = 5

# 2. Load Real Data
print("Loading IMDB data...")
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=MAX_WORDS)

# Pad sequences (Images have fixed pixels, text needs fixed sequence length)
X_train = pad_sequences(X_train, maxlen=MAX_LEN)
X_test = pad_sequences(X_test, maxlen=MAX_LEN)

# 3. The Model
model = Sequential([
    Input(shape=(MAX_LEN,)),
    Embedding(MAX_WORDS, 128),

    # Using a Bi-directional LSTM to understand context from both ends
    Bidirectional(LSTM(64, return_sequences=False)),

    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid') # Binary classification (0: Bad, 1: Good)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Training
# This will take a minute or two on a GPU
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)

# 5. Testing with Custom Text
def predict_sentiment(text):
    # Word index for IMDB
    word_index = imdb.get_word_index()
    # Tokenize and encode custom text
    words = text.lower().split()
    tokens = [word_index.get(w, 0) + 3 for w in words] # +3 is the standard IMDB offset
    tokens = [t if t < MAX_WORDS else 2 for t in tokens] # Handle OOV words

    padded = pad_sequences([tokens], maxlen=MAX_LEN)
    prediction = model.predict(padded, verbose=0)
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    print(f"Text: {text} | Sentiment: {sentiment} ({prediction[0][0]:.4f})")

# Test it again!
predict_sentiment("This was a fantastic and amazing masterpiece")
predict_sentiment("This was a boring and terrible waste of time")

#LSTM IMBD
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D

MAX_WORDS = 10000
MAX_LEN = 100
EMBEDDING_DIM = 128
BATCH_SIZE = 64
EPOCHS = 5

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=MAX_WORDS)

X_train = pad_sequences(X_train, maxlen=MAX_LEN)
X_test = pad_sequences(X_test, maxlen=MAX_LEN)

model = Sequential([
    Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),
    SpatialDropout1D(0.2),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test)
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Final Test Loss: {loss:.4f}")
print(f"Final Test Accuracy: {accuracy*100:.4f}%")
print("\n--- Sample Predictions from Test Set ---")
predictions = model.predict(X_test[:5]) # Predict the first 5 test reviews

for i in range(5):
    prob = predictions[i][0]
    sentiment = "Positive" if prob > 0.5 else "Negative"
    actual = "Positive" if y_test[i] == 1 else "Negative"

    print(f"Review {i+1}:")
    print(f"  Predicted: {sentiment} (Probability: {prob:.4f})")
    print(f"  Actual: {actual}")
    print("-" * 30)
