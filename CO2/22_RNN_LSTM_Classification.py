import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
texts = [
    "I love this movie",
    "This film is amazing",
    "The movie was excellent",
    "I enjoyed this film",
    "This movie is wonderful",
    "I hate this movie",
    "This film is terrible",
    "The movie was boring",
    "I disliked this film",
    "This movie is awful"
]
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=6, padding="post")
y = labels
model = Sequential([
    Embedding(
        input_dim=len(tokenizer.word_index) + 1,
        output_dim=16,
        input_length=6
    ),
    LSTM(16),
    Dense(1, activation="sigmoid")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
print("========== TRAINING LSTM MODEL ==========")

model.fit(X, y, epochs=15, verbose=1)
loss, accuracy = model.evaluate(X, y, verbose=0)

print("\n========== MODEL EVALUATION ==========")
print("Loss:", round(loss, 4))
print("Accuracy:", round(accuracy, 4))
new_text = ["This movie is amazing"]

new_sequence = tokenizer.texts_to_sequences(new_text)
new_padded = pad_sequences(new_sequence, maxlen=6, padding="post")

prediction = model.predict(new_padded, verbose=0)[0][0]

print("\n========== PREDICTION ==========")
print("Text:", new_text[0])
print("Positive probability:", round(float(prediction), 4))

if prediction >= 0.5:
    print("Sentiment: Positive")
else:
    print("Sentiment: Negative")
