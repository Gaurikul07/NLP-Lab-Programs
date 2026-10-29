
# Program 23: Implementing Attention Mechanism using a Basic Custom Model

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Softmax, Multiply, Lambda
from tensorflow.keras.models import Model

# Sample input sequence
# 3 time steps, each with 4 features
sequence_data = np.array([
    [[0.2, 0.5, 0.1, 0.7],
     [0.8, 0.3, 0.6, 0.2],
     [0.4, 0.9, 0.2, 0.5]]
], dtype=np.float32)

# Define input layer
inputs = Input(shape=(3, 4))

# Calculate attention scores for each time step
scores = Dense(1, activation="tanh")(inputs)

# Convert scores into attention weights
scores = Lambda(lambda x: tf.squeeze(x, axis=-1))(scores)
attention_weights = Softmax(axis=1, name="attention_weights")(scores)

# Expand weights for multiplication with input features
weights_expanded = Lambda(lambda x: tf.expand_dims(x, axis=-1))(attention_weights)

# Apply attention weights to input sequence
weighted_sequence = Multiply()([inputs, weights_expanded])

# Sum weighted features to create context vector
context_vector = Lambda(lambda x: tf.reduce_sum(x, axis=1))(weighted_sequence)

# Build model
model = Model(inputs=inputs, outputs=context_vector)

# Build a second model to display attention weights
attention_model = Model(inputs=inputs, outputs=attention_weights)

# Display input
print("========== INPUT SEQUENCE ==========")
print(sequence_data)

# Calculate context vector
context = model.predict(sequence_data, verbose=0)

print("\n========== CONTEXT VECTOR ==========")
print(context)

# Display attention weights
weights = attention_model.predict(sequence_data, verbose=0)

print("\n========== ATTENTION WEIGHTS ==========")
print(weights)

print("\nSum of attention weights:", np.sum(weights))

print("\nHigher attention weight means that the time step")
print("contributes more to the context vector.")
