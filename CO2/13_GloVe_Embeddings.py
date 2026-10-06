
# Program 13: GloVe Embeddings Loading and Vector Representation

import numpy as np

# Load GloVe embeddings
# Make sure the GloVe file is available in the Colab working directory
glove_file = "glove.6B.50d.txt"

# Load embeddings
embeddings = {}

with open(glove_file, "r", encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        embeddings[word] = vector

print("========== GLOVE EMBEDDINGS ==========")

# Select a word
word = "computer"

if word in embeddings:
    print("Word:", word)
    print("Vector:")
    print(embeddings[word])
else:
    print("Word not found in GloVe vocabulary.")

# Display vector dimension
if word in embeddings:
    print("\nVector Dimension:", len(embeddings[word]))
