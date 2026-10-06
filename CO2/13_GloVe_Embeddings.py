import numpy as np

glove_file = "glove.6B.50d.txt"

embeddings = {}

with open(glove_file, "r", encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        embeddings[word] = vector

print("========== GLOVE EMBEDDINGS ==========")

word = "computer"

if word in embeddings:
    print("Word:", word)
    print("Vector:")
    print(embeddings[word])
else:
    print("Word not found in GloVe vocabulary.")

if word in embeddings:
    print("\nVector Dimension:", len(embeddings[word]))
