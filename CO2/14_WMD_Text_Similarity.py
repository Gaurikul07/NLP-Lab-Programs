
# Program 14: Text Similarity using Word Mover's Distance (WMD)

import nltk
import numpy as np
from gensim.models import Word2Vec

# Download required resources
nltk.download("punkt")
nltk.download("punkt_tab")

# Sample documents
documents = [
    "the cat sits on the mat",
    "the dog sits on the rug"
]

# Tokenize documents
sentences = [nltk.word_tokenize(doc.lower()) for doc in documents]

# Train Word2Vec model on the sample corpus
model = Word2Vec(
    sentences,
    vector_size=50,
    window=3,
    min_count=1,
    workers=1,
    seed=42
)

# Calculate Word Mover's Distance
distance = model.wv.wmdistance(sentences[0], sentences[1])

print("========== DOCUMENTS ==========")
print("Document 1:", documents[0])
print("Document 2:", documents[1])

print("\n========== WORD MOVER'S DISTANCE ==========")
print("WMD:", distance)

print("\nA smaller WMD generally indicates greater semantic similarity.")
