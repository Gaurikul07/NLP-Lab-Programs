
# Program 12: Word2Vec Word Embeddings using Gensim

from gensim.models import Word2Vec

# Custom corpus
sentences = [
    ["natural", "language", "processing", "is", "interesting"],
    ["natural", "language", "processing", "is", "useful"],
    ["machine", "learning", "is", "interesting"],
    ["machine", "learning", "is", "useful"],
    ["deep", "learning", "is", "a", "part", "of", "machine", "learning"]
]

# Train Word2Vec model
model = Word2Vec(
    sentences,
    vector_size=50,
    window=3,
    min_count=1,
    workers=1
)

print("========== WORD2VEC WORD EMBEDDINGS ==========")

# Display vector for a word
word = "learning"

print("\nWord:", word)
print("Vector:")
print(model.wv[word])

# Find similar words
print("\n========== SIMILAR WORDS ==========")

similar_words = model.wv.most_similar("learning", topn=3)

for word, similarity in similar_words:
    print(word, "->", round(similarity, 4))
