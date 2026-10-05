
# Program 8: Bag-of-Words (BoW) Vectorization and Representation

from sklearn.feature_extraction.text import CountVectorizer

# Sample documents
documents = [
    "Natural Language Processing is interesting.",
    "Natural Language Processing is useful.",
    "Machine Learning is interesting and useful."
]

# Create CountVectorizer
vectorizer = CountVectorizer()

# Convert documents into BoW representation
bow_matrix = vectorizer.fit_transform(documents)

# Get vocabulary
print("========== VOCABULARY ==========")
print(vectorizer.get_feature_names_out())

# Display BoW matrix
print("\n========== BAG-OF-WORDS MATRIX ==========")
print(bow_matrix.toarray())

# Display each document with its vector
print("\n========== DOCUMENT REPRESENTATION ==========")

for i, vector in enumerate(bow_matrix.toarray()):
    print("Document", i + 1, ":", vector)
