from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
documents = [
    "Natural Language Processing is interesting.",
    "Natural Language Processing is useful.",
    "Machine Learning is interesting."
]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

similarity_matrix = cosine_similarity(tfidf_matrix)

print("========== DOCUMENTS ==========")

for i, document in enumerate(documents):
    print("Document", i + 1, ":", document)

print("\n========== COSINE SIMILARITY MATRIX ==========")
print(similarity_matrix)

print("\n========== SIMILARITY BETWEEN DOCUMENTS ==========")

for i in range(len(documents)):
    for j in range(i + 1, len(documents)):
        print(
            "Document", i + 1,
            "and Document", j + 1,
            ":", round(similarity_matrix[i][j], 4)
        )
