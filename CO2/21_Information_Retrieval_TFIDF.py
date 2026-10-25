from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
documents = [
    "Natural Language Processing helps computers understand human language.",
    "Machine learning is used in artificial intelligence.",
    "Information retrieval finds relevant documents from a collection.",
    "TF-IDF is used to rank documents based on a search query.",
    "Deep learning is a part of machine learning."
]
query = "document ranking using TF-IDF"
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)
query_vector = vectorizer.transform([query])
scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
ranked_indices = scores.argsort()[::-1]

print("========== INFORMATION RETRIEVAL SYSTEM ==========")
print("Search Query:", query)

print("\n========== RANKED RESULTS ==========")

for rank, index in enumerate(ranked_indices, start=1):
    if scores[index] > 0:
        print("\nRank:", rank)
        print("Similarity Score:", round(scores[index], 4))
        print("Document:", documents[index])
if max(scores) == 0:
    print("No relevant documents found.")
