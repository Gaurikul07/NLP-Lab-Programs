
# Program 18: Topic Modeling using Latent Semantic Analysis (LSA)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Sample documents
documents = [
    "Machine learning is a branch of artificial intelligence.",
    "Deep learning uses neural networks for AI.",
    "Football and cricket are popular sports.",
    "Players practice sports to improve performance.",
    "Natural language processing helps computers understand text.",
    "Text analysis is important in natural language processing."
]

# Convert documents into TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(documents)

# Apply LSA using Truncated SVD
lsa_model = TruncatedSVD(n_components=2, random_state=42)
lsa_matrix = lsa_model.fit_transform(tfidf_matrix)

# Display topics
feature_names = vectorizer.get_feature_names_out()

print("========== LSA TOPIC MODELING ==========")

for topic_index, topic in enumerate(lsa_model.components_):
    top_words = topic.argsort()[-5:][::-1]
    words = [feature_names[i] for i in top_words]

    print("\nTopic", topic_index + 1, ":", ", ".join(words))

# Display document-topic representation
print("\n========== DOCUMENT-TOPIC REPRESENTATION ==========")

for i, vector in enumerate(lsa_matrix):
    print("Document", i + 1, ":", vector)
    print(documents[i])

# Display explained variance
print("\nExplained Variance Ratio:")
print(lsa_model.explained_variance_ratio_)
