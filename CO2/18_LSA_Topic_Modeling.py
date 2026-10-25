from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
documents = [
    "Machine learning is a branch of artificial intelligence.",
    "Deep learning uses neural networks for AI.",
    "Football and cricket are popular sports.",
    "Players practice sports to improve performance.",
    "Natural language processing helps computers understand text.",
    "Text analysis is important in natural language processing."
]
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(documents)
lsa_model = TruncatedSVD(n_components=2, random_state=42)
lsa_matrix = lsa_model.fit_transform(tfidf_matrix)
feature_names = vectorizer.get_feature_names_out()

print("========== LSA TOPIC MODELING ==========")

for topic_index, topic in enumerate(lsa_model.components_):
    top_words = topic.argsort()[-5:][::-1]
    words = [feature_names[i] for i in top_words]

    print("\nTopic", topic_index + 1, ":", ", ".join(words))
print("\n========== DOCUMENT-TOPIC REPRESENTATION ==========")

for i, vector in enumerate(lsa_matrix):
    print("Document", i + 1, ":", vector)
    print(documents[i])
print("\nExplained Variance Ratio:")
print(lsa_model.explained_variance_ratio_)
