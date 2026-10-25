from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

documents = [
    "Machine learning and artificial intelligence are growing fields.",
    "Deep learning uses neural networks for AI applications.",
    "Football and cricket are popular sports.",
    "Players practice sports to improve their performance.",
    "Natural language processing helps computers understand text.",
    "Text analysis and language models are important in NLP."
]
vectorizer = CountVectorizer(stop_words="english")
doc_term_matrix = vectorizer.fit_transform(documents)
lda_model = LatentDirichletAllocation(
    n_components=3,
    random_state=42,
    max_iter=10
)

lda_model.fit(doc_term_matrix)
feature_names = vectorizer.get_feature_names_out()

print("========== LDA TOPIC MODELING ==========")

for topic_index, topic in enumerate(lda_model.components_):
    top_words = topic.argsort()[-5:][::-1]
    words = [feature_names[i] for i in top_words]

    print("\nTopic", topic_index + 1, ":", ", ".join(words))
print("\n========== DOCUMENT TOPICS ==========")

document_topics = lda_model.transform(doc_term_matrix)

for i, topic_distribution in enumerate(document_topics):
    dominant_topic = topic_distribution.argmax() + 1
    print("Document", i + 1, "-> Topic", dominant_topic)
    print(documents[i])
