from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
documents = [
    "Natural Language Processing is interesting.",
    "Natural Language Processing is useful.",
    "Machine Learning is interesting and useful."
]

bow_vectorizer = CountVectorizer()

bow_matrix = bow_vectorizer.fit_transform(documents)

print("========== BAG-OF-WORDS ==========")

print("Vocabulary:")
print(bow_vectorizer.get_feature_names_out())

print("\nBoW Matrix:")
print(bow_matrix.toarray())

tfidf_vectorizer = TfidfVectorizer()

tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

print("\n========== TF-IDF ==========")

print("Vocabulary:")
print(tfidf_vectorizer.get_feature_names_out())

print("\nTF-IDF Matrix:")
print(tfidf_matrix.toarray())

print("\n========== COMPARISON: BoW vs TF-IDF ==========")

print("\nBoW represents the frequency of words:")
print(bow_matrix.toarray())

print("\nTF-IDF represents the importance of words:")
print(tfidf_matrix.toarray())
