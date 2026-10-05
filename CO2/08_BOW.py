from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "Natural Language Processing is interesting.",
    "Natural Language Processing is useful.",
    "Machine Learning is interesting and useful."
]

vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(documents)

print("========== VOCABULARY ==========")
print(vectorizer.get_feature_names_out())

print("\n========== BAG-OF-WORDS MATRIX ==========")
print(bow_matrix.toarray())

print("\n========== DOCUMENT REPRESENTATION ==========")

for i, vector in enumerate(bow_matrix.toarray()):
    print("Document", i + 1, ":", vector)
