!pip install -q scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
documents = [
    "Python programming and software development",
    "Java and coding are used in software engineering",
    "Artificial intelligence and machine learning",
    "Deep learning and neural networks",
    "Football and cricket are popular sports",
    "Players practice for sports competitions"
]
labels = [
    "Technology",
    "Technology",
    "AI",
    "AI",
    "Sports",
    "Sports"
]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)
model = MultinomialNB()
model.fit(X, labels)
new_documents = [
    "Machine learning uses artificial intelligence",
    "Python is a programming language",
    "Cricket is played by many people"
]
new_X = vectorizer.transform(new_documents)
predictions = model.predict(new_X)
for document, category in zip(new_documents, predictions):
    print("Document:", document)
    print("Category:", category)
    print()
