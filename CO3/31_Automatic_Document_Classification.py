
# Program 31: Automatic Document Classification and Separation

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample documents
documents = [
    "Python programming and software development",
    "Java and coding are used in software engineering",
    "Artificial intelligence and machine learning",
    "Deep learning and neural networks",
    "Football and cricket are popular sports",
    "Players practice for sports competitions"
]

# Document categories
labels = [
    "Technology",
    "Technology",
    "AI",
    "AI",
    "Sports",
    "Sports"
]

# Convert documents into TF-IDF features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# Train classification model
model = MultinomialNB()
model.fit(X, labels)

# New documents to classify
new_documents = [
    "Machine learning uses artificial intelligence",
    "Python is a programming language",
    "Cricket is played by many people"
]

# Convert new documents into features
new_X = vectorizer.transform(new_documents)

# Predict document categories
predictions = model.predict(new_X)

# Display classification results
for document, category in zip(new_documents, predictions):
    print("Document:", document)
    print("Category:", category)
    print()
