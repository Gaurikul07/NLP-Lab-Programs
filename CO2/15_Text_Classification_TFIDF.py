
# Program 15: Text Classification using Naive Bayes and SVM with TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset
texts = [
    "I love this movie",
    "This film is amazing",
    "The movie was excellent",
    "I enjoyed this film",
    "The story was wonderful",
    "I hate this movie",
    "This film is terrible",
    "The movie was boring",
    "I disliked this film",
    "The story was awful",
    "The movie was fantastic",
    "This film was bad"
]

# Labels: 1 = Positive, 0 = Negative
labels = [
    1, 1, 1, 1, 1,
    0, 0, 0, 0, 0,
    1, 0
]

# Convert text into TF-IDF features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.25, random_state=42, stratify=labels
)

# Train Naive Bayes classifier
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Train SVM classifier
svm_model = LinearSVC(random_state=42)
svm_model.fit(X_train, y_train)

# Evaluate both models
models = {
    "Naive Bayes": nb_model,
    "SVM": svm_model
}

for name, model in models.items():
    predictions = model.predict(X_test)

    print("\n==========", name, "==========")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 2))
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Negative", "Positive"],
        zero_division=0
    ))

# Predict sentiment for a new sentence
new_text = ["The movie was amazing and wonderful"]
new_vector = vectorizer.transform(new_text)

print("New sentence:", new_text[0])
print("Naive Bayes prediction:",
      "Positive" if nb_model.predict(new_vector)[0] == 1 else "Negative")
print("SVM prediction:",
      "Positive" if svm_model.predict(new_vector)[0] == 1 else "Negative")
