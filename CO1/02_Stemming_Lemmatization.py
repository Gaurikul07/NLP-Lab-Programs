
# Program 2: Stemming and Lemmatization

import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("omw-1.4")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

text = "The students are studying hard. They studied different topics and are running towards their goals."

words = word_tokenize(text)

# Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in words]

print("========== STEMMING ==========")
print("Original Words:")
print(words)
print("\nStemmed Words:")
print(stemmed_words)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

print("\n========== LEMMATIZATION ==========")
print("Original Words:")
print(words)
print("\nLemmatized Words:")
print(lemmatized_words)
