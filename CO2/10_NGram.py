
# Program 10: N-Gram Model (Uni-gram, Bi-gram, Tri-gram) Generation

from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import nltk

# Download required resources
nltk.download("punkt")
nltk.download("punkt_tab")

# Sample corpus
text = "Natural Language Processing is a branch of Artificial Intelligence"

# Tokenize the corpus
words = word_tokenize(text.lower())

# Generate Uni-grams
unigrams = list(ngrams(words, 1))

# Generate Bi-grams
bigrams = list(ngrams(words, 2))

# Generate Tri-grams
trigrams = list(ngrams(words, 3))

print("========== CORPUS ==========")
print(text)

print("\n========== UNI-GRAMS ==========")
for gram in unigrams:
    print(gram)

print("\n========== BI-GRAMS ==========")
for gram in bigrams:
    print(gram)

print("\n========== TRI-GRAMS ==========")
for gram in trigrams:
    print(gram)
