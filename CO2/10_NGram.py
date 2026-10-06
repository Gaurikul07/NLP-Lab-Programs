from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")

text = "Natural Language Processing is a branch of Artificial Intelligence"
words = word_tokenize(text.lower())
unigrams = list(ngrams(words, 1))
bigrams = list(ngrams(words, 2))
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
