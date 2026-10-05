# Program 4: Part-of-Speech (POS) Tagging

import nltk

# Download required resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Given sentence
sentence = "The quick brown fox jumps over the lazy dog."

# Tokenize the sentence
words = word_tokenize(sentence)

# Perform POS tagging
pos_tags = pos_tag(words)

# Display the results
print("========== GIVEN SENTENCE ==========")
print(sentence)

print("\n========== POS TAGGING ==========")

for word, tag in pos_tags:
    print(word, "->", tag)
