import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk import pos_tag

sentence = "The quick brown fox jumps over the lazy dog."
words = word_tokenize(sentence)

pos_tags = pos_tag(words)
print("========== GIVEN SENTENCE ==========")
print(sentence)

print("\n========== POS TAGGING ==========")

for word, tag in pos_tags:
    print(word, "->", tag)
