import nltk
nltk.download("punkt")
nltk.download("punkt_tab")

from nltk.tokenize import sent_tokenize, word_tokenize

text = "Natural Language Processing is a branch of Artificial Intelligence. It helps computers understand human language."

print("========== NLTK ==========")

sentences_nltk = sent_tokenize(text)

print("\nSentence Tokenization:")
for sentence in sentences_nltk:
    print(sentence)

words_nltk = word_tokenize(text)

print("\nWord Tokenization:")
print(words_nltk)

import spacy

nlp = spacy.load("en_core_web_sm")

doc = nlp(text)

print("\n========== spaCy ==========")

print("\nSentence Tokenization:")
for sentence in doc.sents:
    print(sentence.text)

print("\nWord Tokenization:")
print([token.text for token in doc])
