
# Tokenization of Sentences and Words using NLTK and spaCy

import nltk

# Download required NLTK data
nltk.download("punkt")
nltk.download("punkt_tab")

from nltk.tokenize import sent_tokenize, word_tokenize

text = "Natural Language Processing is a branch of Artificial Intelligence. It helps computers understand human language."

print("========== NLTK ==========")

# Sentence Tokenization
sentences_nltk = sent_tokenize(text)

print("\nSentence Tokenization:")
for sentence in sentences_nltk:
    print(sentence)

# Word Tokenization
words_nltk = word_tokenize(text)

print("\nWord Tokenization:")
print(words_nltk)


# ---------------- spaCy ----------------
import spacy

# Load English language model
nlp = spacy.load("en_core_web_sm")

doc = nlp(text)

print("\n========== spaCy ==========")

# Sentence Tokenization
print("\nSentence Tokenization:")
for sentence in doc.sents:
    print(sentence.text)

# Word Tokenization
print("\nWord Tokenization:")
print([token.text for token in doc])
