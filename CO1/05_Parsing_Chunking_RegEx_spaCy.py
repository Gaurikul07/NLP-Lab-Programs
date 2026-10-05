# Program 5: Parsing and Chunking using RegEx and spaCy

import re
import nltk

# Download required resources
nltk.download("punkt")
nltk.download("punkt_tab")

from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Given sentence
sentence = "The quick brown fox jumps over the lazy dog."

# ==================== REGEX CHUNKING ====================

# Tokenize and POS tag the sentence
words = word_tokenize(sentence)
pos_tags = pos_tag(words)

print("========== GIVEN SENTENCE ==========")
print(sentence)

print("\n========== POS TAGGING ==========")
for word, tag in pos_tags:
    print(word, "->", tag)

# Regex pattern for noun phrase:
# Determiner + adjectives + noun
grammar = r"""
    NP: {<DT>?<JJ.*>*<NN.*>+}
"""

# Create a chunk parser
chunk_parser = nltk.RegexpParser(grammar)

# Create chunks
chunk_tree = chunk_parser.parse(pos_tags)

print("\n========== REGEX CHUNKING ==========")
print(chunk_tree)

print("\nNoun Phrases:")
for subtree in chunk_tree.subtrees():
    if subtree.label() == "NP":
        print(" ".join(word for word, tag in subtree.leaves()))


# ==================== SPACY PARSING ====================

import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Process the sentence
doc = nlp(sentence)

print("\n========== SPACY DEPENDENCY PARSING ==========")

for token in doc:
    print(
        token.text,
        "-> POS:", token.pos_,
        "| Dependency:", token.dep_,
        "| Head:", token.head.text
    )

print("\n========== SPACY CHUNKING ==========")

print("Noun Phrases:")
for chunk in doc.noun_chunks:
    print(chunk.text)
