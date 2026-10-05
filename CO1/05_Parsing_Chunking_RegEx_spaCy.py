import re
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")

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
grammar = r"""
    NP: {<DT>?<JJ.*>*<NN.*>+}
"""
chunk_parser = nltk.RegexpParser(grammar)

chunk_tree = chunk_parser.parse(pos_tags)

print("\n========== REGEX CHUNKING ==========")
print(chunk_tree)

print("\nNoun Phrases:")
for subtree in chunk_tree.subtrees():
    if subtree.label() == "NP":
        print(" ".join(word for word, tag in subtree.leaves()))
import spacy
nlp = spacy.load("en_core_web_sm")
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
