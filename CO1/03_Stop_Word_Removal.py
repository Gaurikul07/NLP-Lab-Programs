# Program 3: Stop Word Removal from a Document

import nltk

# Download required resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Sample document
text = """
Natural Language Processing is a field of Artificial Intelligence.
It helps computers understand and process human language.
NLP is used in applications such as chatbots, translation,
sentiment analysis, and information retrieval.
"""

# Tokenize the document
words = word_tokenize(text)

# Get English stop words
stop_words = set(stopwords.words("english"))

# Remove stop words
filtered_words = []

for word in words:
    if word.lower() not in stop_words:
        filtered_words.append(word)

# Display results
print("========== ORIGINAL DOCUMENT ==========")
print(text)

print("\n========== WORDS BEFORE STOP WORD REMOVAL ==========")
print(words)

print("\n========== WORDS AFTER STOP WORD REMOVAL ==========")
print(filtered_words)

# Create the final document
filtered_text = " ".join(filtered_words)

print("\n========== FINAL DOCUMENT ==========")
print(filtered_text)
