import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
text = """
Natural Language Processing is a field of Artificial Intelligence.
It helps computers understand and process human language.
NLP is used in applications such as chatbots, translation,
sentiment analysis, and information retrieval.
"""
words = word_tokenize(text)

stop_words = set(stopwords.words("english"))
filtered_words = []

for word in words:
    if word.lower() not in stop_words:
        filtered_words.append(word)

print("========== ORIGINAL DOCUMENT ==========")
print(text)

print("\n========== WORDS BEFORE STOP WORD REMOVAL ==========")
print(words)

print("\n========== WORDS AFTER STOP WORD REMOVAL ==========")
print(filtered_words)

filtered_text = " ".join(filtered_words)

print("\n========== FINAL DOCUMENT ==========")
print(filtered_text)
