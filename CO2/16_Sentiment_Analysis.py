
# Program 16: Sentiment Analysis using TextBlob and VADER

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Sample sentences
sentences = [
    "I love this product. It is amazing!",
    "The service was terrible and disappointing.",
    "The product is okay, nothing special."
]

# Initialize VADER analyzer
analyzer = SentimentIntensityAnalyzer()

print("========== SENTIMENT ANALYSIS ==========")

for sentence in sentences:
    print("\nSentence:", sentence)

    # TextBlob sentiment
    blob = TextBlob(sentence)
    polarity = blob.sentiment.polarity

    if polarity > 0:
        textblob_sentiment = "Positive"
    elif polarity < 0:
        textblob_sentiment = "Negative"
    else:
        textblob_sentiment = "Neutral"

    print("TextBlob Polarity:", polarity)
    print("TextBlob Sentiment:", textblob_sentiment)

    # VADER sentiment
    scores = analyzer.polarity_scores(sentence)
    compound = scores["compound"]

    if compound >= 0.05:
        vader_sentiment = "Positive"
    elif compound <= -0.05:
        vader_sentiment = "Negative"
    else:
        vader_sentiment = "Neutral"

    print("VADER Scores:", scores)
    print("VADER Sentiment:", vader_sentiment)
