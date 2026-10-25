from textblob import TextBlob
reviews = [
    "This mobile phone has an excellent camera and great battery life.",
    "The product quality is poor and the battery drains quickly.",
    "The customer service was helpful and friendly.",
    "Delivery was late and the packaging was damaged.",
    "The product is good and works as expected."
]

print("========== OPINION MINING ==========")

positive = 0
negative = 0
neutral = 0

for review in reviews:
    polarity = TextBlob(review).sentiment.polarity
    if polarity > 0:
        sentiment = "Positive"
        positive += 1
    elif polarity < 0:
        sentiment = "Negative"
        negative += 1
    else:
        sentiment = "Neutral"
        neutral += 1

    print("\nReview:", review)
    print("Polarity:", round(polarity, 2))
    print("Opinion:", sentiment)
print("\n========== REVIEW SUMMARY ==========")
print("Positive Reviews:", positive)
print("Negative Reviews:", negative)
print("Neutral Reviews:", neutral)
print("Total Reviews:", len(reviews))
