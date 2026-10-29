
# Program 29: Text Summarization using BART

from transformers import pipeline

# Load pretrained summarization model
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

# Input document
text = """
Artificial Intelligence is transforming many industries.
It is used in healthcare, education, transportation,
and finance. AI systems can analyze large amounts of data,
automate repetitive tasks, and support decision-making.
Machine learning and deep learning are important parts
of AI. However, AI systems must be developed responsibly
to ensure fairness, privacy, and reliability.
"""

# Generate summary
result = summarizer(
    text,
    max_length=60,
    min_length=15,
    do_sample=False
)

print("Original Text:")
print(text)

print("\nSummarized Text:")
print(result[0]["summary_text"])
