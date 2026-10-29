
!pip install -q transformers torch

from transformers import pipeline
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)
text = """
Artificial Intelligence is transforming many industries.
It is used in healthcare, education, transportation,
and finance. AI systems can analyze large amounts of data,
automate repetitive tasks, and support decision-making.
Machine learning and deep learning are important parts
of AI. However, AI systems must be developed responsibly
to ensure fairness, privacy, and reliability.
"""
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
