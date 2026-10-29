
# Program 28: Machine Translation using MarianMT

from transformers import MarianMTModel, MarianTokenizer

# Load pretrained English-to-French translation model
model_name = "Helsinki-NLP/opus-mt-en-fr"

tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Input English sentences
texts = [
    "Hello, how are you?",
    "Artificial intelligence is changing the world.",
    "I am learning natural language processing."
]

# Tokenize input sentences
inputs = tokenizer(
    texts,
    return_tensors="pt",
    padding=True,
    truncation=True
)

# Translate sentences
translated = model.generate(
    **inputs,
    max_new_tokens=80
)

# Convert token IDs into French text
translations = tokenizer.batch_decode(
    translated,
    skip_special_tokens=True
)

# Display results
for original, translation in zip(texts, translations):
    print("English:", original)
    print("French:", translation)
    print()
