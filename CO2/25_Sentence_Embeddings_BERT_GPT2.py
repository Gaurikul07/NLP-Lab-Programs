import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
sentences = [
    "Natural Language Processing is interesting.",
    "Machine learning helps computers learn.",
    "I enjoy studying artificial intelligence."
]
def generate_embeddings(model_name, sentences):
    print("\nLoading model:", model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    model.eval()
    inputs = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64
    )
    with torch.no_grad():
        outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).float()

    summed_embeddings = (token_embeddings * mask).sum(dim=1)
    token_counts = mask.sum(dim=1).clamp(min=1)

    sentence_embeddings = summed_embeddings / token_counts

    return sentence_embeddings.numpy()
bert_embeddings = generate_embeddings(
    "bert-base-uncased", sentences
)

print("\n========== BERT SENTENCE EMBEDDINGS ==========")
print("Embedding shape:", bert_embeddings.shape)
print(bert_embeddings)
gpt2_embeddings = generate_embeddings(
    "openai-community/gpt2", sentences
)

print("\n========== GPT-2 SENTENCE EMBEDDINGS ==========")
print("Embedding shape:", gpt2_embeddings.shape)
print(gpt2_embeddings)

print("\nEmbedding rows correspond to the input sentences.")
