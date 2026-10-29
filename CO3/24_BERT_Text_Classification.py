import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
texts = [
    "I love this movie",
    "The film was excellent",
    "This product is wonderful",
    "I really enjoyed the story",
    "The experience was amazing",
    "I hate this movie",
    "The film was terrible",
    "This product is awful",
    "I disliked the story",
    "The experience was disappointing"
]
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
encodings = tokenizer(
    texts,
    truncation=True,
    padding=True,
    max_length=64
)
class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {
            key: torch.tensor(value[index])
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[index])
        return item
indices = list(range(len(texts)))
train_indices, test_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

train_encodings = {
    key: [values[i] for i in train_indices]
    for key, values in encodings.items()
}
test_encodings = {
    key: [values[i] for i in test_indices]
    for key, values in encodings.items()
}

train_labels = [labels[i] for i in train_indices]
test_labels = [labels[i] for i in test_indices]

train_dataset = ReviewDataset(train_encodings, train_labels)
test_dataset = ReviewDataset(test_encodings, test_labels)

# Load BERT classification model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)
def compute_metrics(eval_pred):
    logits, true_labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(true_labels, predictions)
    }
training_args = TrainingArguments(
    output_dir="./bert_results",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=1,
    report_to="none"
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

print("========== FINE-TUNING BERT ==========")
trainer.train()

print("\n========== MODEL EVALUATION ==========")
print(trainer.evaluate())
new_text = "This movie was fantastic!"
inputs = tokenizer(
    new_text,
    return_tensors="pt",
    truncation=True,
    padding=True
)

model.eval()
with torch.no_grad():
    outputs = model(**inputs)

prediction = torch.argmax(outputs.logits, dim=1).item()

print("\nReview:", new_text)
print("Predicted sentiment:", "Positive" if prediction == 1 else "Negative")
