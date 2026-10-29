import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "openai-community/gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
custom_texts = [
    "Natural Language Processing helps computers understand human language.",
    "Machine learning enables computers to learn from data.",
    "Artificial intelligence is transforming modern technology."
]

print("========== CUSTOM DATASET ==========")
for text in custom_texts:
    print("-", text)
prompt = "Natural Language Processing"

inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(
    output[0],
    skip_special_tokens=True
)

print("\n========== GPT-2 TEXT GENERATION ==========")
print("Prompt:", prompt)
print("Generated Text:", generated_text)

print("\nNote: This example uses a pretrained GPT-2 model.")
print("The custom corpus is displayed as sample data;")
print("the model is not fine-tuned on that corpus.")
