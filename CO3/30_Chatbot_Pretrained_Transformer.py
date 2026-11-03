!pip install -q transformers torch
from transformers import pipeline
chatbot = pipeline(
    "text-generation",
    model="microsoft/DialoGPT-small"
)

print("Chatbot is ready!")
print("Type 'exit' to stop the conversation.")
while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    response = chatbot(
        user_input,
        max_new_tokens=50,
        pad_token_id=50256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    generated_text = response[0]["generated_text"]

    chatbot_reply = generated_text[len(user_input):].strip()

    print("Chatbot:", chatbot_reply)
