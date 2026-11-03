
# Program 30: Chatbot using a Pre-trained Transformer

from transformers import pipeline

# Load pretrained conversational model
chatbot = pipeline(
    "text-generation",
    model="microsoft/DialoGPT-small"
)

print("Chatbot is ready!")
print("Type 'exit' to stop the conversation.")

# Chat loop
while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    # Generate chatbot response
    response = chatbot(
        user_input,
        max_new_tokens=50,
        pad_token_id=50256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    # Display response
    generated_text = response[0]["generated_text"]

    chatbot_reply = generated_text[len(user_input):].strip()

    print("Chatbot:", chatbot_reply)
