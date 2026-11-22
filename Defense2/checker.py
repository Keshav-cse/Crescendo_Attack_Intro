from transformers import pipeline

chat = pipeline(
    "text-generation",
    model="./llama-3.2-3b-instruct"
)

while True:
    prompt = input("You: ")
    response = chat(prompt, max_new_tokens=200)
    print("Llama:", response[0]["generated_text"])