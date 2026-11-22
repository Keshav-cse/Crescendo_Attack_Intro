import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./llama-3.2-3b-instruct"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

print("Model loaded.")
print("Type 'exit' to quit.\n")

messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant."
    }
]

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    print(f"\nAssistant: {response}\n")

    messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )