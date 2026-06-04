from urllib import response

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("./llama-3.2-3b-instruct")
response = tok("hello world", return_tensors="pt")
print(response)
print(tok("hello world"))