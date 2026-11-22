from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


LLAMA_MODEL = Path("./llama-3.2-3b-instruct")


class LLMWrapper:
    def __init__(
        self,
        model_dir: str = str(LLAMA_MODEL),
        max_new_tokens: int = 200,
        temperature: float = 0.7,
    ):
        self.model_dir = model_dir
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.max_input_length = min(2048, getattr(self.tokenizer, "model_max_length", 2048))
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.model.eval()

    def generate(self, messages: list[dict]) -> str:
        prompt = self._format_messages(messages)
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_input_length,
            return_tensors="pt",
        )
        device = torch.device(self.device)
        encoding = {name: tensor.to(device) for name, tensor in encoding.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=encoding["input_ids"],
                attention_mask=encoding.get("attention_mask"),
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        prompt_length = encoding["input_ids"].shape[-1]
        generated_ids = output_ids[0][prompt_length:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def _format_messages(self, messages: list[dict]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return "\n".join(parts)
