#!/usr/bin/env python3
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Ensure your Hugging Face token is set:
HF_TOKEN = os.getenv("HF_HUB_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Set the HF_HUB_TOKEN environment variable")

# Replace with your model and adapter repo IDs
BASE_MODEL    = "meta-llama/Llama-2-7b-hf"
ADAPTER_REPO  = "username/adapter-repo"

# Load base model and attach LoRA adapters
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.float16,
    use_auth_token=HF_TOKEN
)
model = PeftModel.from_pretrained(base, ADAPTER_REPO, use_auth_token=HF_TOKEN)
model.eval()

# Load tokenizer from the adapter repo
tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_REPO,
    use_fast=False,
    use_auth_token=HF_TOKEN
)

def classify_and_extract(prompt: str,
                         model: torch.nn.Module,
                         tokenizer,
                         max_new_tokens: int = 64) -> str:
    instruction = (
        "<|system|>\n"
        "You are a Socratic‐Questioning assistant.\n"
        "Given an Argument, select exactly two focus types and extract a supporting span for each.\n"
        "Respond only in this exact format:\n\n"
        "Classification:\n"
        "[\"type1\",\"type2\"]\n"
        "Span:\n"
        "[\"span1\",\"span2\"]\n"
        "<|end|>\n"
        "<|user|>\n"
        f"Argument:\n{prompt}\n"
        "<|end|>\n"
        "<|assistant|>\n"
    )

    inputs = tokenizer(instruction, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Example usage:
    argument = "Your argument text here."
    result = classify_and_extract(argument, model, tokenizer)
    print(result)