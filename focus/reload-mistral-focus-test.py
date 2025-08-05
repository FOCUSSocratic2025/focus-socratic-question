#!/usr/bin/env python3
"""
Socratic Question focus classification using a LoRA-fine-tuned Mistral adapter.

Requirements:
  pip install torch transformers peft pandas datasets

Ensure HF_HUB_TOKEN is set in your environment.
"""
import os
import re
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load your Hugging Face token
HF_TOKEN = os.getenv("HF_HUB_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_HUB_TOKEN environment variable is required")

# Use the Mistral-7B-Instruct base model
BASE_MODEL   = "mistralai/Mistral-7B-Instruct-v0.3"
# Replace with your adapter repository
ADAPTER_REPO = "<adapter-repo-id>"
# Path to your one-shot prompt file
PROMPT_PATH  = "path/to/one_shot_prompt.txt"

# Load base Mistral model and attach LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.float16,
    use_auth_token=HF_TOKEN
)
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_REPO,
    use_auth_token=HF_TOKEN
)
model.eval()

# Initialize tokenizer for Mistral
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    use_fast=False,
    use_auth_token=HF_TOKEN
)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

# Read one-shot prompt and demonstration
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    one_shot = f.read()
sys_block, example_block = one_shot.split("🔧 Example", 1)
sys_block = sys_block.strip()

# Extract the demonstration argument, labels, and spans
demo_arg = re.search(
    r"Argument:\s*((?:.|\n)*?)\n\s*Classification:",
    example_block
).group(1).strip()
demo_cls = re.search(r"Classification:\s*(\[[^\]]+\])", example_block).group(1)
demo_spn = re.search(r"Span:\s*(\[[^\]]+\])", example_block).group(1)

MAX_LEN = getattr(model.config, "max_position_embeddings", 4096)

def classify_and_extract(argument: str):
    prompt = (
        f"{tokenizer.bos_token or '<s>'}\n"
        f"<|system|>\n{sys_block}\n<|end|>\n\n"
        f"<|user|>\nArgument:\n{demo_arg}\n"
        f"<|assistant|>\nClassification: {demo_cls}\nSpan: {demo_spn}\n<|end|>\n\n"
        f"<|user|>\nArgument:\n{argument.strip()}\n\n"
        "Output only two JSON arrays (Classification & Span), nothing else.\n"
        "<|assistant|>\n"
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        max_length=MAX_LEN
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
    norm = raw.replace('“','"').replace('”','"')

    cls_list = re.findall(r'Classification:\s*(\[[^\]]*\])', norm)
    spn_list = re.findall(r'Span:\s*(\[[^\]]*\])', norm)
    last_cls, last_spn = cls_list[-1], spn_list[-1]

    extract_items = lambda s: [m[0] or m[1] for m in re.findall(r"'([^']*)'|\"([^\"]*)\"", s)]
    return extract_items(last_cls), extract_items(last_spn)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Classify a CSV of arguments")
    parser.add_argument("input_csv", help="Path to a CSV with an 'Argument' column")
    parser.add_argument("output_csv", help="Path to save the output CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if "Argument" not in df.columns:
        raise KeyError("CSV must contain an 'Argument' column")

    results = df["Argument"].apply(classify_and_extract)
    preds, spans = zip(*results)

    pad2 = lambda lst: (lst + ["", ""])[:2]
    p1, p2 = zip(*(pad2(p) for p in preds))
    s1, s2 = zip(*(pad2(s) for s in spans))

    df["Predicted_Type1"]     = p1
    df["Predicted_Type2"]     = p2
    df["Span_PredictedType1"] = s1
    df["Span_PredictedType2"] = s2

    df.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(df)} rows to {args.output_csv}")