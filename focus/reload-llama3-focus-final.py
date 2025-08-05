#!/usr/bin/env python3
"""
Run span‐grounded Socratic Question focus classification on a CSV.

Requirements:
  pip install torch transformers peft datasets pandas huggingface_hub
Environment variable:
  HF_HUB_TOKEN must be set to your Hugging Face token.
"""
import os
import re
import torch
import pandas as pd
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Configuration
HF_TOKEN     = os.getenv("HF_HUB_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_HUB_TOKEN environment variable is required")

BASE_MODEL   = "meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTER_REPO = "username/llama3-chat-lora"
PROMPT_PATH  = "Socratic_Questioning_Integrated_Prompt_1shot.txt"

# Load the one‐shot system prompt and example
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    prompt_text = f.read()
sys_block, example_block = prompt_text.split("🔧 Example", 1)
sys_block = sys_block.strip()

# Extract the demo argument, classification, and spans
m_arg = re.search(r"Argument:\s*((?:.|\n)*?)\n\s*Classification:", example_block)
m_cls = re.search(r"Classification:\s*(\[[^\]]+\])", example_block)
m_spn = re.search(r"Span:\s*(\[[^\]]+\])", example_block)
if not (m_arg and m_cls and m_spn):
    raise RuntimeError("Failed to parse demo from prompt file")

demo_arg = m_arg.group(1).strip()
demo_cls = m_cls.group(1)
demo_spn = m_spn.group(1)

# Load model + adapter
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.float16,
    use_auth_token=HF_TOKEN
)
model = PeftModel.from_pretrained(base, ADAPTER_REPO, use_auth_token=HF_TOKEN)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_REPO,
    use_fast=False,
    use_auth_token=HF_TOKEN
)

# Maximum input length
max_len = getattr(model.config, "max_position_embeddings", 4096)

def classify_and_extract(argument: str):
    prompt = (
        (tokenizer.bos_token or "<s>") + "\n"
        + "<|system|>\n" + sys_block + "\n<|end|>\n\n"
        + "<|user|>\nArgument:\n" + demo_arg + "\n"
        + "<|assistant|>\n"
        + f"Classification: {demo_cls}\n"
        + f"Span:           {demo_spn}\n"
        + "<|end|>\n\n"
        + "<|user|>\nArgument:\n" + argument.strip() + "\n\n"
        + "Please output only two JSON arrays (Classification & Span), nothing else.\n"
        + "<|assistant|>\n"
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        max_length=max_len
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
    norm = raw.replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'")

    cls_list = re.findall(r'Classification:\s*(\[[^\]]*\])', norm)
    spn_list = re.findall(r'Span:\s*(\[[^\]]*\])', norm)
    if not cls_list or not spn_list:
        raise ValueError(f"Unexpected model output:\n{raw}")

    def extract_items(list_str):
        pairs = re.findall(r"'([^']*)'|\"([^\"]*)\"", list_str)
        return [a or b for a, b in pairs]

    classification = extract_items(cls_list[-1])
    spans          = extract_items(spn_list[-1])
    return classification, spans

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FSQ focus classification on CSV")
    parser.add_argument("input_csv", help="CSV with an 'Argument' column")
    parser.add_argument("output_csv", help="Path for output CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if "Argument" not in df.columns:
        raise KeyError("Input CSV must have an 'Argument' column")

    results = df["Argument"].apply(classify_and_extract).tolist()
    preds, spans = zip(*results)

    def pad2(lst):
        return (lst + ["", ""])[:2]

    p1, p2 = zip(*(pad2(p) for p in preds))
    s1, s2 = zip(*(pad2(s) for s in spans))

    df["Predicted_Type1"]     = p1
    df["Predicted_Type2"]     = p2
    df["Span_PredictedType1"] = s1
    df["Span_PredictedType2"] = s2

    df.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(df)} rows to {args.output_csv}")