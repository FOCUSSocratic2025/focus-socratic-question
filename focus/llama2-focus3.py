#!/usr/bin/env python3
"""
train_lora_llama2.py

Fine-tune Llama-2 with LoRA on ABCI or a local machine using Hugging Face Trainer.

Requirements:
  pip install accelerate peft bitsandbytes transformers datasets pandas huggingface_hub sentencepiece protobuf

Environment variable:
  HF_HUB_TOKEN must be set to your Hugging Face token.
"""

import os
import torch
import pandas as pd
from huggingface_hub import hf_hub_download
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

# Constants
BASE_MODEL = "meta-llama/Llama-2-7b-hf"
OUTPUT_DIR = "./lora_llama2"
REPO_ID    = "Pothong/fsq-lora-v2"
FILENAME   = "fsq_with_span_mistral_llama.csv"
TEST_SPLIT = 0.1

def prepare_dataset(csv_path, test_split):
    df = pd.read_csv(csv_path)
    ds = Dataset.from_pandas(df.reset_index(drop=True))
    return ds.train_test_split(test_size=test_split)

def main():
    # 1) Check for HF token
    token = os.getenv("HF_HUB_TOKEN")
    assert token, "Environment variable HF_HUB_TOKEN is not set"

    # 2) Download CSV from HF Hub
    print(f"Downloading {FILENAME} from {REPO_ID}…")
    csv_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=FILENAME,
        revision="main",
        use_auth_token=token
    )
    print("Local CSV path:", csv_path)

    # 3) Prepare dataset
    ds = prepare_dataset(csv_path, TEST_SPLIT)
    print("Train/Test sizes:", len(ds["train"]), len(ds["test"]))

    # 4) Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        use_fast=False,
        use_auth_token=token,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        use_auth_token=token,
    )
    model.config.use_cache = False  # required for LoRA training

    # 5) Attach LoRA
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 6) Preprocessing function (fixed-length padding)
    def preprocess_fn(examples):
        prompts = [
            f"{ins}\n\n{inp}\n\nResponse:"
            for ins, inp in zip(examples["instruction"], examples["input"])
        ]
        inputs = tokenizer(
            prompts,
            max_length=512,
            truncation=True,
            padding="max_length",    # pad all to length 512
        )
        labels = tokenizer(
            examples["output"],
            max_length=128,
            truncation=True,
            padding="max_length",    # pad all to length 128
        )["input_ids"]
        inputs["labels"] = labels
        return inputs

    # 7) Apply preprocessing
    ds = ds.map(
        preprocess_fn,
        batched=True,
        batch_size=16,
        remove_columns=ds["train"].column_names,
    )

    # 8) Training setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    model.to(device)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=20,
        fp16=torch.cuda.is_available(),
        save_strategy="epoch",
        save_total_limit=2,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 9) Train & save
    trainer.train()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Model and tokenizer saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()