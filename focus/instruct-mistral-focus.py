#!/usr/bin/env python3
"""
instruction_tune_mistral_lora.py

Instruction-tune Mistral-7B with LoRA adapters.

Each example is formatted as:

[INST] Instruction & Argument [/INST]
<expected output>

Requirements:
  pip install accelerate peft transformers datasets pandas \
              huggingface_hub sentencepiece protobuf
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

BASE_MODEL    = "mistralai/Mistral-7B-Instruct-v0.3"
DATASET_REPO  = "username/your-dataset"
FILENAME      = "fsq_with_span.csv"
HUB_REPO_ID   = "username/mistral-chat-lora"

TEST_SPLIT    = 0.1
EPOCHS        = 3
BATCH_SIZE    = 4
LEARNING_RATE = 2e-4
LORA_RANK     = 8
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05

def load_dataset():
    token = os.getenv("HF_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF_HUB_TOKEN environment variable is required")
    path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=FILENAME,
        use_auth_token=token
    )
    df = pd.read_csv(path)
    return Dataset.from_pandas(df).train_test_split(TEST_SPLIT)

def make_prompt(ins, inp, out):
    return (
        "[INST] Classify the argument and extract supporting spans.\n\n"
        f"Argument:\n{ins}\n{inp} [/INST]\n{out}"
    )

def preprocess_fn(examples, tokenizer):
    prompts = [
        make_prompt(ins, inp, out)
        for ins, inp, out in zip(
            examples["instruction"],
            examples["input"],
            examples["output"],
        )
    ]
    tokenized = tokenizer(
        prompts,
        max_length=512,
        truncation=True,
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def main():
    ds = load_dataset()

    token = os.getenv("HF_HUB_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        use_fast=True,
        use_auth_token=token
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.float16,
        use_auth_token=token,
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    tokenized_ds = ds.map(
        lambda ex: preprocess_fn(ex, tokenizer),
        batched=True,
        batch_size=16,
        remove_columns=ds["train"].column_names,
    )

    training_args = TrainingArguments(
        output_dir="./mistral_chat_lora",
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        fp16=torch.cuda.is_available(),
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=1,
        push_to_hub=True,
        hub_model_id=HUB_REPO_ID,
        hub_strategy="end",
        hub_token=token,
        logging_dir="./mistral_chat_lora/logs",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    model.push_to_hub(HUB_REPO_ID)
    tokenizer.push_to_hub(HUB_REPO_ID)

if __name__ == "__main__":
    main()