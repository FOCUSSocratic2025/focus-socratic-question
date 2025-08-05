#!/usr/bin/env python3
"""
instruction_tune_llama2_chat_lora.py

Instruction-tune Llama-2-Chat 7B with LoRA adapters.

Each example is formatted as:

Instruction:
  <your task description>

Argument:
  <instruction text>
  <input text>

Response:
  <expected output>

Requirements:
  pip install accelerate peft transformers datasets pandas \
              huggingface_hub sentencepiece protobuf

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

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
BASE_MODEL   = "meta-llama/Llama-2-7b-chat-hf"  # pretrained Llama-2-Chat
DATASET_REPO = "your-username/your-dataset"     # HF dataset repo
FILENAME     = "fsq_with_span.csv"             # filename in the dataset repo
HUB_REPO_ID  = "your-username/llama2-chat-lora"  # HF model repo to push to

TEST_SPLIT   = 0.1
EPOCHS       = 3
BATCH_SIZE   = 4
LEARNING_RATE= 2e-4
LORA_RANK    = 8
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05

def load_dataset():
    token = os.getenv("HF_HUB_TOKEN")
    if not token:
        raise RuntimeError("Please set the HF_HUB_TOKEN environment variable")
    path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=FILENAME,
        use_auth_token=token
    )
    df = pd.read_csv(path)
    return Dataset.from_pandas(df).train_test_split(TEST_SPLIT)

def preprocess_fn(examples, tokenizer):
    prompts = []
    for ins, inp, out in zip(
        examples["instruction"],
        examples["input"],
        examples["output"]
    ):
        prompt = (
            "Instruction:\n"
            "Classify the argument and extract supporting spans.\n\n"
            "Argument:\n"
            f"{ins}\n{inp}\n\n"
            "Response:\n"
            f"{out}\n"
        )
        prompts.append(prompt)

    tokenized = tokenizer(
        prompts,
        max_length=512,
        truncation=True,
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

if __name__ == "__main__":
    dataset = load_dataset()

    # Initialize tokenizer and model
    hf_token = os.getenv("HF_HUB_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        use_fast=False,
        use_auth_token=hf_token
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.float16,
        use_auth_token=hf_token
    )
    model.config.use_cache = False

    # Attach LoRA adapters
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    # Preprocess dataset
    tokenized = dataset.map(
        lambda ex: preprocess_fn(ex, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./llama2_chat_lora",
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
        hub_token=hf_token,
        logging_dir="./llama2_chat_lora/logs",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train and push to the Hub
    trainer.train()
    model.push_to_hub(HUB_REPO_ID)
    tokenizer.push_to_hub(HUB_REPO_ID)
    print(f"✅ LoRA adapters pushed to: {HUB_REPO_ID}")