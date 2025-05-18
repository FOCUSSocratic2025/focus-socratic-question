# focus-socratic-question
Dataset and baseline code for “FOCUS: A Benchmark for Identifying the Focus of Socratic Questions” (EMNLP 2025 submission).
# FOCUS: A Benchmark for Identifying the Focus of Socratic Questions

This repository contains the dataset and code accompanying the EMNLP 2025 submission:  
**"FOCUS: A Benchmark for Identifying the Focus of Socratic Questions"**

## 📘 Dataset
FOCUS is a benchmark of 140 annotated arguments with span-level labels indicating the *focus* of Socratic questions. Each entry includes:
- The argument text
- One or two FSQ labels
- Span(s) used to generate the questions
- Ground-truth Socratic questions

All data is released under CC BY-NC 4.0.

## 🧠 FSQ Types
The dataset includes 11 fine-grained Socratic subtypes (e.g., Lacks Evidence, Temporal Contrast). See `annotation_guidelines.pdf` for details.

## 🧪 Baseline Code
Includes scripts for:
- GPT-3.5 / GPT-4 prompting
- Span-level evaluation (BERTScore, Jaccard)
- Annotation agreement

## 🚀 Quickstart
```bash
pip install -r requirements.txt
python scripts/evaluate_metrics.py --input data/focus_dataset.jsonl
