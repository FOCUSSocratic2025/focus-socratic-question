****Repository: focus-socratic-question
****
This repository hosts the FOCUS benchmark and associated code, data, and documentation for targeted Socratic question generation via source‑span grounding.

Current Paper

This repository contains the dataset, code, and materials accompanying the ACL 2025 paper:

"FOCUS: A Benchmark for Targeted Socratic Question Generation via Source‑Span Grounding"

**📘 Dataset**

FOCUS comprises 440 annotated arguments with span‑level labels indicating the focus of Socratic questions. Each entry includes:

The argument text

One or two FSQ labels

The source span(s) used for question grounding

Ground‑truth Socratic questions

**Data splits:**

Development set: 140 instances

Test set: 300 instances

All data is released under CC BY‑NC 4.0. See data/README.md for file details and formats.

**🧠 FSQ Types
**
The dataset includes 11 fine‑grained Socratic subtypes:

Other Stakeholder Perspective, Temporal Contrast, Vague/Ambiguous Term, Overgeneralized Statement, Implicit Existence, Bias and Subjectivity, Questionable Cause–Effect, Causality Flipped
, Lack of Evidence, Weak Evidence, None of the Above

See annotation_guidelines.pdf for full definitions and exemplars.


