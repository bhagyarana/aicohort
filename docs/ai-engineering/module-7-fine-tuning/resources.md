---
sidebar_position: 4
title: "Resources"
description: Papers, tools, and further reading on fine-tuning, LoRA, QLoRA, DPO, and model adaptation.
---

# Resources: Fine-Tuning & Adaptation

## Papers & Research

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al. (2021). The paper that introduced LoRA. Remarkably clear for a research paper — the math in Section 4 is worth reading directly; the rank decomposition idea is explained with good intuition.

- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Dettmers et al. (2023). Introduces NF4 quantization and the insight that you can combine 4-bit base weights with full-precision LoRA adapters. Made 7B+ fine-tuning accessible on consumer hardware.

- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) — Rafailov et al. (2023). Shows that you can derive a training objective that implicitly optimizes a reward model without ever training one explicitly. More practical than RLHF for most teams.

- [InstructGPT: Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — Ouyang et al. (2022). The paper behind ChatGPT's alignment. Describes the RLHF pipeline (SFT → reward model → PPO) clearly. Useful context for understanding why DPO was developed as a simpler alternative.

- [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) — Muennighoff et al. (2023). Studies what happens when you repeat data across epochs during fine-tuning. Key finding: quality matters more than quantity, and repeating high-quality data is often better than adding low-quality data.

- [The Unreasonable Ineffectiveness of the Deeper Layers](https://arxiv.org/abs/2403.17887) — Gromov et al. (2024). Shows that many layers in large models are nearly redundant — has implications for layer selection in LoRA and model pruning.

- [Phi-3 Technical Report](https://arxiv.org/abs/2404.14219) — Microsoft (2024). Documents how Phi-3 achieves remarkable performance at 3.8B parameters through data curation, not just scale. Essential reading if you're considering SLMs.

## Official Documentation

- [HuggingFace PEFT](https://huggingface.co/docs/peft) — The standard library for LoRA, QLoRA, adapters, and other PEFT methods. Complete examples for most model families.

- [HuggingFace TRL (Transformer Reinforcement Learning)](https://huggingface.co/docs/trl) — SFTTrainer, DPOTrainer, RewardTrainer. The practical library for running SFT and DPO pipelines. The `SFTTrainer` is the starting point for most fine-tuning projects.

- [HuggingFace Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) — The base training loop. Understanding this before using SFTTrainer helps when you need to customize training.

- [bitsandbytes](https://huggingface.co/docs/bitsandbytes) — The quantization library that enables QLoRA. Documents `load_in_4bit`, `BitsAndBytesConfig`, and `prepare_model_for_kbit_training`.

- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) — Configuration-driven fine-tuning framework. Define your entire training run in a YAML file. The fastest way to go from idea to trained model for standard fine-tuning tasks.

- [Unsloth](https://github.com/unslothai/unsloth) — 2× faster, 70% less memory fine-tuning for Llama, Mistral, Phi. Drop-in replacement for HuggingFace training — start here if raw training speed matters.

- [OpenAI Fine-tuning Guide](https://platform.openai.com/docs/guides/fine-tuning) — If you want to fine-tune GPT-4o-mini or GPT-3.5-turbo via API. Handles the infrastructure; you just provide the dataset in JSONL format.

## Videos & Courses

- [Fine-Tuning LLMs with LoRA — Sebastian Raschka](https://www.youtube.com/watch?v=eC6Hd1hFvos) — Raschka is the clearest explainer of ML fundamentals. This video covers LoRA math and implementation in ~45 minutes. Watch before reading the paper.

- [QLoRA Fine-Tuning in Practice — Tim Dettmers](https://www.youtube.com/watch?v=TPcXVJ1VSRI) — The author of bitsandbytes and QLoRA explaining the practical details. Covers NF4, double quantization, and the memory math.

- [DPO Explained — Yannic Kilcher](https://www.youtube.com/watch?v=hvGa5Mba4c8) — Paper walkthrough with good intuition on why DPO is equivalent to RLHF. Covers the mathematical reformulation clearly.

- [LLM Fine-Tuning Masterclass — Maxime Labonne](https://www.youtube.com/watch?v=pK4pkpnhbAE) — End-to-end: dataset preparation, SFT with QLoRA, DPO alignment, merging adapters, quantizing for deployment. One of the best practical walkthroughs available.

## Tools to Know

| Tool | What it does | When to use |
|------|-------------|-------------|
| HuggingFace PEFT | LoRA, QLoRA, adapters implementation | Standard starting point for all PEFT |
| HuggingFace TRL | SFTTrainer, DPOTrainer — fine-tuning pipelines | SFT and DPO workflows |
| Axolotl | Config-driven fine-tuning (YAML) | Standardizing fine-tuning runs, reproducibility |
| Unsloth | 2× faster training, 70% less memory | When training speed matters |
| bitsandbytes | INT8/INT4 quantization for training | QLoRA base quantization |
| Weights & Biases | Experiment tracking, loss curves, eval metrics | Any training run you want to reproduce |
| LM Evaluation Harness | Standard benchmark evaluation (MMLU, etc.) | Measuring catastrophic forgetting |
| OpenAI Fine-tuning API | Managed fine-tuning for GPT-3.5/4o-mini | When you want no infrastructure overhead |
| Together AI | Managed fine-tuning for open models (Llama, Mistral) | Open model fine-tuning without your own GPU |

## What to Read Next

- [Module 8: Agents & System Design](/learn/ai-engineering/module-8-agents-system-design) — Fine-tuned models are often deployed inside agents. Module 8 covers how to build the surrounding orchestration logic.
- [Module 9: Evaluation & Safety](/learn/ai-engineering/module-9-evaluation-safety) — Systematic evaluation of fine-tuned models requires more than accuracy metrics. Module 9 covers RAGAs, LLM-as-judge, hallucination detection, and safety evaluation.
- [Module 11: Production AI Systems](/learn/ai-engineering/module-11-production-systems) — How to serve fine-tuned models efficiently in production, including adapter merging, quantization post-fine-tune, and cost analysis.
