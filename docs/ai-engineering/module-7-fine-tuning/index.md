---
sidebar_position: 1
title: "Module 7: Fine-Tuning"
description: Know when fine-tuning is the right tool — and when it's not. Covers SFT, LoRA, QLoRA, DPO, catastrophic forgetting, and how to evaluate adaptation results.
---

# Module 7: Fine-Tuning & Adaptation

Most AI engineers never need to fine-tune a model. Most AI engineers also eventually discover they needed to fine-tune a model — they just didn't recognize when they hit that moment.

Fine-tuning is not the answer to every problem, and it's not the last resort when everything else fails. It's one specific tool with a specific use case: changing *how a model behaves* (its style, format, or domain vocabulary), not what it knows. Understanding when it's the right tool — and when prompting or RAG is better — is the most important skill this module teaches.

## What You'll Learn

- The three-way decision framework: when to use prompting vs RAG vs fine-tuning
- When fine-tuning clearly wins over the alternatives, with concrete examples
- Supervised Fine-Tuning (SFT): how dataset curation determines outcome quality
- Why 500 high-quality examples beat 50,000 noisy ones
- Parameter-Efficient Fine-Tuning (PEFT): LoRA, QLoRA, and why they work
- The math behind LoRA: low-rank matrix decomposition explained simply
- QLoRA: combining 4-bit quantization with LoRA to fine-tune 7B+ models on consumer hardware
- Reward models, RLHF, and DPO: high-level understanding of preference-based training
- Catastrophic forgetting: why fine-tuning can break general capabilities, and how to check
- Evaluating fine-tuned models: task accuracy, quality degradation, and held-out test sets

## Prerequisites

- [Module 1: LLM Fundamentals](/learn/ai-engineering/module-1-llm-fundamentals) — you need to understand what pretraining produces before you can reason about adapting it
- [Module 6: Model Optimization](/learn/ai-engineering/module-6-model-optimization) — QLoRA relies on INT4 quantization; this module assumes you already understand that
- Python with `transformers`, `peft`, `datasets`, and `trl` installable via pip
- GPU access for Exercise 3 (Google Colab T4 works)

## Time Estimate

~4.5 hours

## Module Structure

| Page | What's covered |
|------|----------------|
| [Overview](./overview) | The prompting/RAG/fine-tuning decision tree, SFT dataset design, LoRA and QLoRA math, RLHF and DPO conceptually, catastrophic forgetting, and evaluation strategy |
| [Hands-On](./hands-on) | Fine-tune GPT-2 with HuggingFace Trainer, apply LoRA and compare trainable parameter counts, attempt QLoRA on a 7B model, and run before/after evaluation |
| [Resources](./resources) | Papers (LoRA, QLoRA, DPO, RLHF), HuggingFace tooling docs, and curated videos on fine-tuning in practice |

---

**Ready to start? →** [Overview: Fine-Tuning & Adaptation](./overview)
