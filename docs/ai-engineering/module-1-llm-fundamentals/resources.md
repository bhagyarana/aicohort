---
sidebar_position: 4
title: "Resources"
description: Papers, tools, and visualizers for understanding LLM fundamentals.
---

# Resources: LLM Fundamentals

---

## Papers & Research

- [Language Models are Few-Shot Learners (Brown et al., 2020)](https://arxiv.org/abs/2005.14165) — the GPT-3 paper that introduced the idea of in-context learning; Section 2 (Approach) explains autoregressive generation clearly
- [Lost in the Middle (Liu et al., 2023)](https://arxiv.org/abs/2307.03172) — empirical evidence that models perform worse at retrieving information placed in the middle of long contexts; directly relevant to Exercise 3
- [Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361) — establishes the relationship between model size, data, compute, and performance; the basis of "bigger models learn more"
- [The Unreasonable Effectiveness of Data (Halevy et al., 2009)](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/35179.pdf) — older but foundational; explains why scale matters more than clever algorithms

---

## Official Documentation

- [tiktoken (OpenAI)](https://github.com/openai/tiktoken) — the tokenizer library used in exercises; supports cl100k_base (GPT-4), o200k_base (GPT-4o)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference) — especially the `/chat/completions` endpoint; pay attention to `usage` object and `logprobs` parameter
- [Anthropic API Reference](https://docs.anthropic.com/en/api) — Claude's API; note that `usage` field returns input/output tokens separately
- [OpenAI Tokenizer (web tool)](https://platform.openai.com/tokenizer) — paste any text and see the exact token boundaries highlighted; the fastest way to build tokenization intuition

---

## Videos & Courses

- [Andrej Karpathy: Let's Build GPT from Scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) — 2 hours; builds a working GPT character-level model from scratch in PyTorch; the most concrete explanation of autoregressive generation that exists
- [3Blue1Brown: But What Is a Neural Network?](https://www.youtube.com/watch?v=aircAruvnKk) — visual intuition for how neural networks learn; watch before the transformer series
- [3Blue1Brown: Attention in Transformers](https://www.youtube.com/watch?v=eMlx5fFNoYc) — covers the attention mechanism; directly relevant to Module 2
- [Andrej Karpathy: Tokenization Deep Dive](https://www.youtube.com/watch?v=zduSFxRajkE) — 2 hours on BPE tokenization specifically; the most thorough treatment available

---

## Tools to Know

| Tool | What it does | When to use |
|------|-------------|-------------|
| `tiktoken` | Tokenize text for OpenAI models | Cost estimation, context budget |
| `transformers` (HuggingFace) | Load and run open-source model tokenizers | When using non-OpenAI models |
| OpenAI Tokenizer (web) | Visual token inspection | Quick ad-hoc checks |
| `logprobs` API param | Get per-token probabilities from API | Debugging model confidence, building calibrated outputs |
| LM Studio | Run open-source models locally | Experimenting without API costs |
| Ollama | Local model runner (CLI-first) | Quick local inference for any supported model |

---

## Useful Mental Cheatsheet

```
1 token      ≈ 4 characters of English
1 token      ≈ ¾ of a word
100 tokens   ≈ 75 words ≈ one short paragraph
1,000 tokens ≈ 750 words ≈ a page of text
1M tokens    ≈ 750K words ≈ a full novel

Cost rule of thumb (GPT-4o, 2025):
  Input:  $2.50 / 1M tokens
  Output: $10.00 / 1M tokens
  1,000 requests × 1,000-token prompts × 200-token responses = ~$2.70
```

---

## What to Read Next

→ **[Module 2: Transformer Internals](/learn/ai-engineering/module-2-transformer-internals)** — Now that you know *what* LLMs do (predict tokens), Module 2 explains *how* — the attention mechanism, embeddings, KV cache, and why transformers replaced every previous architecture.
