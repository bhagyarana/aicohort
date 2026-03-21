---
sidebar_position: 4
title: "Resources"
description: Foundational papers, tools, and videos for understanding transformer architecture and LLM internals.
---

# Resources: Transformer & LLM Internals

---

## Papers & Research

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) — the paper that introduced the transformer architecture; Sections 3.2 (Attention) and 3.3 (Multi-Head Attention) are directly relevant to this module's overview; essential reading
- [Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361) — establishes the power-law relationship between model size, data, compute, and loss; the foundation of modern LLM training strategy
- [Training Compute-Optimal Large Language Models / Chinchilla (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556) — revised the Kaplan scaling laws; shows the compute-optimal ratio is ~20 tokens per parameter; fundamentally changed how models like Llama 2/3 are trained
- [RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)](https://arxiv.org/abs/2104.09864) — introduces RoPE; the positional embedding scheme used in Llama, Mistral, Qwen, and most modern open-source models
- [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135) — reorders attention computation to minimize GPU memory reads; the implementation underlying fast inference in vLLM, HuggingFace, and most production systems
- [FlashAttention-2 (Dao, 2023)](https://arxiv.org/abs/2307.08691) — 2× faster than the original; the current standard for efficient attention on modern GPUs

---

## Official Documentation

- [HuggingFace Transformers — Model Documentation](https://huggingface.co/docs/transformers/index) — covers every architectural variant (BERT, GPT-2, Llama, Mistral, etc.); see the "Model Internals" section for attention implementation details
- [HuggingFace Transformers — Generation Utilities](https://huggingface.co/docs/transformers/main_classes/text_generation) — explains `use_cache`, `past_key_values`, and generation configuration; directly relevant to Exercise 3
- [BertViz GitHub (Jesse Vig)](https://github.com/jessevig/bertviz) — the attention visualization tool used in Exercise 2; documentation explains the `head_view` and `model_view` modes
- [TransformerLens Documentation](https://transformerlensorg.github.io/TransformerLens/) — a research library for mechanistic interpretability; lets you hook into specific attention heads and residual stream activations; useful for going deeper after this module

---

## Videos & Courses

- [3Blue1Brown: Attention in Transformers, Visually Explained](https://www.youtube.com/watch?v=eMlx5fFNoYc) — the clearest visual walkthrough of Q/K/V attention; highly recommended before attempting Exercise 4; ~26 minutes
- [Andrej Karpathy: Let's Build GPT from Scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) — builds a fully working character-level GPT in PyTorch from scratch (~2 hours); the most concrete implementation walkthrough of everything in this module; watch after doing Exercise 4
- [Yannic Kilcher: Attention Is All You Need (Paper Walkthrough)](https://www.youtube.com/watch?v=iDulhoQ2pro) — dense but thorough; covers the full paper including encoder-decoder structure, multi-head attention, and the original positional encoding; ~1.5 hours
- [Andrej Karpathy: Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) — broader context for how pretraining, RLHF, and scaling fit together; covers scaling laws section of this module; ~1 hour

---

## Tools to Know

| Tool | What it does | When to use |
|------|-------------|-------------|
| BertViz | Interactive attention head visualization for BERT/GPT models | Understanding what individual heads learn; debugging attention patterns |
| HuggingFace Transformers | Load, run, and fine-tune open-source transformer models | Production inference, fine-tuning, benchmarking |
| TransformerLens | Hook into activations, attention heads, residual stream for mechanistic interpretability | Research-level introspection; understanding *why* a model produces specific outputs |
| LM Studio | Run open-source models (Llama, Mistral, Phi) locally with a GUI | Experimenting with KV cache settings, context lengths, without API costs |
| vLLM | High-throughput inference server with PagedAttention (efficient KV cache management) | Production deployment; the most important inference optimization library as of 2025 |
| `torch.cuda.memory_summary()` | Print detailed GPU memory usage including KV cache | Profiling memory during inference; understanding cache growth in Exercise 3 |

---

## Useful Reference Numbers

```
Architecture quick-reference:

Model             Layers  Heads  d_model  d_head  Context   Parameters
─────────────────────────────────────────────────────────────────────
GPT-2               12     12     768       64      1K         117M
GPT-2 XL            48     25    1600       64      1K         1.5B
GPT-3 175B          96     96   12288      128      4K         175B
Llama 3 8B          32     32    4096      128    128K         8B
Llama 3 70B         80     64    8192      128    128K         70B
Mistral 7B          32     32    4096      128     32K         7B

KV cache sizes (fp16, per request):
  Llama 3 8B  @ 4K tokens  ≈  0.5 GB
  Llama 3 8B  @ 32K tokens ≈  4 GB
  Llama 3 70B @ 4K tokens  ≈  4 GB
  Llama 3 70B @ 32K tokens ≈  32 GB
```

---

## What to Read Next

→ **[Module 3: Prompting & Reasoning Systems](/learn/ai-engineering/module-3-prompting-reasoning)** — Now that you understand the architecture, Module 3 builds on it to turn prompting from guesswork into engineering: zero-shot vs few-shot, Chain of Thought, structured outputs, tool calling, and prompt injection defense.
