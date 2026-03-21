---
sidebar_position: 3
title: "Module 2: Transformer & LLM Internals"
description: Understand why LLMs behave the way they do — embeddings, attention, KV cache, and scaling laws — enough to debug and optimize.
---

# Module 2: Transformer & LLM Internals

Module 1 showed you what an LLM does: it predicts the next token. This module explains *how* — and more importantly, *why* it behaves the way it does in production. You will trace data through the transformer architecture piece by piece: from raw integer token IDs to the final probability distribution. By the end, phrases like "KV cache miss", "attention head saturation", and "Chinchilla-optimal training run" will have precise, mechanical meanings you can reason about.

## What You'll Learn

- How embeddings turn tokens into geometric vectors in high-dimensional space
- The self-attention mechanism step by step: Q, K, V matrices, scores, softmax, weighted sum
- Why attention replaced recurrent networks (RNNs/LSTMs) for language modeling
- What feed-forward layers add on top of attention
- The KV cache: what it caches, why it exists, what invalidates it
- Positional embeddings: absolute vs rotary (RoPE) and why position matters
- Scaling laws: why bigger models + more data = predictably better performance

## Prerequisites

- [Module 1: LLM Fundamentals](/learn/ai-engineering/module-1-llm-fundamentals) — you need to understand tokens, context windows, and autoregressive generation
- Basic linear algebra: matrix multiplication, dot products, vectors
- Python and NumPy (all exercises use NumPy; no GPU required for Exercises 1, 4)

## Time Estimate

~4 hours

## Module Structure

| Page | What's covered |
|------|----------------|
| [Overview](./overview) | Embeddings, self-attention (Q/K/V), feed-forward layers, the full transformer block, KV cache, positional embeddings, and scaling laws — with diagrams, code, and tables |
| [Hands-On](./hands-on) | Word embedding geometry, attention visualization with BertViz, KV cache benchmarking, and building scaled dot-product attention from scratch in NumPy |
| [Resources](./resources) | Foundational papers (Attention Is All You Need, Chinchilla, RoPE, FlashAttention), tools, and videos |

---

**Ready to start? →** [Overview: Transformer & LLM Internals](./overview)
