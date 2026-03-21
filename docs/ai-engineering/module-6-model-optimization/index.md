---
sidebar_position: 1
title: "Module 6: Model Optimization"
description: Make models fast and cheap without sacrificing quality — KV caching, quantization, distillation, Flash Attention, speculative decoding, and the SLM vs large model decision.
---

# Module 6: Model Optimization & Efficiency

Production AI systems are expensive. A naively deployed LLM will burn your budget, hit latency limits, and fail under load — not because the model is bad, but because nobody tuned how it runs. This module covers the engineering layer between "the model works" and "the model works at scale."

You'll learn how inference actually executes, where time and money are spent at each stage, and which optimizations give the biggest return. Some of these (streaming, batching) are software choices you can make today. Others (quantization, Flash Attention) require understanding how hardware and math interact.

## What You'll Learn

- How the KV Cache works, when it's invalidated, and how batch size affects it
- Static vs dynamic batching — why throughput and latency pull in opposite directions
- Streaming: how token-by-token delivery changes perceived latency even when total time is the same
- Quantization: FP32 → FP16 → BF16 → INT8 → INT4, with real size and accuracy tradeoffs
- Knowledge distillation: training a small model to mimic a large model's behavior, not just its labels
- When small language models (Phi-3, Gemma, Mistral 7B) outperform GPT-4 on specific tasks
- Mixture of Experts (MoE): why activating only a subset of parameters per token is a free lunch
- Flash Attention: memory-efficient attention via tiling and why it matters for long context
- Speculative decoding: how a draft model + verifier achieves 2–3× speedups

## Prerequisites

- [Module 1: LLM Fundamentals](/learn/ai-engineering/module-1-llm-fundamentals) — you need to understand tokens, sampling, and context windows
- [Module 2: Transformer Internals](/learn/ai-engineering/module-2-transformer-internals) — KV Cache and attention optimizations only make sense if you know what's being cached and what attention is doing
- Python with `openai`, `transformers`, and `bitsandbytes` installable via pip

## Time Estimate

~4 hours

## Module Structure

| Page | What's covered |
|------|----------------|
| [Overview](./overview) | KV Cache internals, batching tradeoffs, quantization math, distillation, SLMs, MoE, Flash Attention, and speculative decoding — with decision trees, diagrams, and cost comparisons |
| [Hands-On](./hands-on) | Compare GPT-4o-mini vs GPT-4o latency, implement streaming with time-to-first-token measurement, run INT8 quantization with bitsandbytes, and build a prompt compression function |
| [Resources](./resources) | Papers on Flash Attention, speculative decoding, quantization, official model cards for Phi-3/Gemma/Mistral, and profiling tools |

---

**Ready to start? →** [Overview: Model Optimization & Efficiency](./overview)
