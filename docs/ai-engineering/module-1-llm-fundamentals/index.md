---
sidebar_position: 1
title: "Module 1: LLM Fundamentals"
description: Build the correct mental model of what an LLM is — tokens, sampling, context windows, and the full generation pipeline with real numbers.
---

# Module 1: LLM Fundamentals

Most developers use LLMs as black boxes: put text in, get text out. This module tears open the box. You will learn exactly what happens between your prompt and the response — token by token — so that when something goes wrong (and it will), you know where to look. This mental model is the prerequisite for everything else in this track.

## What You'll Learn

- Why LLMs predict text rather than "understand" it — and why this distinction matters
- How tokenization works (BPE), what it costs, and when it surprises you
- What pretraining teaches a model and what it fundamentally cannot know
- How context windows work, what happens at their boundary, and how to work within them
- The complete pipeline: prompt → tokens → embeddings → attention → logits → sampling → output
- How to control output randomness with temperature, top-k, and top-p

## Prerequisites

- [Module 0: Foundations](/learn/ai-engineering/module-0-foundations) or equivalent comfort with Python, NumPy, and softmax
- An API key for any LLM provider (OpenAI, Anthropic, Google, or Groq)

## Time Estimate

~3 hours

## Module Structure

| Page | What's covered |
|------|----------------|
| [Overview](./overview) | Autoregressive generation, tokenization deep dive, context windows, sampling strategies — with diagrams and numbers |
| [Hands-On](./hands-on) | tiktoken exercises, temperature experiments, context limit probing, token budget tracker |
| [Resources](./resources) | Papers, tools, and visualizers worth knowing |

---

**Ready to start? →** [Overview: LLM Fundamentals Deep Dive](./overview)
