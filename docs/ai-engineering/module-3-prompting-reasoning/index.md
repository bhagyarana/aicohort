---
sidebar_position: 4
title: "Module 3: Prompting & Reasoning Systems"
description: Turn prompting from guesswork into engineering — zero-shot, few-shot, CoT, structured outputs, tool calling, and prompt injection defense.
---

# Module 3: Prompting & Reasoning Systems

Most developers treat prompting as an art: try something, see if it works, adjust by instinct. This module treats it as engineering. You will learn *why* each prompting technique works, *when* to use it, and *how* to measure whether it is actually helping. By the end, you will have a systematic toolkit — from simple zero-shot classification to multi-step reasoning with tool calls and injection-resistant RAG pipelines.

## What You'll Learn

- The difference between zero-shot, few-shot, and instruction prompting — and when each wins
- How to design instructions with specificity, constraints, and output format specifications
- How to use system vs user messages correctly and why the distinction matters
- How to extract structured outputs reliably: JSON mode, function calling, Pydantic validation
- Why Chain of Thought (CoT) improves accuracy on multi-step reasoning tasks
- Self-consistency sampling and Tree of Thought for high-stakes decisions
- How tool/function calling works mechanically — schema design and result injection
- What prompt injection is and how to defend against it

## Prerequisites

- [Module 1: LLM Fundamentals](/learn/ai-engineering/module-1-llm-fundamentals) — especially context windows, sampling, and the role of system messages
- An API key for any LLM provider (OpenAI or Anthropic recommended for structured output exercises)
- Python with the `openai` package installed

## Time Estimate

~3.5 hours

## Module Structure

| Page | What's covered |
|------|----------------|
| [Overview](./overview) | Zero-shot vs few-shot vs CoT, system/user message roles, instruction design, structured outputs (JSON/Pydantic), self-consistency, Tree of Thought, tool calling, and prompt injection — with code, tables, and callouts |
| [Hands-On](./hands-on) | Empirical prompting comparison, Pydantic extraction pipeline, a working tool-calling loop, and building + defending against prompt injection |
| [Resources](./resources) | Foundational papers (CoT, self-consistency, ToT, Constitutional AI), tools, docs, and videos |

---

**Ready to start? →** [Overview: Prompting & Reasoning Systems](./overview)
