---
sidebar_position: 1
title: "AI Engineering Track"
description: A concept-first, production-focused course covering everything from LLM fundamentals to shipping reliable AI systems at scale.
---

# AI Engineering Track

A concept-first, production-focused course for developers who want to understand *how* AI systems work — and *why* they behave the way they do.

:::info Self-Paced Track
This is a **self-paced** learning path. Work through the modules in order or jump to the topic you need. Each module is independent once you have the foundations.
:::

---

## What You'll Build Towards

By the end of this track you will be able to:

- Explain how LLMs generate text — tokens, sampling, context windows, attention
- Design and build RAG pipelines that actually retrieve the right content
- Choose between prompting, RAG, and fine-tuning for a given problem
- Build agents that use tools reliably without looping indefinitely
- Evaluate whether your AI system is actually working
- Ship AI features to production with observability, cost controls, and fallbacks

---

## Learning Roadmap

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FOUNDATIONS TIER                             │
│                                                                     │
│   Module 0          Module 1          Module 2                      │
│  Foundations  ───►  LLM          ───► Transformer                   │
│  (prereqs)          Fundamentals      Internals                     │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│                         APPLIED TIER                                │
│                                                                     │
│   Module 3          Module 4          Module 5                      │
│  Prompting &  ───►  RAG         ───►  Vector                        │
│  Reasoning          Systems           Databases                     │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│                       ADVANCED TIER                                 │
│                                                                     │
│   Module 6          Module 7          Module 8                      │
│  Model        ───►  Fine-       ───►  Agents &                      │
│  Optimization        Tuning           System Design                 │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│                      PRODUCTION TIER                                │
│                                                                     │
│   Module 9          Module 10         Module 11                     │
│  Evaluation   ───►  Multimodal  ───►  Production                    │
│  & Safety           Systems           AI Systems                    │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│                        CAPSTONE                                     │
│                                                                     │
│   Module 12: Four complete projects combining all prior modules     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## All 13 Modules at a Glance

| # | Module | What You'll Learn | Time |
|---|--------|------------------|------|
| 0 | [Foundations](/learn/ai-engineering/module-0-foundations) | Python for ML, async APIs, vectors, probability | ~2 hrs |
| 1 | [LLM Fundamentals](/learn/ai-engineering/module-1-llm-fundamentals) | Tokens, sampling, context windows, the full generation pipeline | ~3 hrs |
| 2 | [Transformer Internals](/learn/ai-engineering/module-2-transformer-internals) | Attention, embeddings, KV cache, scaling laws | ~3 hrs |
| 3 | [Prompting & Reasoning](/learn/ai-engineering/module-3-prompting-reasoning) | CoT, structured outputs, tool calling, prompt injection | ~3 hrs |
| 4 | [RAG Systems](/learn/ai-engineering/module-4-rag-systems) | Full RAG pipeline, chunking, reranking, failure modes | ~4 hrs |
| 5 | [Vector Databases](/learn/ai-engineering/module-5-vector-databases) | HNSW, hybrid search, quantization, DB selection | ~3 hrs |
| 6 | [Model Optimization](/learn/ai-engineering/module-6-model-optimization) | Quantization, distillation, streaming, speculative decoding | ~3 hrs |
| 7 | [Fine-Tuning](/learn/ai-engineering/module-7-fine-tuning) | SFT, LoRA, QLoRA, when fine-tuning beats prompting | ~4 hrs |
| 8 | [Agents & System Design](/learn/ai-engineering/module-8-agents-system-design) | ReAct, routing, memory systems, multi-agent architecture | ~4 hrs |
| 9 | [Evaluation & Safety](/learn/ai-engineering/module-9-evaluation-safety) | RAGAs, LLM-as-judge, guardrails, prompt injection defense | ~3 hrs |
| 10 | [Multimodal Systems](/learn/ai-engineering/module-10-multimodal) | VLMs, document AI, image + text agents, audio | ~3 hrs |
| 11 | [Production AI Systems](/learn/ai-engineering/module-11-production-systems) | Observability, caching, cost engineering, deployment | ~4 hrs |
| 12 | [Capstone Projects](/learn/ai-engineering/module-12-capstone) | Four complete end-to-end AI systems | ~8 hrs |

**Total: ~47 hours of structured learning**

---

## How Each Module Is Structured

Every module follows the same 4-page pattern so you always know what to expect:

| Page | Purpose |
|------|---------|
| **Index** | Module overview, prerequisites, learning goals |
| **Overview** | Theory, mental models, diagrams, comparison tables, quizzes |
| **Hands-On** | Exercises (Beginner → Intermediate → Advanced) + mini-project |
| **Resources** | Papers, documentation, tools, what to read next |

---

## Prerequisites

:::note Before You Start
You need **Module 0** if you are not already comfortable with all of:
- Python: list comprehensions, generators, decorators, type hints
- HTTP: request/response cycle, REST, JSON, headers
- Async: `async`/`await`, event loops, `asyncio.gather`
- Vectors: what a dot product is, what cosine similarity means geometrically
- Probability: conditional probability, softmax as a probability distribution
:::

If you can check all five — skip Module 0 and go straight to [Module 1](/learn/ai-engineering/module-1-llm-fundamentals).

---

## How This Differs From the Cohort Track

| | Cohort Track (Modules 1–5) | AI Engineering Track |
|---|---|---|
| **Focus** | Tool-specific (LangChain, LangGraph) | Concept-first, framework-agnostic |
| **Audience** | Cohort participants | Self-paced learners |
| **Style** | Instructor-led with live sessions | Asynchronous, self-contained |
| **Depth** | Applied and practical | Theory + applied |
| **Goal** | Build agents fast | Understand AI systems deeply |

:::tip Use Both Tracks Together
The tracks complement each other. After studying **RAG Systems** here (Module 4), go implement it hands-on in the **[Cohort Track Module 2](/learn/modules/module-2)** using LangChain. After studying **Agents** here (Module 8), apply it in **[Cohort Track Module 4](/learn/modules/module-4)** using LangGraph.
:::

---

## Start Here

Ready? Choose your entry point:

- **Start from zero →** [Module 0: Foundations](/learn/ai-engineering/module-0-foundations)
- **Have the basics →** [Module 1: LLM Fundamentals](/learn/ai-engineering/module-1-llm-fundamentals)
- **Know LLMs, want RAG →** [Module 4: RAG Systems](/learn/ai-engineering/module-4-rag-systems)
- **Ready to ship →** [Module 11: Production AI Systems](/learn/ai-engineering/module-11-production-systems)
- **See the projects →** [Module 12: Capstone](/learn/ai-engineering/module-12-capstone)
