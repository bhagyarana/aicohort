---
sidebar_position: 1
title: "Module 9: Evaluation, Safety & Reliability"
description: Know if your AI system is actually working — and catch it when it's not. Covers LLM evaluation frameworks, hallucination detection, prompt injection, guardrails, and production safety patterns.
---

# Module 9: Evaluation, Safety & Reliability

Most AI engineering courses end at deployment. This one doesn't.

Shipping an AI system is day one. Knowing whether it's working correctly on day thirty — and on day three hundred — requires evaluation infrastructure that most teams build too late, after something has gone wrong in production.

This module covers the hard problems: how do you measure quality when there's no single correct answer? How do you detect hallucinations before users do? How do you prevent attackers from hijacking your system through the user input? How do you build guardrails that block bad outputs without over-blocking good ones?

## What You'll Learn

- Why evaluating AI systems is fundamentally harder than evaluating traditional software
- Offline vs online evaluation: when to use each and what each misses
- Standard benchmark suites (MMLU, HumanEval, TruthfulQA) — what they measure and their limits
- Task-specific metrics: RAGAs for RAG systems, F1 for classification, ROUGE/BLEU for generation
- LLM-as-judge: using a strong model to evaluate another model's output — scalable but biased
- Hallucination detection: factual grounding checks, citation verification, confidence calibration
- Prompt injection: how attacks work, real examples, and how to defend against them
- Data leakage: how models memorize and reproduce training data, including PII
- Guardrails: input validation, output filtering, Nemo Guardrails, Guardrails AI
- Safe tool usage: principle of least privilege, audit logs, confirm-before-act patterns
- Building an evaluation harness that runs on every code change

## Prerequisites

- [Module 4: RAG Systems](/learn/ai-engineering/module-4-rag-systems) — Exercise 1 runs RAGAs evaluation on a RAG pipeline
- [Module 8: Agents & System Design](/learn/ai-engineering/module-8-agents-system-design) — Exercise 3 tests prompt injection against a tool-calling agent
- Python with `ragas`, `openai`, and `guardrails-ai` installable via pip

## Time Estimate

~4.5 hours

## Module Structure

| Page | What's covered |
|------|----------------|
| [Overview](./overview) | Evaluation frameworks (offline/online, RAGAs, LLM-as-judge), hallucination detection, prompt injection attack patterns and defenses, data leakage, guardrail architecture, and safe tool usage |
| [Hands-On](./hands-on) | Run RAGAs on a RAG pipeline, build an LLM-as-judge evaluator, craft and defend against 5 prompt injection attacks, set up a CI evaluation harness |
| [Resources](./resources) | Papers (TruthfulQA, RAGAs, Constitutional AI), framework docs (Guardrails AI, Nemo Guardrails), and curated reading on AI safety in production |

---

**Ready to start? →** [Overview: Evaluation, Safety & Reliability](./overview)
