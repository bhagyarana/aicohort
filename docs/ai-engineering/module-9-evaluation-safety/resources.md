---
sidebar_position: 4
title: "Resources"
description: Papers, frameworks, and further reading on LLM evaluation, hallucination detection, prompt injection, guardrails, and AI safety in production.
---

# Resources: Evaluation, Safety & Reliability

## Papers & Research

- [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/abs/2109.07958) — Lin et al. (2021). Benchmark measuring whether LLMs repeat common human misconceptions. The introduction makes a compelling case for why MMLU-style benchmarks miss an important quality dimension — models can score well on knowledge while still hallucinating on misconception-prone questions.

- [RAGAs: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — Es et al. (2023). The paper behind the RAGAs framework. Introduces the three-metric evaluation framework (context relevance, answer faithfulness, answer relevance) and shows strong correlation with human judgment. Essential reading before using the library.

- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — Zheng et al. (2023). The foundational study on using GPT-4 as a judge for open-ended generation evaluation. Quantifies position bias, verbosity bias, and self-preference bias. Introduces the pairwise comparison format as more reliable than scalar scoring.

- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Bai et al. / Anthropic (2022). Introduces the idea of using AI-generated feedback (based on a written "constitution" of principles) to guide model behavior. Foundational for understanding how safety and helpfulness can be aligned during training.

- [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043) — Zou et al. (2023). Shows that gradient-based suffix attacks can reliably bypass aligned models' safety training. Important for understanding the limits of instruction-following as a safety mechanism.

- [Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) — Greshake et al. (2023). The definitive paper on indirect prompt injection — attacks embedded in retrieved documents, web pages, and emails. Essential reading for anyone building RAG or agent systems.

- [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300) — Hendrycks et al. (2020). The MMLU benchmark paper. Read to understand what MMLU actually measures (57-subject multiple choice knowledge) and why high MMLU scores don't predict real-world task performance.

- [Extracting Training Data from Large Language Models](https://arxiv.org/abs/2012.07805) — Carlini et al. (2021). Demonstrates that models memorize and reproduce verbatim training data including PII. The threat model section is essential reading for anyone handling sensitive data.

## Framework Documentation

- [RAGAs](https://docs.ragas.io/) — The standard evaluation framework for RAG systems. Covers the three core metrics, evaluation dataset format, integration with LangChain, and how to use custom LLMs as the evaluator.

- [Guardrails AI](https://www.guardrailsai.com/docs) — Python library for validating LLM inputs and outputs. Has a hub of pre-built validators (PII detection, toxic language, fact-checking, JSON format compliance). The quickstart guide gets you to a working guard in ~10 minutes.

- [NVIDIA NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) — More complex than Guardrails AI but more powerful for dialog-level safety. Uses Colang (a custom language) to define conversation flows and safety rules. Better for complex multi-turn agent safety.

- [DeepEval](https://docs.confident-ai.com/) — Alternative to RAGAs with a broader metric suite (G-Eval, hallucination metric, bias metric, toxicity). Has built-in CI integration and a hosted dashboard. Worth comparing to RAGAs for your specific use case.

- [HELM (Holistic Evaluation of Language Models)](https://crfm.stanford.edu/helm/latest/) — Stanford's comprehensive evaluation framework. Very broad but complex to run. Most useful for understanding what multi-dimensional LLM evaluation looks like in practice, even if you don't run it yourself.

- [LangSmith](https://docs.smith.langchain.com/) — LangChain's evaluation and observability platform. Has built-in LLM-as-judge evaluators, dataset management, and A/B testing. Best-in-class for teams already using LangChain/LangGraph.

- [Promptfoo](https://promptfoo.dev/docs/intro) — CLI-first evaluation and red-teaming tool. Define test cases in YAML, run against any LLM, compare outputs across models or prompt versions. Excellent for CI integration without writing Python evaluation code from scratch.

## Videos & Courses

- [Evaluating and Debugging Generative AI — DeepLearning.AI](https://www.deeplearning.ai/short-courses/evaluating-debugging-generative-ai/) — Short course covering Weights & Biases Weave for tracing, RAGAs for evaluation, and LLM-as-judge setup. Well-paced introduction to the practical evaluation stack.

- [Adversarial Robustness of LLMs — NeurIPS 2023 Tutorial](https://www.youtube.com/watch?v=f9TK5FwlA24) — Academic but accessible overview of adversarial attacks on language models: prompt injection, jailbreaking, and data poisoning. Good conceptual foundation before diving into defenses.

- [Building Safe AI Products — AI Engineer Summit](https://www.youtube.com/watch?v=ZEfVvk_ZYVU) — Practitioner perspective on shipping AI safely. Covers the difference between safety research and production safety engineering, guardrail design, and incident response for AI systems.

- [RAGAs Deep Dive — Shahul ES (RAGAs author)](https://www.youtube.com/watch?v=zzLsNM5V5ig) — The creator of RAGAs explaining the design decisions behind the three metrics, common pitfalls, and how to interpret scores. Highly recommended before using the library in production.

## Tools to Know

| Tool | What it does | When to use |
|------|-------------|-------------|
| RAGAs | RAG evaluation framework (3 core metrics) | Standard starting point for RAG evaluation |
| DeepEval | Broader metric suite, CI integration | When you need metrics beyond RAGAs |
| Guardrails AI | Input/output validation + hub of validators | Adding safety checks to any LLM endpoint |
| NeMo Guardrails | Dialog-level safety with Colang | Complex multi-turn agent safety rules |
| Promptfoo | YAML-based eval + red-teaming CLI | CI evaluation without Python boilerplate |
| LangSmith | Trace + evaluate + A/B test LangChain apps | LangChain/LangGraph production monitoring |
| Helicone | LLM proxy with logging, analytics, caching | Cost tracking and latency monitoring |
| Weights & Biases Weave | Trace and evaluate AI pipelines | Experiment tracking across eval runs |

## What to Read Next

- [Module 10: Multimodal Systems](/learn/ai-engineering/module-10-multimodal) — Evaluation for multimodal systems introduces new challenges: how do you score image captioning, document parsing, or visual QA? Module 10 covers VLM-specific evaluation patterns.
- [Module 11: Production AI Systems](/learn/ai-engineering/module-11-production-systems) — Evaluation is one piece of production reliability. Module 11 covers the rest: observability, fallback chains, caching, and cost management in deployed AI systems.
