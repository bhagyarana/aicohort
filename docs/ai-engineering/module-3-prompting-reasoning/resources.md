---
sidebar_position: 4
title: "Resources"
description: Foundational papers, tools, docs, and videos for prompting, Chain of Thought, structured outputs, and prompt injection defense.
---

# Resources: Prompting & Reasoning Systems

---

## Papers & Research

- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022)](https://arxiv.org/abs/2201.11903) — the paper that formalized CoT; shows that adding reasoning steps dramatically improves multi-step problem-solving; includes the GSM8K benchmark results cited in the overview
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models (Wang et al., 2022)](https://arxiv.org/abs/2203.11171) — introduces the majority-vote sampling approach; demonstrates consistent gains over single CoT on math and commonsense reasoning benchmarks
- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models (Yao et al., 2023)](https://arxiv.org/abs/2305.10601) — introduces the BFS/DFS framework over thought branches; includes Game of 24 and creative writing experiments showing where ToT outperforms CoT
- [Large Language Models are Zero-Shot Reasoners (Kojima et al., 2022)](https://arxiv.org/abs/2205.11916) — establishes "Let's think step by step" as a zero-shot CoT trigger; simpler and often as effective as few-shot CoT
- [Constitutional AI: Harmlessness from AI Feedback (Anthropic, 2022)](https://arxiv.org/abs/2212.08073) — describes how Anthropic uses prompting-based critique and revision to align model behavior; highly relevant to instruction design and system prompt engineering
- [Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection (Greshake et al., 2023)](https://arxiv.org/abs/2302.12173) — the most cited academic treatment of indirect prompt injection; describes real-world attack vectors in LLM-integrated systems; required reading before deploying RAG applications

---

## Official Documentation

- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling) — the canonical reference for tool schemas, `tool_choice` options, parallel function calls, and strict mode; Exercise 3 is built directly on this
- [Anthropic Tool Use Documentation](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — Claude's equivalent; note differences in how tool results are returned (using `tool_result` content blocks instead of a `tool` role)
- [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs) — covers `response_format: json_schema` (stricter than `json_object`); enforces the exact schema at the API level using constrained decoding
- [instructor library (Python)](https://python.useinstructor.com/) — the leading library for Pydantic + LLM integration; handles schema injection, retries, and validation automatically; used in Exercise 2 as an alternative to manual JSON parsing
- [Pydantic V2 Documentation](https://docs.pydantic.dev/latest/) — reference for validators, field types, and `model_validate`; understanding Pydantic is a prerequisite for production structured-output pipelines
- [promptfoo Documentation](https://www.promptfoo.dev/docs/) — open-source prompt testing and evaluation framework; define test suites to run all your prompts against expected outputs; essential for regression testing prompt changes

---

## Videos & Courses

- [Andrej Karpathy: Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) — the second half of this 1-hour talk covers prompt engineering, in-context learning, and LLM security (including prompt injection); one of the clearest high-level treatments available
- [Simon Willison: Prompt Injection and LLM Security](https://simonwillison.net/2023/Apr/25/dual-llm-pattern/) — a series of blog posts and talks from one of the most prominent researchers tracking real-world LLM security; the "dual LLM" pattern (two-model separation) described in the overview comes from his work
- [Jason Liu: Structured Outputs with instructor](https://www.youtube.com/watch?v=yj-wSRJwrrc) — the creator of the `instructor` library walks through Pydantic-based LLM extraction patterns; covers retries, nested schemas, and streaming structured outputs

---

## Tools to Know

| Tool | What it does | When to use |
|------|-------------|-------------|
| [instructor](https://python.useinstructor.com/) | Pydantic + LLM integration: schema → prompt → validated output | Production structured extraction pipelines; replaces manual JSON parsing |
| [guidance](https://github.com/guidance-ai/guidance) | Microsoft's library for constrained LLM generation (interleave code and model calls) | Complex generation patterns where you need programmatic control mid-generation |
| [promptfoo](https://www.promptfoo.dev/) | Prompt evaluation framework: run test suites, compare prompt versions, detect regressions | CI/CD for prompts; A/B testing prompt changes before deploying to production |
| [LangSmith](https://smith.langchain.com/) | Prompt tracking, tracing, and evaluation dashboard | Debugging prompt chains in production; logging all LLM calls with inputs and outputs |
| [Braintrust](https://www.braintrustdata.com/) | LLM evaluation and prompt management platform | Scoring model outputs against expected answers; tracking prompt performance over time |
| [outlines](https://github.com/outlines-dev/outlines) | Constrained generation for open-source models (local inference) | Enforcing output schemas on Llama/Mistral without an external API |

---

## Useful Reference Patterns

```python
# The production-grade structured output pattern (no external libraries)

import json
from pydantic import BaseModel, ValidationError
from openai import OpenAI

client = OpenAI()

def extract_with_retry(
    text: str,
    schema: type[BaseModel],
    system_prompt: str,
    max_retries: int = 3,
) -> BaseModel | None:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]

    for attempt in range(max_retries):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        try:
            return schema.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as e:
            if attempt < max_retries - 1:
                messages += [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"Invalid output. Error: {e}. Please fix and return valid JSON."},
                ]

    return None
```

---

## What to Read Next

→ **[Module 4: RAG Systems](/learn/ai-engineering/module-4-rag-systems)** — With prompting fundamentals established, Module 4 applies them to Retrieval-Augmented Generation: embedding documents, building vector stores, retrieval strategies, and designing RAG pipelines that are both accurate and injection-resistant.
