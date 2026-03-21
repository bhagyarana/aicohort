---
sidebar_position: 4
title: "Resources"
description: Python, async, NumPy, and math references for the Foundations module.
---

# Resources: Foundations

Curated references — only what is actually worth your time.

---

## Official Documentation

- [Python `asyncio` docs](https://docs.python.org/3/library/asyncio.html) — the authoritative reference; the "coroutines and tasks" page is the one to read
- [NumPy user guide](https://numpy.org/doc/stable/user/index.html) — especially the "absolute beginners" and "broadcasting" sections
- [httpx async docs](https://www.python-httpx.org/async/) — the async HTTP client used throughout this track (prefer over `aiohttp` for most use cases)
- [tenacity docs](https://tenacity.readthedocs.io/) — retry logic library; the decorator API is clean and handles most production cases

---

## Papers & Research

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) — the original Transformer paper; Section 3 (Model Architecture) is the part relevant to Module 0's math
- [Word2Vec: Efficient Estimation of Word Representations (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781) — the paper that introduced the geometric intuition of word embeddings ("king - man + woman ≈ queen")

---

## Videos & Courses

- [3Blue1Brown: Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) — the best visual explanation of vectors, dot products, and matrix operations that exists; watch chapters 1–5
- [Corey Schafer: Python Generators](https://www.youtube.com/watch?v=bD05uGo_sVI) — concise, practical walkthrough; 11 minutes well spent
- [Real Python: Async IO](https://realpython.com/async-io-python/) — the most readable written explanation of Python async; goes from event loop basics to practical patterns

---

## Tools to Know

| Tool | What it does | When to use |
|------|-------------|-------------|
| `httpx` | Async-capable HTTP client | All HTTP calls in this track |
| `tenacity` | Retry with backoff | Any API call that can fail transiently |
| `numpy` | Fast numerical arrays | Anything involving vectors, matrices, math |
| `json5` | Lenient JSON parser | Parsing model output that may have trailing commas |
| `tracemalloc` | Memory usage tracing | Verifying pipelines don't grow in memory |
| `uvicorn` + `fastapi` | Local API server | Testing async clients without a real API key |

---

## Math Primers (If You Need Them)

- [Khan Academy: Vectors](https://www.khanacademy.org/math/linear-algebra/vectors-and-spaces) — start here if dot products feel abstract; free and visual
- [Probability Cheatsheet (Harvard)](https://static1.squarespace.com/static/54bf3241e4b0f0d81bf7ff36/t/55e9494fe4b011aed10e48e5/1441352015658/probability_cheatsheet.pdf) — one-page reference for conditional probability, Bayes, and distributions

---

## What to Read Next

You have the prerequisites. Move to the first module of the track:

→ **[Module 1: LLM Fundamentals](/learn/ai-engineering/module-1-llm-fundamentals)** — How LLMs actually generate text: tokens, context windows, sampling, and the full inference pipeline.
