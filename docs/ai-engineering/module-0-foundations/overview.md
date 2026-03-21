---
sidebar_position: 2
title: "Overview"
description: Python for ML, async APIs, data handling, and the applied math behind AI systems.
---

# Foundations — Deep Dive

Every AI engineering failure mode traces back to one of four things: writing Python that is too slow for ML workloads, not knowing how async actually works, mishandling data at boundaries, or misreading a probability or vector as a magic number. This module closes those gaps.

---

## 1. Python for ML

### NumPy Arrays vs Python Lists

Python lists are flexible but slow. NumPy arrays are homogeneous, contiguous in memory, and vectorized — operations run in C, not Python.

```python
import numpy as np

# Python list: loop in Python interpreter
scores = [0.1, 0.4, 0.9, 0.2]
normalized = [x / sum(scores) for x in scores]  # slow at scale

# NumPy: loop in C
scores_np = np.array([0.1, 0.4, 0.9, 0.2])
normalized_np = scores_np / scores_np.sum()       # fast, vectorized
```

**Where it breaks without this:** Embedding 10,000 documents with a Python list loop takes minutes. The same operation with NumPy batch ops takes seconds.

### Generators for Memory-Efficient Pipelines

Loading an entire dataset into memory crashes on large files. Generators yield one item at a time:

```python
def read_jsonl(filepath):
    """Yields one record at a time — never loads the whole file."""
    with open(filepath) as f:
        for line in f:
            yield json.loads(line)

def batch(iterable, size=32):
    """Groups items into fixed-size batches."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch  # don't forget the last partial batch

# Usage: stream a 1M-line file without loading it
for chunk in batch(read_jsonl("data.jsonl"), size=64):
    embed(chunk)  # process 64 records at a time
```

### Key NumPy Operations for AI Work

| Operation | Code | What it computes |
|-----------|------|-----------------|
| Dot product | `np.dot(a, b)` | Similarity between two vectors |
| Norm | `np.linalg.norm(v)` | Magnitude/length of a vector |
| Stack | `np.stack(arrays)` | Combine embeddings into a matrix |
| Argmax | `np.argmax(logits)` | Index of highest-probability token |
| Softmax | `np.exp(x) / np.exp(x).sum()` | Convert logits to probabilities |

:::tip
`np.einsum` and broadcasting eliminate most explicit loops in ML code. When you find yourself writing `for i in range(len(array))`, there is almost always a vectorized alternative.
:::

---

## 2. APIs & Async

### The HTTP Request/Response Cycle

Every LLM API call is an HTTP POST:

```
Your code                  LLM API server
    │                           │
    │──── POST /v1/chat ────────►│
    │     headers: Authorization │
    │     body: {model, messages}│
    │                           │ ← model runs inference
    │◄─── 200 OK ───────────────│
    │     body: {choices, usage} │
```

Rate limits live in response headers: `x-ratelimit-remaining-requests`, `retry-after`. Always check them.

### `async`/`await` — The Mental Model

Synchronous code blocks the thread while waiting. Async code **suspends** the coroutine and lets the event loop run other work:

```python
import asyncio
import httpx

# Synchronous — waits on each call before starting the next
def get_embeddings_sync(texts):
    results = []
    for text in texts:
        r = httpx.post(url, json={"input": text})  # blocks here
        results.append(r.json())
    return results

# Asynchronous — all requests in flight simultaneously
async def get_embeddings_async(texts):
    async with httpx.AsyncClient() as client:
        tasks = [client.post(url, json={"input": text}) for text in texts]
        responses = await asyncio.gather(*tasks)
    return [r.json() for r in responses]

# 100 texts: sync ≈ 100 × latency, async ≈ 1 × latency (roughly)
```

### Rate Limiting and Retry Logic

APIs enforce requests-per-minute and tokens-per-minute limits. Naive parallel calls will hit 429 errors:

```python
import asyncio
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60)
)
async def call_api_with_retry(client, payload):
    response = await client.post("/v1/chat/completions", json=payload)
    if response.status_code == 429:
        retry_after = int(response.headers.get("retry-after", 5))
        await asyncio.sleep(retry_after)
        raise Exception("Rate limited")
    response.raise_for_status()
    return response.json()
```

:::note
The `tenacity` library handles exponential backoff correctly. Don't write retry loops by hand — off-by-one errors in sleep durations compound badly under load.
:::

---

## 3. Data Handling

### JSON Parsing Patterns

LLM APIs return JSON. Two failure modes to handle:

```python
import json

# Failure 1: model returns markdown-wrapped JSON
raw = '```json\n{"answer": "Paris"}\n```'
clean = raw.strip().removeprefix("```json").removesuffix("```").strip()
data = json.loads(clean)

# Failure 2: model returns invalid JSON (e.g. trailing commas)
# Use a lenient parser for recovery
import json5  # pip install json5
data = json5.loads(raw)
```

### Streaming Responses

For long outputs, stream token-by-token instead of waiting for the full response:

```python
async def stream_completion(client, prompt):
    async with client.stream("POST", "/v1/chat/completions",
                             json={"stream": True, "messages": [...]}) as r:
        async for line in r.aiter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line[6:])
                delta = chunk["choices"][0]["delta"].get("content", "")
                print(delta, end="", flush=True)
                yield delta
```

**Where it breaks without this:** A 2,000-token response at 30 tokens/sec takes ~67 seconds without streaming. With streaming, the user sees the first token in under a second.

---

## 4. Applied Math

### Vectors — The Geometric Intuition

A vector is a point in space. For AI, each word or chunk of text becomes a vector in a high-dimensional space (e.g., 1536 dimensions for OpenAI embeddings).

```
2D example:

"cat"  → [0.9, 0.2]   ●  ← close together
"dog"  → [0.8, 0.3]   ●
"car"  → [0.1, 0.9]         ●  ← far away
```

**Cosine similarity** measures the angle between vectors — not the distance. Two vectors pointing in the same direction are maximally similar (score = 1.0), regardless of their magnitude.

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

cat  = np.array([0.9, 0.2])
dog  = np.array([0.8, 0.3])
car  = np.array([0.1, 0.9])

print(cosine_similarity(cat, dog))  # 0.992 — very similar
print(cosine_similarity(cat, car))  # 0.407 — not similar
```

### Dot Product — The Geometric Meaning

`a · b = |a| × |b| × cos(θ)` — it's highest when vectors point in the same direction and both have large magnitudes. This is why normalized vectors (unit vectors) make cosine similarity and dot product equivalent.

### Probability as Confidence Scores

Softmax converts raw model outputs (logits) into a probability distribution that sums to 1:

```python
import numpy as np

def softmax(logits):
    exp = np.exp(logits - logits.max())  # subtract max for numerical stability
    return exp / exp.sum()

logits = np.array([2.1, 1.0, 0.5, -0.3])
probs  = softmax(logits)
# → [0.563, 0.207, 0.125, 0.054]  (sums to ~1.0)

# The model is 56.3% confident in the first token
# Temperature scaling: divide logits by T before softmax
# T < 1 → sharper (more confident), T > 1 → flatter (more random)
```

:::tip Mental Model
Think of softmax probabilities as the model's "vote distribution" across all possible next tokens. Temperature controls how decisive the vote is.
:::

---

## Mental Model

Think of AI engineering as **plumbing**: data flows in (JSON, text, files), transforms happen (embedding, inference, parsing), and results flow out (structured responses, vectors, tokens). Every bug is either a **leak** (data lost or corrupted at a boundary) or a **blockage** (synchronous code waiting when it should be async). These foundations are the pipes — everything else builds on top.

---

## Common Mistakes

| Mistake | Why it happens | Fix |
|---------|----------------|-----|
| Loading entire dataset into RAM | Forgot generators exist | Use `yield`-based pipelines |
| Sequential API calls for batches | Not knowing `asyncio.gather` | Async with semaphore for concurrency control |
| Ignoring `retry-after` header on 429 | Assuming fixed backoff is enough | Read the header, sleep exactly that long |
| Using `==` to compare floats | Floating point imprecision | Use `np.isclose()` or a tolerance |
| Forgetting the last partial batch | Off-by-one in batch logic | Always flush remaining items after the loop |

---

## Quiz

> **Q: You have a list of 10,000 text chunks to embed. Your API allows 500 requests per minute. What is the right architecture?**
>
> <details><summary>Show Answer</summary>
>
> Use an **async pipeline with a semaphore** to cap concurrency, plus **exponential backoff** on 429s. Do not use a synchronous loop (too slow) and do not fire all 10,000 requests simultaneously (will hit rate limits immediately). A semaphore of ~20 concurrent requests with retry logic is a good starting point. Adjust based on the actual RPM limit.
>
> ```python
> sem = asyncio.Semaphore(20)
>
> async def embed_with_limit(client, text):
>     async with sem:
>         return await call_api_with_retry(client, {"input": text})
>
> async def embed_all(texts):
>     async with httpx.AsyncClient() as client:
>         tasks = [embed_with_limit(client, t) for t in texts]
>         return await asyncio.gather(*tasks)
> ```
>
> </details>

---

## Summary Table

| Concept | What it is | When to use |
|---------|-----------|-------------|
| NumPy array | C-backed typed array | Any numerical computation at scale |
| Generator | Lazy iterator (`yield`) | Large files, streaming data |
| `asyncio.gather` | Run coroutines concurrently | Parallel API calls |
| Semaphore | Concurrency limiter | Rate-limited API calls |
| Cosine similarity | Angle between vectors | Semantic similarity in embeddings |
| Softmax | Logits → probabilities | Reading model confidence scores |

---

## Next Steps

→ [Hands-On: Foundations Exercises](./hands-on)
