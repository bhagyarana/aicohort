---
sidebar_position: 3
title: "Hands-On"
description: Four exercises covering data pipelines, async API calls, cosine similarity, and softmax interpretation.
---

# Hands-On: Foundations

Work through these exercises in order. Each one targets a specific skill gap that will slow you down in every subsequent module.

---

## Exercise 1: JSONL Data Pipeline (Beginner)

**Goal:** Build a memory-efficient pipeline that reads a large JSONL file and processes records in batches.
**Time:** ~20 min

### Setup

Create a sample JSONL file to work with:

```python
import json
import random

# Generate sample data
records = [
    {"id": i, "text": f"Sample document {i}", "score": random.random()}
    for i in range(10_000)
]

with open("sample.jsonl", "w") as f:
    for record in records:
        f.write(json.dumps(record) + "\n")
```

### Step 1 — Stream records from disk

```python
import json

def read_jsonl(filepath: str):
    """Yields one record at a time. Never loads the whole file."""
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
```

### Step 2 — Group into batches

```python
from typing import Generator, TypeVar

T = TypeVar("T")

def batch(iterable, size: int = 32) -> Generator:
    """Groups items into fixed-size batches. Yields the last partial batch."""
    buf = []
    for item in iterable:
        buf.append(item)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf
```

### Step 3 — Process with a mock embedding function

```python
import time

def fake_embed(texts: list[str]) -> list[list[float]]:
    """Simulates an embedding API call with a small delay."""
    time.sleep(0.01)
    return [[0.1, 0.2, 0.3] for _ in texts]  # dummy 3-dim embeddings

def run_pipeline(filepath: str, batch_size: int = 64):
    records_processed = 0
    for chunk in batch(read_jsonl(filepath), size=batch_size):
        texts = [r["text"] for r in chunk]
        embeddings = fake_embed(texts)
        records_processed += len(chunk)

    print(f"Processed {records_processed} records")

run_pipeline("sample.jsonl")
```

### What to verify

- Memory usage stays flat (use `tracemalloc` or Activity Monitor — should not scale with file size)
- The last batch is processed even if `10000 % batch_size != 0`
- No `IndexError` or silent data loss

### Challenge

Modify the pipeline to write results to an output JSONL file as each batch completes (streaming output, not buffering everything first).

---

## Exercise 2: Async API Caller with Rate Limiting (Intermediate)

**Goal:** Write an async function that calls a (mock) API for a list of inputs, respects a concurrency cap, and retries on failure.
**Time:** ~30 min

### Setup

```bash
pip install httpx tenacity
```

### Step 1 — Fake API server (for local testing)

```python
# fake_server.py — run this in a separate terminal: python fake_server.py
from fastapi import FastAPI
import uvicorn, asyncio, random

app = FastAPI()
request_count = 0

@app.post("/embed")
async def embed(payload: dict):
    global request_count
    request_count += 1
    await asyncio.sleep(0.05)  # simulate latency
    if random.random() < 0.1:  # 10% failure rate
        from fastapi import HTTPException
        raise HTTPException(status_code=429, detail="rate limited")
    return {"embedding": [random.random() for _ in range(8)]}

if __name__ == "__main__":
    uvicorn.run(app, port=8000)
```

### Step 2 — Async client with retry and semaphore

```python
import asyncio
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

API_URL = "http://localhost:8000/embed"
MAX_CONCURRENT = 10  # semaphore cap

sem = asyncio.Semaphore(MAX_CONCURRENT)

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    reraise=True,
)
async def call_embed(client: httpx.AsyncClient, text: str) -> list[float]:
    async with sem:
        response = await client.post(API_URL, json={"text": text})
        response.raise_for_status()
        return response.json()["embedding"]

async def embed_all(texts: list[str]) -> list[list[float]]:
    async with httpx.AsyncClient(timeout=30) as client:
        tasks = [call_embed(client, t) for t in texts]
        return await asyncio.gather(*tasks)

# Run it
texts = [f"document {i}" for i in range(100)]
results = asyncio.run(embed_all(texts))
print(f"Got {len(results)} embeddings")
```

### What to verify

- All 100 embeddings are returned even with the 10% failure rate
- At most `MAX_CONCURRENT` requests are in flight at any moment (add print statements to confirm)
- Retry delay grows exponentially (check the tenacity logs)

### Challenge

Add a **token bucket rate limiter** that allows at most 500 requests per minute regardless of concurrency.

---

## Exercise 3: Cosine Similarity from Scratch (Beginner)

**Goal:** Implement cosine similarity without using any library functions beyond NumPy array math.
**Time:** ~20 min

### The math (no black boxes)

```
cosine_similarity(a, b) = (a · b) / (||a|| × ||b||)

where:
  a · b   = dot product = sum of element-wise products
  ||a||   = norm = sqrt(sum of squared elements)
```

### Implementation

```python
import numpy as np

def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Manual dot product — do NOT use np.dot."""
    return float(sum(x * y for x, y in zip(a, b)))

def vector_norm(v: np.ndarray) -> float:
    """Manual L2 norm — do NOT use np.linalg.norm."""
    return (sum(x ** 2 for x in v)) ** 0.5

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Returns a value in [-1, 1]. 1 = identical direction, 0 = orthogonal, -1 = opposite."""
    dot = dot_product(a, b)
    norm_a = vector_norm(a)
    norm_b = vector_norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
```

### Test it

```python
# Semantic similarity test (imagine these are 4-dim embeddings)
food_cat  = np.array([0.9, 0.8, 0.1, 0.0])   # "cat" — close to "dog"
food_dog  = np.array([0.85, 0.75, 0.15, 0.05])
food_car  = np.array([0.0, 0.1, 0.9, 0.85])   # "car" — far from both

print(f"cat ↔ dog: {cosine_similarity(food_cat, food_dog):.4f}")  # expect > 0.99
print(f"cat ↔ car: {cosine_similarity(food_cat, food_car):.4f}")  # expect < 0.3

# Verify against NumPy
import numpy as np
def np_cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

assert abs(cosine_similarity(food_cat, food_dog) - np_cosine(food_cat, food_dog)) < 1e-9
print("Manual implementation matches NumPy ✓")
```

### What to verify

- Identical vectors return 1.0
- Orthogonal vectors return 0.0
- Opposite vectors return -1.0
- Zero vector handled without division by zero

### Challenge

Compute pairwise similarity for a matrix of 1,000 vectors. First using your loop-based function, then using `np.dot(matrix, matrix.T)`. Measure the speedup with `timeit`.

---

## Exercise 4: Softmax and Token Probabilities (Beginner)

**Goal:** Implement softmax and use it to interpret LLM logits as token probabilities.
**Time:** ~20 min

### Step 1 — Implement numerically stable softmax

```python
import numpy as np

def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Converts raw logits to probabilities.
    Subtracting max prevents overflow in np.exp.
    """
    shifted = logits - logits.max()
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum()
```

### Step 2 — Simulate a model choosing the next token

```python
# Pretend these are logits from a model predicting the next word after "The cat sat on the"
vocabulary = [" mat", " floor", " roof", " table", " ground", " rug"]
logits     = np.array([2.1, 1.4, 0.8, 0.6, 0.3, -0.2])

probs = softmax(logits)

print("Token probability distribution:")
for token, logit, prob in sorted(zip(vocabulary, logits, probs), key=lambda x: -x[2]):
    bar = "█" * int(prob * 40)
    print(f"  {token:10s}  logit={logit:5.1f}  p={prob:.4f}  {bar}")
```

### Step 3 — Explore temperature

```python
def softmax_with_temp(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature > 1 → flatter (more random). Temperature < 1 → sharper (more confident)."""
    if temperature <= 0:
        raise ValueError("Temperature must be > 0")
    return softmax(logits / temperature)

print("\nEffect of temperature on distribution:")
for temp in [0.1, 0.5, 1.0, 2.0, 5.0]:
    probs = softmax_with_temp(logits, temp)
    top_token = vocabulary[probs.argmax()]
    entropy = -np.sum(probs * np.log(probs + 1e-9))
    print(f"  T={temp:.1f}  top={top_token:8s}  entropy={entropy:.3f}")
```

### What to verify

- All probabilities are positive and sum to exactly 1.0
- Temperature = 0.01 makes the distribution near-deterministic (p ≈ 1.0 for the top token)
- Temperature = 10.0 makes the distribution nearly uniform
- Entropy increases with temperature (higher entropy = more randomness)

### Challenge

Implement **top-k sampling**: given a probability distribution, zero out all tokens except the top-k, renormalize, then sample. Compare the outputs for k=1, k=3, and k=10.

---

## Checklist

- [ ] Completed Exercise 1: JSONL pipeline processes all records without memory growth
- [ ] Completed Exercise 2: Async caller handles 100 requests with retries and concurrency cap
- [ ] Completed Exercise 3: Cosine similarity matches NumPy reference implementation
- [ ] Completed Exercise 4: Softmax sums to 1.0 and responds correctly to temperature changes
- [ ] (Optional) Completed at least one Challenge extension

---

**Next →** [Resources: Foundations](./resources)
