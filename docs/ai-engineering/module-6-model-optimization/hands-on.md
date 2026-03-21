---
sidebar_position: 3
title: "Hands-On"
description: Practical exercises — benchmark model tiers, implement streaming with TTFT measurement, run INT8 quantization, and build a prompt compression function.
---

# Hands-On: Model Optimization & Efficiency

These exercises move from API-level optimizations (streaming, model selection) to local inference optimizations (quantization, prompt compression). You can do Exercises 1 and 2 with just an OpenAI API key. Exercises 3 and 4 require a local Python environment with GPU access (or Google Colab).

---

## Exercise 1: Model Tier Benchmarking — Build a Decision Matrix (Beginner)

**Goal:** Empirically compare GPT-4o vs GPT-4o-mini on latency, cost, and output quality for different task types.

**Time:** ~30 min

**Setup:**
```bash
pip install openai python-dotenv
```

**Step 1 — Define your task suite:**
```python
tasks = [
    {
        "name": "simple_extraction",
        "prompt": "Extract the company name from this text: 'We are pleased to announce that Acme Corp has secured Series B funding.'",
        "expected_contains": "Acme Corp",
    },
    {
        "name": "multi_step_reasoning",
        "prompt": "A train leaves Chicago at 9am at 60mph. Another leaves New York at 10am at 80mph. Chicago to New York is 790 miles. When do they meet, and how far from Chicago?",
        "expected_contains": None,  # evaluate manually
    },
    {
        "name": "code_generation",
        "prompt": "Write a Python function that takes a list of dicts with 'name' and 'score' keys and returns the top 3 by score, sorted descending.",
        "expected_contains": "def ",
    },
    {
        "name": "creative_writing",
        "prompt": "Write a one-paragraph product description for a smart water bottle that tracks hydration.",
        "expected_contains": None,
    },
]
```

**Step 2 — Benchmark function:**
```python
import time
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def benchmark_model(model: str, tasks: list, runs: int = 3) -> dict:
    results = {}
    for task in tasks:
        latencies = []
        outputs = []
        for _ in range(runs):
            start = time.perf_counter()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": task["prompt"]}],
                temperature=0,
            )
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)
            outputs.append(response.choices[0].message.content)

        results[task["name"]] = {
            "avg_latency_s": round(sum(latencies) / runs, 2),
            "min_latency_s": round(min(latencies), 2),
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "sample_output": outputs[-1][:200],
        }
    return results
```

**Step 3 — Run and compare:**
```python
import json

models = ["gpt-4o", "gpt-4o-mini"]
all_results = {}

for model in models:
    print(f"\nBenchmarking {model}...")
    all_results[model] = benchmark_model(model, tasks)

# Print comparison table
print(f"\n{'Task':<25} {'gpt-4o latency':>15} {'mini latency':>15} {'Speedup':>10}")
print("-" * 70)
for task in tasks:
    name = task["name"]
    t_full = all_results["gpt-4o"][name]["avg_latency_s"]
    t_mini = all_results["gpt-4o-mini"][name]["avg_latency_s"]
    speedup = round(t_full / t_mini, 1)
    print(f"{name:<25} {t_full:>14}s {t_mini:>14}s {speedup:>9}x")
```

**Step 4 — Cost estimation:**
```python
# Approximate pricing (verify current rates at platform.openai.com)
PRICING = {
    "gpt-4o":      {"input": 5.00, "output": 15.00},   # per 1M tokens
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},    # per 1M tokens
}

def estimate_monthly_cost(results, model, requests_per_day):
    total_input = sum(r["input_tokens"] for r in results[model].values())
    total_output = sum(r["output_tokens"] for r in results[model].values())
    cost_per_request = (
        total_input / 1_000_000 * PRICING[model]["input"] +
        total_output / 1_000_000 * PRICING[model]["output"]
    )
    return cost_per_request * requests_per_day * 30

for model in models:
    monthly = estimate_monthly_cost(all_results, model, requests_per_day=10_000)
    print(f"{model}: ${monthly:.2f}/month at 10K req/day")
```

**What to observe:** For simple extraction and formatting tasks, gpt-4o-mini is often indistinguishable in quality at 10× lower cost and 3× lower latency. GPT-4o's advantage is most visible on multi-step reasoning.

---

## Exercise 2: Streaming with Time-to-First-Token Measurement (Intermediate)

**Goal:** Implement streaming responses and measure TTFT vs total generation time.

**Time:** ~25 min

**Step 1 — Non-streaming baseline:**
```python
import time
from openai import OpenAI

client = OpenAI()

prompt = "Explain how transformers work in AI, covering attention mechanisms, positional embeddings, and why they replaced RNNs. Be thorough."

# Baseline: no streaming
start = time.perf_counter()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    stream=False,
)
total_time = time.perf_counter() - start
output = response.choices[0].message.content

print(f"No streaming:")
print(f"  Total time:     {total_time:.2f}s")
print(f"  Output tokens:  {response.usage.completion_tokens}")
print(f"  First char at:  {total_time:.2f}s (same as total — user waits)")
```

**Step 2 — Streaming with TTFT measurement:**
```python
def stream_with_metrics(prompt: str, model: str = "gpt-4o-mini") -> dict:
    start = time.perf_counter()
    first_token_time = None
    full_text = []
    token_count = 0

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            if first_token_time is None:
                first_token_time = time.perf_counter() - start
            full_text.append(delta)
            token_count += 1
            # Simulate real-time display
            print(delta, end="", flush=True)

    total_time = time.perf_counter() - start
    print()  # newline after streamed output

    return {
        "ttft_s": round(first_token_time, 3),
        "total_time_s": round(total_time, 2),
        "tokens_generated": token_count,
        "tpot_ms": round((total_time - first_token_time) / token_count * 1000, 1),
    }

print("\nWith streaming:")
metrics = stream_with_metrics(prompt)
print(f"\nMetrics:")
print(f"  Time to first token: {metrics['ttft_s']}s")
print(f"  Total time:          {metrics['total_time_s']}s")
print(f"  Tokens generated:    {metrics['tokens_generated']}")
print(f"  Time per token:      {metrics['tpot_ms']}ms")
```

**Step 3 — Async streaming for concurrent requests:**
```python
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI()

async def stream_request(request_id: int, prompt: str) -> dict:
    start = time.perf_counter()
    first_token_time = None
    tokens = 0

    async with async_client.chat.completions.stream(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = time.perf_counter() - start
                tokens += 1

    return {
        "id": request_id,
        "ttft_s": round(first_token_time or 0, 3),
        "total_s": round(time.perf_counter() - start, 2),
    }

async def benchmark_concurrent(n: int = 5):
    prompts = [f"Write a haiku about the number {i}" for i in range(n)]
    results = await asyncio.gather(*[stream_request(i, p) for i, p in enumerate(prompts)])
    for r in results:
        print(f"Request {r['id']}: TTFT={r['ttft_s']}s  Total={r['total_s']}s")

asyncio.run(benchmark_concurrent(5))
```

---

## Exercise 3: INT8 Quantization with bitsandbytes (Intermediate)

**Goal:** Run a local model in INT8 and compare speed, memory, and output quality vs FP16.

**Time:** ~45 min (GPU recommended — use Google Colab T4 if no local GPU)

**Setup:**
```bash
pip install transformers accelerate bitsandbytes torch
```

**Step 1 — Load model in FP16 (baseline):**
```python
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "microsoft/phi-2"  # 2.7B params — small enough for a T4 GPU

print("Loading FP16 model...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model_fp16 = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

def get_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0

fp16_memory = get_memory_mb()
print(f"FP16 GPU memory: {fp16_memory:.0f} MB")
```

**Step 2 — Load same model in INT8:**
```python
print("\nLoading INT8 model...")
model_int8 = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,       # bitsandbytes INT8
    device_map="auto",
    trust_remote_code=True,
)

int8_memory = get_memory_mb() - fp16_memory
print(f"INT8 GPU memory: {int8_memory:.0f} MB")
print(f"Memory reduction: {(1 - int8_memory/fp16_memory)*100:.1f}%")
```

**Step 3 — Benchmark inference speed:**
```python
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 100) -> tuple[str, float]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    elapsed = time.perf_counter() - start
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated, elapsed

test_prompts = [
    "Explain the concept of recursion in programming with an example:",
    "What are the main differences between SQL and NoSQL databases?",
    "Write a Python function to find all prime numbers up to N:",
]

print(f"\n{'Prompt':<50} {'FP16 time':>10} {'INT8 time':>10} {'Speedup':>8}")
print("-" * 80)

for prompt in test_prompts:
    text_fp16, t_fp16 = generate(model_fp16, tokenizer, prompt)
    text_int8, t_int8 = generate(model_int8, tokenizer, prompt)
    speedup = t_fp16 / t_int8
    short_prompt = prompt[:47] + "..."
    print(f"{short_prompt:<50} {t_fp16:>9.2f}s {t_int8:>9.2f}s {speedup:>7.1f}x")
```

**Step 4 — Quality comparison:**
```python
# Compare outputs on a reasoning task
reasoning_prompt = "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain your reasoning:"

output_fp16, _ = generate(model_fp16, tokenizer, reasoning_prompt, max_new_tokens=150)
output_int8, _ = generate(model_int8, tokenizer, reasoning_prompt, max_new_tokens=150)

print("\n=== FP16 Output ===")
print(output_fp16)
print("\n=== INT8 Output ===")
print(output_int8)
```

**What to observe:** INT8 typically delivers 30–50% memory reduction and 10–20% speedup on inference, with output quality differences ranging from imperceptible (on 7B+ models) to noticeable (on very small models like Phi-2).

---

## Exercise 4: Prompt Compression (Intermediate)

**Goal:** Build a function that reduces token count by 30%+ without measurable quality loss.

**Time:** ~35 min

**The problem:** Long system prompts are expensive. A 500-token system prompt on 100K daily requests = 50M tokens/day = $250/day at GPT-4o pricing.

**Step 1 — Measure baseline:**
```python
import tiktoken
from openai import OpenAI

client = OpenAI()
enc = tiktoken.encoding_for_model("gpt-4o")

verbose_system_prompt = """
You are a highly professional and courteous customer service representative for TechCorp,
a leading technology company that specializes in providing innovative software solutions
to businesses of all sizes across various industries worldwide.

When responding to customers, you should always:
- Greet the customer warmly and professionally at the beginning of every conversation
- Listen carefully to understand their complete issue or question before providing any response
- Provide clear, accurate, and helpful information that directly addresses their specific concern
- Maintain a positive, empathetic, and patient tone throughout the entire conversation
- If you are uncertain about any information, let the customer know and offer to find out
- Always ask if there is anything else you can help with before concluding the conversation
- Thank the customer for contacting TechCorp at the end of every interaction

You have access to information about TechCorp's products, including:
- TechCorp Enterprise Suite (our flagship business management platform)
- TechCorp Analytics Pro (data analysis and reporting tool)
- TechCorp Connect (communication and collaboration software)
- TechCorp Shield (cybersecurity and compliance solution)

Your goal is to ensure every customer leaves the conversation fully satisfied with the assistance provided.
"""

baseline_tokens = len(enc.encode(verbose_system_prompt))
print(f"Baseline token count: {baseline_tokens}")
```

**Step 2 — Rule-based compression:**
```python
import re

def compress_prompt_rules(prompt: str) -> str:
    """Apply deterministic compression rules."""
    # Remove filler phrases
    filler_patterns = [
        r'\bhighly\b ',
        r'\bvery\b ',
        r'\bfully\b ',
        r'\bentirely\b ',
        r'\bcompletely\b ',
        r'\balways\b ',  # used redundantly
        r'at the beginning of every conversation',
        r'throughout the entire conversation',
        r'before providing any response',
        r'their specific concern',
        r'of all sizes across various industries worldwide',
        r'that specializes in providing innovative software solutions to businesses',
    ]
    result = prompt
    for pattern in filler_patterns:
        result = re.sub(pattern, '', result, flags=re.IGNORECASE)

    # Collapse multiple spaces/newlines
    result = re.sub(r'  +', ' ', result)
    result = re.sub(r'\n{3,}', '\n\n', result)
    result = result.strip()
    return result

compressed_rules = compress_prompt_rules(verbose_system_prompt)
rules_tokens = len(enc.encode(compressed_rules))
print(f"After rule-based compression: {rules_tokens} tokens ({(1-rules_tokens/baseline_tokens)*100:.1f}% reduction)")
```

**Step 3 — LLM-assisted compression:**
```python
def compress_prompt_llm(prompt: str) -> str:
    """Use a cheap model to rewrite the prompt more concisely."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Rewrite the following system prompt to be as concise as possible while preserving all instructions and information. Remove redundancy, filler words, and verbose phrasing. Preserve all bullet points as short phrases. Output only the rewritten prompt, nothing else."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )
    return response.choices[0].message.content

compressed_llm = compress_prompt_llm(verbose_system_prompt)
llm_tokens = len(enc.encode(compressed_llm))
print(f"After LLM compression: {llm_tokens} tokens ({(1-llm_tokens/baseline_tokens)*100:.1f}% reduction)")
print("\nCompressed version:")
print(compressed_llm)
```

**Step 4 — Quality validation:**
```python
test_query = "I'm having trouble logging into TechCorp Enterprise Suite. It says my account is locked."

def test_quality(system_prompt: str, user_query: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        temperature=0,
    )
    return response.choices[0].message.content

print("\n=== Original prompt response ===")
print(test_quality(verbose_system_prompt, test_query))

print("\n=== LLM-compressed prompt response ===")
print(test_quality(compressed_llm, test_query))
```

**Cost calculation:**
```python
daily_requests = 100_000
monthly_savings_tokens = (baseline_tokens - llm_tokens) * daily_requests * 30
monthly_savings_dollars = monthly_savings_tokens / 1_000_000 * 5.00  # gpt-4o input price

print(f"\nMonthly savings at 100K req/day: ${monthly_savings_dollars:,.2f}")
```

---

## Mini-Project: Adaptive Model Router

**Goal:** Build a router that sends requests to the cheapest model capable of handling the task.

**Architecture:**
```
User request
    ↓
[Classifier] → estimates task complexity (simple/medium/complex)
    ↓
simple  → gpt-4o-mini
medium  → gpt-4o-mini (with more careful prompting)
complex → gpt-4o
    ↓
Response
```

**Starter code:**
```python
from openai import OpenAI
import json

client = OpenAI()

COMPLEXITY_CLASSIFIER_PROMPT = """Classify the complexity of the following user request.

Rules:
- simple: factual lookup, single-step task, formatting, extraction
- medium: multi-step reasoning with clear steps, code with known patterns
- complex: open-ended reasoning, ambiguous requirements, novel problem-solving, math proof

Respond with ONLY a JSON object: {"complexity": "simple"|"medium"|"complex", "reason": "one sentence"}

Request: {request}"""

MODEL_ROUTING = {
    "simple": "gpt-4o-mini",
    "medium": "gpt-4o-mini",
    "complex": "gpt-4o",
}

def classify_complexity(request: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # classifier always uses cheap model
        messages=[{"role": "user", "content": COMPLEXITY_CLASSIFIER_PROMPT.format(request=request)}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)

def route_and_respond(user_request: str) -> dict:
    # Step 1: Classify
    classification = classify_complexity(user_request)
    complexity = classification["complexity"]
    model = MODEL_ROUTING[complexity]

    # Step 2: Generate response with selected model
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_request}],
        temperature=0,
    )
    latency = time.perf_counter() - start

    return {
        "complexity": complexity,
        "model_used": model,
        "latency_s": round(latency, 2),
        "response": response.choices[0].message.content,
        "tokens": response.usage.total_tokens,
    }

# Test with varied requests
test_requests = [
    "What is the capital of France?",
    "Write a Python function to merge two sorted lists into one sorted list.",
    "Design an algorithm for scheduling N tasks with dependencies on M machines to minimize total completion time, considering task migration costs.",
]

import time
for req in test_requests:
    result = route_and_respond(req)
    print(f"\nRequest: {req[:60]}...")
    print(f"  Complexity: {result['complexity']} → {result['model_used']}")
    print(f"  Latency: {result['latency_s']}s | Tokens: {result['tokens']}")
```

**Extension ideas:**
- Add response caching for repeated queries
- Track routing decisions in a log to improve classifier over time
- Add a confidence threshold — if classifier is uncertain, escalate to GPT-4o

---

## Checklist

- [ ] Completed Exercise 1: benchmarked GPT-4o vs GPT-4o-mini across task types
- [ ] Completed Exercise 2: implemented streaming and measured TTFT separately from total time
- [ ] Completed Exercise 3: ran INT8 quantization and compared memory + speed vs FP16
- [ ] Completed Exercise 4: built prompt compression and validated output quality is preserved
- [ ] Built the mini-project: adaptive model router with complexity classification
