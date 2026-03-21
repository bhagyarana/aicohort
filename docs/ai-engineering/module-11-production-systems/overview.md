---
sidebar_position: 2
title: "Overview"
description: Theory and concepts — production architecture, caching, model routing, observability, cost engineering, and deployment patterns
---

# Production AI Systems — Deep Dive

A demo proves the idea. A production system proves you can engineer it.

The gap between a working prototype and a reliable product is filled with decisions most AI courses never cover: how do you handle the model API going down? How do you prevent a single user from spending $300 in one session? How do you know, three months after deployment, whether quality has silently degraded?

This module covers the systems thinking that separates AI engineers who ship from those who demo.

---

## System Architecture Layers

Production AI systems have a consistent layered structure. Understanding which layer solves which problem is the first step to designing them well.

```
┌──────────────────────────────────────────────┐
│                Client / UI                    │
│    (web app, mobile app, API consumer)        │
├──────────────────────────────────────────────┤
│            API Gateway / Auth                 │
│  rate limiting, auth, request validation,     │
│  user-level token quotas                      │
├──────────────────────────────────────────────┤
│          Orchestration Layer                  │
│  routing, chaining, agent logic,              │
│  prompt construction, response parsing        │
│  [LangChain, LangGraph, custom code]          │
├──────────────────────────────────────────────┤
│             Cache Layer                       │
│  semantic cache, response cache,              │
│  (Redis, in-memory, vector DB)                │
├──────────────────────────────────────────────┤
│            LLM API Layer                      │
│  model selection, fallbacks,                  │
│  retry logic, timeout handling                │
│  [OpenAI, Anthropic, Gemini, local models]    │
├──────────────────────────────────────────────┤
│           Retrieval Layer                     │
│  embeddings, vector search, reranking         │
│  [Qdrant, pgvector, Pinecone]                 │
├──────────────────────────────────────────────┤
│           Data / Storage                      │
│  conversation history, documents,             │
│  user data, evaluation logs                   │
│  [Postgres, Redis, S3]                        │
├──────────────────────────────────────────────┤
│          Observability Layer                  │
│  logging, tracing, monitoring, alerting       │
│  [LangSmith, Langfuse, Helicone, Datadog]     │
└──────────────────────────────────────────────┘
```

Each layer has a single responsibility. The failure modes, scaling concerns, and optimization levers at each layer are distinct.

:::tip
Build the layers bottom-up: data → LLM → retrieval → orchestration → gateway → observability. Trying to add observability after launch is painful. Adding it from day one is ~4 hours of work.
:::

---

## Token Optimization

LLM cost = (input tokens + output tokens) × price per token. Reducing token count at scale translates directly to cost reduction.

### Prompt Compression

Every token in your prompt costs money and adds latency. Audit your prompts for:

| Anti-pattern | Example | Fix |
|---|---|---|
| Verbose instructions | "Please carefully read the following and make sure you..." | "Analyze the following:" |
| Repeated context | Full system prompt in every message | Use system prompt caching |
| Redundant examples | 10 few-shot examples for a simple task | 2–3 is usually sufficient |
| Boilerplate JSON wrappers | Extensive schema descriptions | Use structured output mode |
| Full conversation history | Every prior turn in context | Summarize old turns |

### Prefix Caching

Many LLM providers support **prompt caching** — if the same prefix appears in multiple requests, the provider caches the KV state and charges less for subsequent hits.

```python
# Structure prompts to maximize cache hits:
# GOOD: stable content first, variable content last

messages = [
    {
        "role": "system",
        "content": LONG_STATIC_SYSTEM_PROMPT  # ← cached after first call
    },
    {
        "role": "user",
        "content": f"Here are the documents: {retrieved_docs}"  # ← variable, not cached
    },
    {
        "role": "user",
        "content": user_question  # ← variable, not cached
    }
]

# Anthropic pricing (as of 2025): cached input = 10% of normal input price
# For a 5000-token system prompt, 90% cost reduction on the cached portion
```

### Batch Requests

For non-real-time tasks (evaluations, embeddings, offline processing), batch requests reduce cost significantly:

```python
# Instead of:
for doc in documents:
    embedding = embed(doc)  # 1000 API calls

# Do:
embeddings = embed_batch(documents, batch_size=100)  # 10 API calls, same result
```

OpenAI's Batch API offers 50% cost reduction for async workloads with up to 24h turnaround.

---

## Caching Strategies

The cache layer sits between your orchestration code and the LLM API. Three distinct strategies:

### 1. Exact-Match Response Cache

Cache by exact prompt string. Fast, simple, effective for repeated identical queries.

```python
import hashlib
import redis

cache = redis.Redis()

def cached_completion(prompt: str, model: str, ttl: int = 3600) -> str:
    cache_key = f"llm:{hashlib.sha256(f'{model}:{prompt}'.encode()).hexdigest()}"

    cached = cache.get(cache_key)
    if cached:
        return cached.decode()

    response = call_llm(prompt, model)
    cache.setex(cache_key, ttl, response)
    return response
```

**When to use**: FAQ bots, document summarization where the same document gets queried repeatedly, any high-repeat-query workload.

**Limitation**: A single character difference = cache miss. Useless for conversational, variable-input systems.

### 2. Semantic Cache

Cache by embedding similarity. Semantically similar queries return the cached response.

```python
import numpy as np

def semantic_cache_lookup(query: str, threshold: float = 0.95) -> str | None:
    query_embedding = embed(query)

    # Search vector DB for similar past queries
    results = vector_db.search(
        collection="query_cache",
        vector=query_embedding,
        top_k=1
    )

    if results and results[0].score >= threshold:
        return results[0].payload["response"]

    return None

def semantic_cache_store(query: str, response: str):
    embedding = embed(query)
    vector_db.upsert(
        collection="query_cache",
        vectors=[{"vector": embedding, "payload": {"query": query, "response": response}}]
    )
```

**When to use**: Customer support bots where "how do I reset my password?" and "forgot my password" should return the same answer.

**Threshold tuning**: 0.95 is conservative (high precision, lower recall). 0.85 returns more cache hits but risks returning wrong answers for similar-but-different queries.

### 3. Provider-Side KV Cache

LLM providers cache the key-value states of processed tokens. If your request starts with the same prefix as a previous request, they skip recomputing attention for those tokens.

```
Request 1: [SYSTEM_PROMPT (500 tokens)] + [DOC1 (200 tokens)] + [Q1 (10 tokens)]
                    ↑
           Provider caches KV state for SYSTEM_PROMPT

Request 2: [SYSTEM_PROMPT (500 tokens)] + [DOC2 (200 tokens)] + [Q2 (10 tokens)]
                    ↑
           Cache HIT — 500 tokens computed for free
```

**To maximize KV cache hits**: always put stable content (system prompt, static context) at the beginning of your messages. Never randomize or shuffle prompt components.

---

## Model Selection Strategy

The most expensive mistake in AI systems: routing every request to the most capable model.

```
Decision framework:
─────────────────────────────────────────────────

Is this task simple? (classification, extraction, summarization)
  YES → Use small fast model (GPT-4o-mini, Gemini 1.5 Flash, Haiku)
        Cost: ~$0.0001–0.0003 per call

Does the task require reasoning, planning, or nuanced judgment?
  YES → Use large capable model (GPT-4o, Claude Sonnet, Gemini Pro)
        Cost: ~$0.003–0.015 per call

Is this a low-latency, high-volume endpoint?
  YES → Benchmark small model first; only upgrade if quality is insufficient

Is quality critical and cost secondary?
  YES → Use large model; add evaluation to detect if quality regresses
```

### Automated Router Pattern

```python
COMPLEXITY_PROMPT = """
Classify this request complexity:
- SIMPLE: factual lookup, format conversion, short extraction
- COMPLEX: multi-step reasoning, synthesis across sources, code generation

Return only: SIMPLE or COMPLEX
"""

def route_to_model(user_query: str) -> str:
    complexity = call_llm(
        prompt=COMPLEXITY_PROMPT + f"\n\nRequest: {user_query}",
        model="gpt-4o-mini",  # use cheap model for routing
        max_tokens=10
    )
    return "gpt-4o" if complexity.strip() == "COMPLEX" else "gpt-4o-mini"
```

At scale, routing 80% of requests to mini-models and 20% to large models can reduce model costs by 60–70%.

---

## Fallback Chains

LLM APIs fail. Providers have outages. Rate limits kick in. Your system must handle this.

```
Primary: OpenAI GPT-4o
    ↓ (timeout 10s or error)
Fallback 1: Anthropic Claude Sonnet
    ↓ (timeout 10s or error)
Fallback 2: Google Gemini Pro
    ↓ (all fail)
Graceful degradation: cached response / error message
```

```python
import time
from typing import Callable

def with_fallback(providers: list[Callable], prompt: str, timeout: float = 10.0) -> str:
    last_error = None

    for provider in providers:
        try:
            result = call_with_timeout(provider, prompt, timeout)
            return result
        except Exception as e:
            last_error = e
            print(f"Provider {provider.__name__} failed: {e}, trying next...")

    # All providers failed — return degraded response
    raise RuntimeError(f"All providers failed. Last error: {last_error}")
```

:::note
Fallbacks introduce inconsistency: different models respond differently to the same prompt. Design your prompts to work reasonably across all fallback models. Test your fallback chain before you need it.
:::

---

## Rate Limiting

### Why Both Sides Need Rate Limiting

You are rate-limited by providers (API rate limits). Your users also need to be rate-limited by you (token budgets, abuse prevention).

```
Provider limits:     Protect your API keys from bursts
User limits:         Protect your budget from single users
Endpoint limits:     Protect expensive features from overuse
```

### Implementation Patterns

```python
# Token bucket rate limiter using Redis
class TokenBucketRateLimiter:
    def __init__(self, redis_client, max_tokens: int, refill_rate: float):
        self.redis = redis_client
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate  # tokens per second

    def check_and_consume(self, user_id: str, tokens_needed: int) -> bool:
        key = f"rate_limit:{user_id}"
        # Lua script for atomic check-and-consume
        result = self.redis.eval(RATE_LIMIT_LUA_SCRIPT, 1, key,
                                  self.max_tokens, self.refill_rate,
                                  tokens_needed, time.time())
        return bool(result)
```

**Recommended limits by tier:**

| Tier | Requests/min | Tokens/day | Monthly budget |
|------|-------------|-----------|----------------|
| Free | 10 | 50K | $5 |
| Pro | 100 | 500K | $50 |
| Enterprise | Custom | Unlimited | Negotiated |

---

## Observability Stack

The observability stack answers four questions: What happened? Why was it slow? Is quality degrading? How much did it cost?

### The Four Pillars

**1. Logging — What happened?**

```python
import structlog

log = structlog.get_logger()

def call_llm_with_logging(prompt: str, model: str, user_id: str) -> str:
    start_time = time.time()

    response = call_llm(prompt, model)

    latency = time.time() - start_time
    input_tokens = count_tokens(prompt)
    output_tokens = count_tokens(response)

    log.info("llm_call",
        user_id=user_id,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=round(latency * 1000),
        cost_usd=calculate_cost(model, input_tokens, output_tokens)
    )

    return response
```

**2. Tracing — Why was it slow?**

LangSmith, Langfuse, and Helicone provide distributed tracing for LLM call chains. You can see exactly where time was spent in a multi-step RAG or agent pipeline.

```python
from langsmith import traceable

@traceable(name="rag_pipeline")
def rag_pipeline(question: str) -> str:
    with trace("retrieval"):
        docs = retrieve_documents(question)

    with trace("reranking"):
        docs = rerank(docs, question)

    with trace("generation"):
        response = generate_answer(question, docs)

    return response
# LangSmith shows: retrieval=120ms, reranking=45ms, generation=1800ms
# → clear that generation is the bottleneck
```

**3. Monitoring — Is it healthy right now?**

Key metrics to track continuously:

| Metric | Threshold to Alert | Why |
|--------|-------------------|-----|
| Latency p50 | > 2s | User experience |
| Latency p95 | > 5s | Long tail pain |
| Error rate | > 1% | Reliability |
| Cost per request | > $0.05 | Budget runaway |
| Cache hit rate | < 20% | Optimization opportunity |
| Quality score | < 0.8 | Silent quality regression |

**4. Alerting — Did something go wrong?**

```python
# Pseudo-code for a simple alerting rule
if metrics.cost_per_hour > HOURLY_BUDGET_ALERT_THRESHOLD:
    pagerduty.alert(
        severity="high",
        message=f"LLM cost spike: ${metrics.cost_per_hour:.2f}/hour"
    )
```

Set alerts at 50%, 80%, and 100% of your monthly budget. The 80% alert gives you time to investigate before hitting the cap.

---

## Deployment Patterns

### Serverless (AWS Lambda / Cloud Run)

```
Pros:  Zero infrastructure management, scales to zero (no idle cost)
Cons:  Cold start latency (1–3s for Python), max timeout limits (15min Lambda)
       Poor for GPU workloads, poor for streaming responses

Best for: Low-traffic or bursty workloads, simple query/response endpoints
```

### Container (ECS / GKE)

```
Pros:  Full control, persistent connections, streaming support,
       GPU instances available, predictable latency
Cons:  Always-on cost, infrastructure management burden

Best for: Sustained production traffic, streaming responses, GPU inference
```

### Edge (Cloudflare Workers / Vercel Edge)

```
Pros:  Ultra-low latency for static/cached responses
Cons:  Limited runtime (no heavy Python deps), tiny memory limits

Best for: Caching layer, auth/routing gateway — NOT for LLM inference
```

### Self-Hosted Models

When to self-host instead of using APIs:

| Reason | When It Applies |
|--------|----------------|
| Data privacy / compliance | Healthcare, legal, financial data that can't leave your network |
| Latency requirements | < 200ms inference with custom hardware |
| Cost at scale | > 10M tokens/day — frontier API may exceed self-hosting cost |
| Fine-tuned model | Custom model that can't be served via provider API |

---

## Cost Engineering

### The Math

```
Cost = (input_tokens × input_price + output_tokens × output_price) × volume

Example — GPT-4o (May 2025):
  Input:  $2.50 / 1M tokens
  Output: $10.00 / 1M tokens

Typical RAG request:
  Input:  1000 tokens (system prompt + retrieved context + question)
  Output: 300 tokens (answer)

Cost per request = (1000 × $0.0000025) + (300 × $0.00001)
                 = $0.0025 + $0.003
                 = $0.0055 per call

At 10,000 requests/day: $55/day = $1,650/month
```

### Cost Reduction Levers

1. **Model downsizing**: GPT-4o → GPT-4o-mini = 15x cheaper
2. **Prompt compression**: Reduce input tokens by 30% = 30% cost reduction
3. **Caching**: 40% cache hit rate = 40% fewer API calls
4. **Batching**: Use Batch API for async = 50% discount
5. **Output length control**: Add `max_tokens` limit, instruct model to be concise

### Production Checklist

Before shipping any AI feature to production:

- [ ] Prompt versioning and experiment tracking in place
- [ ] Token budget per user/session enforced (no unbounded loops)
- [ ] Timeout set on every LLM call (never `await` without a deadline)
- [ ] Fallback behavior defined (what happens when model API is down?)
- [ ] PII detection before logging (never log raw user messages without scrubbing)
- [ ] Cost alerts at 50%, 80%, 100% of monthly budget
- [ ] Model version pinned (providers update models — pin to a specific version)
- [ ] Evaluation CI configured (automated quality check on every deployment)
- [ ] Rate limiting per user (prevent $300 single-session abuse)
- [ ] Graceful degradation tested (manually kill the LLM API, verify behavior)

---

## Mental Model

**Your AI system is a supply chain, not a function call.**

A function either succeeds or throws. A supply chain has suppliers who occasionally fail, variable costs, quality fluctuations, and lead times. Managing it requires the same tooling as any distributed system: health monitoring, supplier diversity (fallbacks), cost tracking, and quality audits.

The LLM API is just one supplier in your supply chain. Design the rest of the system with the assumption that any component can be slow, wrong, or unavailable at any time.

---

## Common Mistakes

| Mistake | Why It Happens | Fix |
|---------|---------------|-----|
| No timeout on LLM calls | "It's usually fast" | Always set `timeout=30` — LLM APIs stall silently |
| Logging raw user messages | "We'll need this for debugging" | Scrub PII before logging; use anonymized session IDs |
| No cost alerts | "We'll watch it manually" | Set billing alerts on day one; cost spikes happen overnight |
| Single model dependency | "It's the best model" | Add at least one fallback; outages happen on Friday evenings |
| Unpinned model versions | "Latest is best" | Pin model IDs; providers silently update behavior |
| Context window creep | "More context = better answers" | Enforce max context budget per request |
| Testing with happy path only | "The demo worked" | Test with: empty retrieval, malformed output, timeout, rate limit hit |
| Skipping evaluation CI | "Manual testing is sufficient" | Quality degrades silently; automate the check |

---

## Quiz

> **Q1: You have a RAG chatbot. 40% of user questions are variations of the same 200 FAQ questions. LLM calls cost $0.005 each. You get 50,000 requests/day. What caching strategy would have the highest ROI, and what would it save?**
>
> <details><summary>Show Answer</summary>
>
> **Semantic cache** would have the highest ROI.
>
> Exact-match cache would catch only perfectly identical queries. Users rephrase: "how do I reset my password" vs "forgot password help" vs "I can't log in, how to reset."
>
> Semantic cache with a 0.92+ similarity threshold would catch these variants.
>
> Math:
> - 40% of 50K requests = 20,000 potential cache hits/day
> - At 80% cache hit rate on those = 16,000 cache hits/day
> - Savings: 16,000 × $0.005 = $80/day = $2,400/month
> - Implementation cost: Redis instance ($30/month) + 2–3 days of engineering
>
> Break-even: ~2 weeks. Strong ROI.
> </details>

> **Q2: Your AI system has a p95 latency of 8 seconds. Users are complaining. The breakdown is: retrieval 150ms, reranking 200ms, LLM generation 7.6s. Where do you optimize first and how?**
>
> <details><summary>Show Answer</summary>
>
> **LLM generation is the bottleneck (95% of latency).**
>
> Retrieval and reranking optimizations are irrelevant until generation is fixed.
>
> Options in order of impact:
>
> 1. **Enable streaming**: Users start reading after 300–500ms (time to first token) instead of waiting 7.6s for the complete response. Perceived latency drops dramatically.
>
> 2. **Reduce context size**: 7.6s generation often means a large prompt. Audit for bloated system prompts or excessive retrieved context. Cutting from 3000 to 1500 tokens can halve generation time.
>
> 3. **Use a faster model**: If quality allows, GPT-4o-mini or Claude Haiku generate at 3–5x the speed of large models for the same task.
>
> 4. **Output length control**: Add `max_tokens=500` and tell the model to be concise. Many systems default to unrestricted output length.
>
> Address in this order: streaming → context reduction → model downgrade → output limits.
> </details>

> **Q3: A model provider (your primary LLM) announces they're deprecating your pinned model version in 30 days. What steps do you take and in what order?**
>
> <details><summary>Show Answer</summary>
>
> 1. **Read the migration guide**: Understand what changed. Sometimes it's a version bump with near-identical behavior; sometimes there are systematic behavior differences.
>
> 2. **Run your evaluation suite on the new model version**: Before changing anything in production, test the new model against your existing benchmark. Compare outputs for regression.
>
> 3. **A/B test in production with a small percentage of traffic**: Route 5–10% of real requests to the new model, compare quality metrics (LLM-as-judge, thumbs up/down, task completion).
>
> 4. **Update the model string in code once quality is confirmed**: Pin to the new specific model ID, not "latest" or "default" — you need predictable behavior.
>
> 5. **Update prompts if needed**: New model versions sometimes respond differently to the same prompts. Adjust if evaluation reveals systematic differences.
>
> 6. **Deploy with a fallback to the old model**: If the deprecated version is still accessible during the transition, use it as a fallback while you roll out the new version.
>
> **Never**: deploy an untested model update directly to 100% of production traffic.
> </details>

---

## Summary Table

| Concept | What It Is | When to Use |
|---------|-----------|------------|
| Exact-match cache | Cache by exact prompt hash | High-repeat identical queries |
| Semantic cache | Cache by embedding similarity | FAQ bots, paraphrase-heavy workloads |
| KV cache | Provider-side prefix caching | Long shared system prompts |
| Model router | Classify complexity → select model | Cost optimization at scale |
| Fallback chain | Try next provider on failure | All production systems |
| Rate limiter | Per-user token/request quota | User-facing products |
| LangSmith / Langfuse | Distributed tracing for LLM chains | Debugging slow or wrong responses |
| Prompt versioning | Track prompt changes like code | Any system that A/B tests prompts |
| Batch API | Async 50%-discounted LLM calls | Offline processing, evaluations |
| Serverless deployment | Lambda/Cloud Run for AI endpoints | Variable-load, stateless endpoints |

---

## Next Steps

→ [Hands-On: Production AI Systems](./hands-on)
