---
sidebar_position: 3
title: "Hands-On"
description: Practical exercises — LangSmith tracing, semantic cache, model router, cost dashboard
---

# Hands-On: Production AI Systems

## Setup

```bash
pip install openai langsmith redis sentence-transformers python-dotenv
```

```python
import os
import openai
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
```

---

## Exercise 1: Instrument a RAG Pipeline with LangSmith (Beginner)

**Goal:** Add end-to-end tracing to a RAG pipeline so you can see exactly where time and tokens are spent.

**Time:** ~30 min

### Step 1: Set up LangSmith

Sign up at [smith.langchain.com](https://smith.langchain.com) — free tier is sufficient. Get your API key.

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_key_here
export LANGCHAIN_PROJECT=ai-engineering-course
```

### Step 2: Trace a simple LLM call

```python
from langsmith import traceable
import openai
import time

client = openai.OpenAI()

@traceable(name="simple_llm_call")
def call_llm(prompt: str, model: str = "gpt-4o-mini") -> str:
    """A traced LLM call."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response.choices[0].message.content

# This call will appear in LangSmith dashboard
result = call_llm("What is the capital of France?")
print(result)
# Go to smith.langchain.com and view the trace
```

### Step 3: Trace a full RAG pipeline

```python
from langsmith import traceable
import chromadb

# Simple in-memory vector store for this exercise
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("docs")

# Add some sample documents
sample_docs = [
    "The Eiffel Tower is located in Paris, France. It was built in 1889.",
    "Python is a high-level programming language created by Guido van Rossum in 1991.",
    "Machine learning is a subset of artificial intelligence focused on learning from data.",
    "The capital of Japan is Tokyo, which has a population of over 13 million.",
]

collection.add(
    documents=sample_docs,
    ids=[f"doc_{i}" for i in range(len(sample_docs))]
)


@traceable(name="retrieve_documents")
def retrieve(query: str, n_results: int = 2) -> list[str]:
    """Retrieve relevant documents for a query."""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results["documents"][0]


@traceable(name="generate_answer")
def generate(question: str, context: list[str]) -> str:
    """Generate an answer given context documents."""
    context_text = "\n\n".join(f"- {doc}" for doc in context)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Answer questions based only on the provided context. If the answer isn't in the context, say so."
            },
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion: {question}"
            }
        ],
        max_tokens=300
    )
    return response.choices[0].message.content


@traceable(name="rag_pipeline")
def rag_pipeline(question: str) -> dict:
    """Full RAG pipeline: retrieve → generate."""
    docs = retrieve(question)
    answer = generate(question, docs)
    return {"answer": answer, "sources": docs}


# Run several queries and view traces in LangSmith
queries = [
    "When was the Eiffel Tower built?",
    "What programming language did Guido van Rossum create?",
    "What is the population of Tokyo?",
]

for query in queries:
    result = rag_pipeline(query)
    print(f"\nQ: {query}")
    print(f"A: {result['answer']}")
```

### Step 4: Analyze your traces

In LangSmith, for each trace you can see:
- **Latency breakdown**: retrieval vs generation time
- **Token counts**: input and output tokens at each step
- **Cost**: per-call cost based on model pricing
- **Inputs and outputs**: exact prompts and responses

Look for: Which step takes the longest? What's the input token count for the generation step? Is context being injected correctly?

---

## Exercise 2: Semantic Cache with Redis (Intermediate)

**Goal:** Implement a semantic cache that returns cached responses for similar (not identical) queries.

**Time:** ~45 min

### Step 1: Set up Redis (local)

```bash
# Using Docker:
docker run -d --name redis-cache -p 6379:6379 redis:alpine

# Or install locally:
# brew install redis && redis-server (macOS)
# apt-get install redis-server && redis-server (Ubuntu)
```

### Step 2: Implement the semantic cache

```python
import redis
import json
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize
redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
encoder = SentenceTransformer("all-MiniLM-L6-v2")  # Fast, ~80MB

CACHE_TTL = 3600  # Cache entries expire after 1 hour
SIMILARITY_THRESHOLD = 0.92  # Minimum cosine similarity for cache hit


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def semantic_cache_lookup(query: str) -> str | None:
    """Look up a query in the semantic cache. Returns response or None."""
    query_embedding = encoder.encode(query).tolist()

    # Get all cached queries (in production, use vector DB for scale)
    all_keys = redis_client.keys("semantic_cache:*")

    best_score = 0.0
    best_key = None

    for key in all_keys:
        entry = redis_client.get(key)
        if not entry:
            continue

        data = json.loads(entry)
        cached_embedding = data["embedding"]
        score = cosine_similarity(query_embedding, cached_embedding)

        if score > best_score:
            best_score = score
            best_key = key

    if best_score >= SIMILARITY_THRESHOLD and best_key:
        data = json.loads(redis_client.get(best_key))
        print(f"  [CACHE HIT] similarity={best_score:.3f}, original_query='{data['original_query']}'")
        return data["response"]

    print(f"  [CACHE MISS] best_similarity={best_score:.3f}")
    return None


def semantic_cache_store(query: str, response: str):
    """Store a query-response pair in the semantic cache."""
    embedding = encoder.encode(query).tolist()
    key = f"semantic_cache:{hashlib.sha256(query.encode()).hexdigest()}"

    entry = {
        "original_query": query,
        "embedding": embedding,
        "response": response
    }

    redis_client.setex(key, CACHE_TTL, json.dumps(entry))
    print(f"  [CACHED] '{query[:50]}...'")


def cached_llm_call(query: str) -> tuple[str, bool]:
    """Call LLM with semantic caching. Returns (response, was_cached)."""
    # Check cache first
    cached_response = semantic_cache_lookup(query)
    if cached_response:
        return cached_response, True

    # Call LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": query}],
        max_tokens=200
    )
    result = response.choices[0].message.content

    # Store in cache
    semantic_cache_store(query, result)

    return result, False
```

### Step 3: Test and measure cache hit rate

```python
# First batch: seed the cache
seed_queries = [
    "How do I reset my password?",
    "What are your business hours?",
    "How do I cancel my subscription?",
    "Where can I download the mobile app?",
    "What payment methods do you accept?",
]

print("=== Seeding cache ===")
for query in seed_queries:
    response, was_cached = cached_llm_call(query)
    print(f"Query: '{query[:40]}...' | Cached: {was_cached}")

print("\n=== Testing cache with similar queries ===")
test_queries = [
    "I forgot my password, how to reset it?",       # similar to #1
    "When are you open?",                             # similar to #2
    "I want to cancel my account",                   # similar to #3
    "How to install your app on my phone?",          # similar to #4
    "Do you take credit cards?",                     # similar to #5
    "What's the weather like today?",               # dissimilar — should miss
]

hits = 0
for query in test_queries:
    response, was_cached = cached_llm_call(query)
    if was_cached:
        hits += 1

print(f"\nCache hit rate: {hits}/{len(test_queries)} = {hits/len(test_queries)*100:.0f}%")
```

### Step 4: Measure cost savings

```python
def calculate_savings(n_requests: int, hit_rate: float,
                      avg_input_tokens: int = 100, avg_output_tokens: int = 150) -> dict:
    """Calculate cost savings from semantic caching."""
    # GPT-4o-mini pricing (approximate)
    input_price_per_1m = 0.15
    output_price_per_1m = 0.60

    cost_per_call = (avg_input_tokens / 1_000_000 * input_price_per_1m +
                     avg_output_tokens / 1_000_000 * output_price_per_1m)

    cost_without_cache = n_requests * cost_per_call
    cost_with_cache = n_requests * (1 - hit_rate) * cost_per_call

    return {
        "requests": n_requests,
        "hit_rate": f"{hit_rate*100:.0f}%",
        "cost_without_cache": f"${cost_without_cache:.4f}",
        "cost_with_cache": f"${cost_with_cache:.4f}",
        "savings": f"${cost_without_cache - cost_with_cache:.4f}",
        "savings_pct": f"{hit_rate*100:.0f}%"
    }

# Model the economics
for daily_requests in [1000, 10_000, 100_000]:
    for hit_rate in [0.3, 0.5, 0.7]:
        savings = calculate_savings(daily_requests, hit_rate)
        print(f"Requests/day: {daily_requests:,} | Hit rate: {savings['hit_rate']} | "
              f"Daily savings: {savings['savings']}")
```

---

## Exercise 3: Model Router (Intermediate)

**Goal:** Build a classifier that routes queries to the right model based on complexity.

**Time:** ~35 min

### Step 1: Define routing criteria

```python
ROUTER_SYSTEM_PROMPT = """You are a query complexity classifier.

Classify the complexity of the given query:

SIMPLE: The query can be answered with basic lookup, simple extraction, or short generation.
  - Factual questions with clear answers
  - Format conversions (JSON to CSV, etc.)
  - Single-step tasks
  - Short text classification

COMPLEX: The query requires reasoning, synthesis, or multi-step work.
  - Multi-step analysis or planning
  - Code generation or debugging
  - Comparing and synthesizing multiple sources
  - Open-ended questions requiring nuanced judgment

Respond with ONLY one word: SIMPLE or COMPLEX"""


def classify_complexity(query: str) -> str:
    """Classify query complexity using a cheap model."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Use cheap model for routing
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ],
        max_tokens=5,
        temperature=0  # Deterministic
    )
    classification = response.choices[0].message.content.strip().upper()
    return classification if classification in ("SIMPLE", "COMPLEX") else "COMPLEX"


def route_query(query: str) -> tuple[str, str]:
    """Route query to appropriate model. Returns (model, classification)."""
    complexity = classify_complexity(query)

    model_map = {
        "SIMPLE": "gpt-4o-mini",  # Fast, cheap
        "COMPLEX": "gpt-4o"       # Capable, expensive
    }

    return model_map[complexity], complexity
```

### Step 2: Routed query execution

```python
import time

def execute_with_routing(query: str, verbose: bool = True) -> dict:
    """Execute a query with automatic model routing."""
    start = time.time()

    # Route
    model, complexity = route_query(query)

    if verbose:
        print(f"  Classification: {complexity} → {model}")

    # Execute on selected model
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": query}],
        max_tokens=500
    )

    elapsed = time.time() - start
    usage = response.usage

    # Calculate cost
    prices = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o":      {"input": 2.50, "output": 10.00},
    }
    cost = (usage.prompt_tokens / 1_000_000 * prices[model]["input"] +
            usage.completion_tokens / 1_000_000 * prices[model]["output"])

    return {
        "query": query,
        "model": model,
        "complexity": complexity,
        "response": response.choices[0].message.content,
        "latency_s": round(elapsed, 2),
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "cost_usd": round(cost, 6)
    }
```

### Step 3: Benchmark routing decisions

```python
test_queries = [
    # SIMPLE expected:
    "What is the capital of Australia?",
    "Convert this temperature: 100°F to Celsius",
    "Is 'hello world' a palindrome?",
    "What year was Python created?",

    # COMPLEX expected:
    "Compare the tradeoffs between microservices and monolithic architecture for a startup with 5 engineers",
    "Debug this Python code and explain what's wrong: def fib(n): return fib(n-1) + fib(n-2)",
    "Write a function that finds all prime numbers up to N using the Sieve of Eratosthenes",
    "What are the key considerations when choosing between RAG and fine-tuning for an enterprise chatbot?",
]

print(f"{'Query':<55} {'Classification':<12} {'Model':<15} {'Cost'}")
print("-" * 100)

total_cost_with_routing = 0
total_cost_without_routing = 0

for query in test_queries:
    result = execute_with_routing(query, verbose=False)

    # What it would cost without routing (always GPT-4o)
    gpt4_cost = (result["input_tokens"] / 1_000_000 * 2.50 +
                 result["output_tokens"] / 1_000_000 * 10.00)

    total_cost_with_routing += result["cost_usd"]
    total_cost_without_routing += gpt4_cost

    print(f"{query[:54]:<55} {result['complexity']:<12} {result['model']:<15} ${result['cost_usd']:.6f}")

savings_pct = (1 - total_cost_with_routing / total_cost_without_routing) * 100
print(f"\nTotal cost with routing:    ${total_cost_with_routing:.6f}")
print(f"Total cost without routing: ${total_cost_without_routing:.6f}")
print(f"Cost savings:               {savings_pct:.1f}%")
```

---

## Mini-Project: Cost Tracking Dashboard (Advanced)

**Goal:** Build a real-time cost tracking system that monitors spend per user, per feature, and per day.

**Time:** ~60 min

### Step 1: Cost tracker class

```python
import json
import time
from collections import defaultdict
from datetime import datetime, date
from dataclasses import dataclass, asdict

MODEL_PRICING = {
    "gpt-4o":           {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":      {"input": 0.15,  "output": 0.60},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "gemini-1.5-pro":   {"input": 1.25,  "output": 5.00},
}


@dataclass
class LLMCallRecord:
    timestamp: float
    user_id: str
    feature: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int


class CostTracker:
    def __init__(self):
        self.records: list[LLMCallRecord] = []

    def record_call(self, user_id: str, feature: str, model: str,
                    input_tokens: int, output_tokens: int, latency_ms: int):
        """Record a single LLM API call."""
        pricing = MODEL_PRICING.get(model, {"input": 1.0, "output": 1.0})
        cost = (input_tokens / 1_000_000 * pricing["input"] +
                output_tokens / 1_000_000 * pricing["output"])

        record = LLMCallRecord(
            timestamp=time.time(),
            user_id=user_id,
            feature=feature,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms
        )
        self.records.append(record)
        return record

    def cost_by_user(self, since_days: int = 30) -> dict[str, float]:
        """Total cost per user over the last N days."""
        cutoff = time.time() - (since_days * 86400)
        totals = defaultdict(float)
        for r in self.records:
            if r.timestamp >= cutoff:
                totals[r.user_id] += r.cost_usd
        return dict(sorted(totals.items(), key=lambda x: x[1], reverse=True))

    def cost_by_feature(self, since_days: int = 30) -> dict[str, float]:
        """Total cost per feature over the last N days."""
        cutoff = time.time() - (since_days * 86400)
        totals = defaultdict(float)
        for r in self.records:
            if r.timestamp >= cutoff:
                totals[r.feature] += r.cost_usd
        return dict(sorted(totals.items(), key=lambda x: x[1], reverse=True))

    def daily_cost(self, last_days: int = 7) -> dict[str, float]:
        """Total cost per day for the last N days."""
        daily = defaultdict(float)
        for r in self.records:
            day = datetime.fromtimestamp(r.timestamp).strftime("%Y-%m-%d")
            daily[day] += r.cost_usd
        # Return last N days
        sorted_days = sorted(daily.items())[-last_days:]
        return dict(sorted_days)

    def top_users_by_cost(self, n: int = 5, since_days: int = 1) -> list[dict]:
        """Top N highest-cost users in the last N days."""
        by_user = self.cost_by_user(since_days)
        return [{"user_id": u, "cost": c} for u, c in list(by_user.items())[:n]]

    def check_budget_alerts(self, daily_budget: float = 10.0) -> list[str]:
        """Return alert messages if budget thresholds are exceeded."""
        today_costs = self.daily_cost(1)
        today_total = sum(today_costs.values())
        alerts = []

        if today_total >= daily_budget:
            alerts.append(f"CRITICAL: Daily budget exceeded! ${today_total:.2f} / ${daily_budget:.2f}")
        elif today_total >= daily_budget * 0.8:
            alerts.append(f"WARNING: 80% of daily budget used. ${today_total:.2f} / ${daily_budget:.2f}")
        elif today_total >= daily_budget * 0.5:
            alerts.append(f"INFO: 50% of daily budget used. ${today_total:.2f} / ${daily_budget:.2f}")

        # Check per-user limits
        for user_data in self.top_users_by_cost(n=10, since_days=1):
            if user_data["cost"] > daily_budget * 0.2:
                alerts.append(f"WARNING: User {user_data['user_id']} spent ${user_data['cost']:.4f} today")

        return alerts

    def print_dashboard(self):
        """Print a cost dashboard summary."""
        print("\n" + "=" * 60)
        print("COST DASHBOARD")
        print("=" * 60)

        print("\nDaily Cost (last 7 days):")
        for day, cost in self.daily_cost(7).items():
            bar = "█" * int(cost * 1000)
            print(f"  {day}: ${cost:.4f} {bar}")

        print("\nCost by Feature (last 30 days):")
        for feature, cost in list(self.cost_by_feature(30).items())[:5]:
            print(f"  {feature:<30}: ${cost:.4f}")

        print("\nTop Users by Cost (today):")
        for user_data in self.top_users_by_cost(n=5, since_days=1):
            print(f"  {user_data['user_id']:<20}: ${user_data['cost']:.4f}")

        alerts = self.check_budget_alerts()
        if alerts:
            print("\nAlerts:")
            for alert in alerts:
                print(f"  ⚠ {alert}")
        else:
            print("\n  ✓ All budgets within limits")
```

### Step 2: Wrap LLM calls with tracking

```python
def tracked_llm_call(user_id: str, feature: str, messages: list[dict],
                      model: str = "gpt-4o-mini", tracker: CostTracker = None) -> str:
    """LLM call with automatic cost tracking."""
    start = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=300
    )

    latency_ms = int((time.time() - start) * 1000)

    if tracker:
        tracker.record_call(
            user_id=user_id,
            feature=feature,
            model=model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            latency_ms=latency_ms
        )

    return response.choices[0].message.content
```

### Step 3: Simulate traffic and view dashboard

```python
import random

tracker = CostTracker()

# Simulate a week of usage
users = [f"user_{i}" for i in range(1, 21)]
features = ["chat", "summarize", "rag_search", "code_assist", "translation"]

print("Simulating API calls...")
for day_offset in range(7):
    # Vary traffic by day (weekend dip)
    n_calls = random.randint(50, 200) if day_offset < 5 else random.randint(20, 60)

    for _ in range(n_calls):
        user = random.choice(users)
        feature = random.choices(features, weights=[40, 20, 25, 10, 5])[0]
        model = random.choices(
            ["gpt-4o-mini", "gpt-4o"],
            weights=[75, 25]
        )[0]

        # Inject a "heavy user" — user_1 makes 3x more calls
        if random.random() < 0.15:
            user = "user_1"

        # Simulate token counts
        input_tokens = random.randint(200, 1500)
        output_tokens = random.randint(100, 600)

        # Fake timestamp for different days
        fake_timestamp = time.time() - (6 - day_offset) * 86400 - random.randint(0, 86400)

        pricing = MODEL_PRICING.get(model, {"input": 1.0, "output": 1.0})
        cost = (input_tokens / 1_000_000 * pricing["input"] +
                output_tokens / 1_000_000 * pricing["output"])

        record = LLMCallRecord(
            timestamp=fake_timestamp,
            user_id=user,
            feature=feature,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=random.randint(200, 3000)
        )
        tracker.records.append(record)

tracker.print_dashboard()
```

---

## Checklist

- [ ] Exercise 1: LangSmith traces visible in dashboard for all 3 RAG queries
- [ ] Exercise 1: Can explain the latency breakdown (retrieval vs generation)
- [ ] Exercise 2: Semantic cache achieves >60% hit rate on the test query set
- [ ] Exercise 2: Understand why the similarity threshold matters
- [ ] Exercise 3: Router correctly classifies at least 6/8 test queries
- [ ] Exercise 3: Cost comparison shows >40% savings from routing
- [ ] Mini-project: Dashboard shows daily cost, cost by feature, and top users
- [ ] Mini-project: Budget alerts trigger correctly

---

**Next: →** [Resources: Production AI Systems](./resources)
