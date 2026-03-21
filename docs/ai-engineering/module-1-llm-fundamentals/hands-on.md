---
sidebar_position: 3
title: "Hands-On"
description: Tokenization with tiktoken, temperature experiments, context window probing, and a token budget tracker.
---

# Hands-On: LLM Fundamentals

These four exercises move from token counting to building a production-ready token budget system. You need an LLM API key for Exercises 2–4 (any provider works).

---

## Exercise 1: Tokenization with tiktoken (Beginner)

**Goal:** Develop intuition for how text becomes tokens and calculate real API costs.
**Time:** ~25 min

### Setup

```bash
pip install tiktoken
```

### Step 1 — Basic tokenization

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")

def show_tokens(text: str):
    tokens = enc.encode(text)
    decoded = [enc.decode([t]) for t in tokens]
    print(f"Text:    {text!r}")
    print(f"Count:   {len(tokens)} tokens")
    print(f"Tokens:  {decoded}")
    print()

# Explore these — predict the count before running
show_tokens("Hello, world!")
show_tokens("hello, world!")          # lowercase changes tokens
show_tokens(" hello, world!")         # leading space changes tokens
show_tokens("2024-01-15")
show_tokens("GPT-4o")
show_tokens("    ")                   # 4 spaces
show_tokens("def calculate_total():") # Python code
show_tokens("日本語テスト")             # Japanese
```

### Step 2 — Cost estimator

```python
# OpenAI GPT-4o pricing (as of 2025 — check current prices)
PRICES = {
    "gpt-4o":            {"input": 2.50,  "output": 10.00},  # per 1M tokens
    "gpt-4o-mini":       {"input": 0.15,  "output": 0.60},
    "gpt-3.5-turbo":     {"input": 0.50,  "output": 1.50},
}

def estimate_cost(prompt: str, expected_output_tokens: int, model: str = "gpt-4o") -> dict:
    enc = tiktoken.encoding_for_model(model)
    prompt_tokens = len(enc.encode(prompt))
    price = PRICES[model]
    input_cost  = (prompt_tokens / 1_000_000) * price["input"]
    output_cost = (expected_output_tokens / 1_000_000) * price["output"]
    return {
        "prompt_tokens": prompt_tokens,
        "expected_output_tokens": expected_output_tokens,
        "estimated_cost_usd": round(input_cost + output_cost, 6),
        "model": model,
    }

# Compare cost across models for the same prompt
prompt = "Summarize the following article in 3 bullet points:\n\n" + "word " * 500
for model in PRICES:
    print(estimate_cost(prompt, expected_output_tokens=150, model=model))
```

### Step 3 — Tokenization surprises

```python
# Find inputs where tokenization is non-obvious
surprises = [
    ("Normal English", "The quick brown fox jumps over the lazy dog"),
    ("Repeated chars", "aaaaaaaaaa"),                    # 10 'a's
    ("Email address",  "user@example.com"),
    ("URL",            "https://api.openai.com/v1/chat"),
    ("JSON",           '{"key": "value", "num": 42}'),
    ("Code",           "for i in range(100):\n    print(i)"),
    ("Markdown",       "## Section\n- item 1\n- item 2"),
]

for name, text in surprises:
    tokens = enc.encode(text)
    ratio  = len(text) / len(tokens)
    print(f"{name:20s}: {len(tokens):3d} tokens  ({ratio:.1f} chars/token)")
```

**Expected insight:** English prose ≈ 4 chars/token, code ≈ 2–3 chars/token, JSON ≈ 2–3 chars/token.

### What to verify

- The same word tokenizes differently with and without a leading space
- Japanese text uses many more tokens per character than English
- Code and JSON cost roughly 2× more tokens than equivalent prose

---

## Exercise 2: Temperature and Output Variance (Intermediate)

**Goal:** Empirically observe how temperature changes LLM output and build intuition for when to use each setting.
**Time:** ~25 min

### Setup

```bash
pip install openai  # or anthropic / google-generativeai
```

### Step 1 — Run the same prompt at different temperatures

```python
from openai import OpenAI

client = OpenAI()  # uses OPENAI_API_KEY from environment

def generate(prompt: str, temperature: float, n: int = 3) -> list[str]:
    """Generate n completions at the given temperature."""
    responses = []
    for _ in range(n):
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=100,
        )
        responses.append(r.choices[0].message.content.strip())
    return responses

prompt = "Complete this sentence in one creative way: 'The robot opened the door and'"

print("=" * 60)
for temp in [0.0, 0.5, 1.0, 1.5, 2.0]:
    outputs = generate(prompt, temperature=temp)
    print(f"\nTemperature = {temp}")
    for i, out in enumerate(outputs, 1):
        print(f"  [{i}] {out}")
```

### Step 2 — Structured extraction: verify temperature=0 stability

```python
import json

extract_prompt = """
Extract the following fields from this text and return as JSON:
- name (string)
- age (integer)
- city (string)

Text: "Sarah Johnson is 34 years old and lives in Austin, Texas."

Return only valid JSON, no markdown.
"""

print("Structured extraction at temperature=0 (should be identical every time):")
for i in range(5):
    result = generate(extract_prompt, temperature=0.0, n=1)[0]
    try:
        parsed = json.loads(result)
        print(f"  Run {i+1}: {parsed}")
    except json.JSONDecodeError:
        print(f"  Run {i+1}: INVALID JSON: {result}")
```

### Step 3 — Measure variance quantitatively

```python
def output_variety(prompt: str, temperature: float, n: int = 10) -> float:
    """Returns the fraction of unique outputs (0 = all identical, 1 = all unique)."""
    outputs = generate(prompt, temperature=temperature, n=n)
    return len(set(outputs)) / n

simple_prompt = "What is 2 + 2?"

print("\nVariety score (0=identical, 1=all unique):")
for temp in [0.0, 0.3, 0.7, 1.0, 1.5]:
    variety = output_variety(simple_prompt, temperature=temp)
    print(f"  T={temp:.1f}: {variety:.1f}")
```

### What to verify

- `temperature=0` produces identical or near-identical results across runs for a simple factual prompt
- `temperature=2.0` sometimes produces incoherent or malformed output
- JSON extraction is reliable at `temperature=0`, unreliable at `temperature=1.5+`

---

## Exercise 3: Hitting the Context Window (Intermediate)

**Goal:** Observe what happens when a conversation exceeds the model's context limit.
**Time:** ~20 min

### Step 1 — Build a conversation that grows over time

```python
import tiktoken
from openai import OpenAI

client = OpenAI()
enc    = tiktoken.encoding_for_model("gpt-4o-mini")

def count_messages_tokens(messages: list[dict]) -> int:
    """Rough token count for a list of chat messages."""
    total = 0
    for msg in messages:
        total += len(enc.encode(msg["content"])) + 4  # 4 tokens overhead per message
    return total + 2  # 2 tokens for priming

# Plant a fact at the start of the conversation
system_msg = {"role": "system", "content": "You are a helpful assistant."}
planted_fact = {"role": "user", "content": "Remember this: my lucky number is 7777."}
planted_ack  = {"role": "assistant", "content": "Got it! Your lucky number is 7777."}

messages = [system_msg, planted_fact, planted_ack]

# Pad the conversation with filler to push the planted fact out of a small context
filler = "Tell me a random interesting fact about science. " * 10
for i in range(30):
    messages.append({"role": "user",      "content": filler})
    messages.append({"role": "assistant", "content": f"Here is fact {i}: " + "x " * 50})

# Ask about the planted fact
messages.append({"role": "user", "content": "What is my lucky number?"})

print(f"Total tokens in conversation: {count_messages_tokens(messages)}")
```

### Step 2 — Test with a short-context model

```python
# gpt-4o-mini has 128K context — hard to overflow in practice
# Use max_tokens to simulate what happens when context is constrained

# Instead, let's test the "lost in the middle" effect
def test_retrieval_position(position: str) -> str:
    """
    Tests whether the model finds a needle based on where it's placed.
    """
    needle = "The secret code is ALPHA-7."
    haystack = ["This is a document about machine learning. " * 20] * 10

    if position == "start":
        docs = [needle] + haystack
    elif position == "middle":
        mid = len(haystack) // 2
        docs = haystack[:mid] + [needle] + haystack[mid:]
    else:  # end
        docs = haystack + [needle]

    context = "\n\n".join(docs)
    prompt = f"Based on the documents below, what is the secret code?\n\n{context}"

    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=50,
    )
    return r.choices[0].message.content.strip()

for pos in ["start", "middle", "end"]:
    result = test_retrieval_position(pos)
    print(f"Needle at {pos:6s}: {result}")
```

### Step 3 — Track token usage in API responses

```python
r = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain photosynthesis in two sentences."}],
    temperature=0,
)

usage = r.usage
print(f"Prompt tokens:     {usage.prompt_tokens}")
print(f"Completion tokens: {usage.completion_tokens}")
print(f"Total tokens:      {usage.total_tokens}")
print(f"Cost estimate:     ${usage.total_tokens / 1_000_000 * 0.15:.6f}")
```

### What to verify

- The "needle in the middle" is harder to retrieve than needles at start or end
- Token usage is always available in the API response — you don't need to estimate it

---

## Exercise 4: Token Budget Tracker (Intermediate)

**Goal:** Build a conversation manager that tracks token usage and warns before hitting the context limit.
**Time:** ~30 min

### The full implementation

```python
import tiktoken
from openai import OpenAI
from dataclasses import dataclass, field

@dataclass
class ConversationManager:
    model: str = "gpt-4o-mini"
    context_limit: int = 128_000
    max_response_tokens: int = 1_000
    warning_threshold: float = 0.8     # warn at 80% of context used
    messages: list[dict] = field(default_factory=list)
    _client: OpenAI = field(default_factory=OpenAI, repr=False)

    def __post_init__(self):
        self._enc = tiktoken.encoding_for_model(self.model)

    def _count_tokens(self, messages: list[dict]) -> int:
        total = sum(len(self._enc.encode(m["content"])) + 4 for m in messages)
        return total + 2

    @property
    def tokens_used(self) -> int:
        return self._count_tokens(self.messages)

    @property
    def tokens_remaining(self) -> int:
        return self.context_limit - self.tokens_used - self.max_response_tokens

    @property
    def context_usage_pct(self) -> float:
        return self.tokens_used / self.context_limit

    def _trim_if_needed(self):
        """Remove oldest non-system messages until we have headroom."""
        while self.tokens_remaining < 0 and len(self.messages) > 1:
            # Find the first non-system message
            for i, msg in enumerate(self.messages):
                if msg["role"] != "system":
                    removed = self.messages.pop(i)
                    print(f"  [Trimmed oldest message: {removed['content'][:40]}...]")
                    break

    def add_system(self, content: str):
        self.messages.insert(0, {"role": "system", "content": content})

    def chat(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})
        self._trim_if_needed()

        # Warn if context is getting full
        if self.context_usage_pct > self.warning_threshold:
            print(f"  ⚠ Context {self.context_usage_pct:.0%} full "
                  f"({self.tokens_used:,}/{self.context_limit:,} tokens)")

        response = self._client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=0.7,
            max_tokens=self.max_response_tokens,
        )
        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})

        print(f"  [Tokens: {self.tokens_used:,} used / {self.tokens_remaining:,} remaining]")
        return assistant_message

# Usage
manager = ConversationManager()
manager.add_system("You are a concise assistant. Keep responses under 3 sentences.")

questions = [
    "What is a neural network?",
    "What is backpropagation?",
    "What is gradient descent?",
    "How do transformers differ from RNNs?",
    "What is an embedding?",
]

for q in questions:
    print(f"\nUser: {q}")
    answer = manager.chat(q)
    print(f"Assistant: {answer}")
```

### What to verify

- Token count increases with each turn
- The trim function removes old messages when approaching the limit
- The warning fires at 80% capacity

### Challenge

Extend `ConversationManager` to **summarize** trimmed messages instead of discarding them. When the oldest N messages are removed, send them to the model for summarization and inject the summary as a system message.

---

## Checklist

- [ ] Completed Exercise 1: Can explain why "hello" and " hello" tokenize differently
- [ ] Completed Exercise 2: Observed temperature's effect on output variance experimentally
- [ ] Completed Exercise 3: Can explain the "lost in the middle" problem with evidence from your test
- [ ] Completed Exercise 4: Token budget tracker warns and trims messages near context limit
- [ ] (Optional) Implemented summary-based context trimming

---

**Next →** [Resources: LLM Fundamentals](./resources)
