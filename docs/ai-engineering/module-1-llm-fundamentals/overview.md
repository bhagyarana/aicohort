---
sidebar_position: 2
title: "Overview"
description: Autoregressive generation, tokenization, context windows, sampling strategies — how LLMs actually work.
---

# LLM Fundamentals — Deep Dive

Every weird LLM behavior — hallucinations, refusals, context confusion, unexpected truncation, inconsistent outputs — has a mechanistic explanation. This module gives you the mental model to diagnose them instead of guessing.

---

## 1. Autoregressive Language Modeling

### What LLMs Actually Do

An LLM is a **next-token predictor**. Nothing more, nothing less. It takes a sequence of tokens as input and outputs a probability distribution over every possible next token. It then samples one token from that distribution, appends it to the sequence, and repeats.

```
Input:   "The Eiffel Tower is located in"
              ↓
Model:   P(next token | input) = {" Paris": 0.71, " France": 0.14, " the": 0.06, ...}
              ↓ sample
Output:  " Paris"

New input: "The Eiffel Tower is located in Paris"
              ↓
Model:   P(next token | new input) = {".": 0.52, ",": 0.21, " and": 0.08, ...}
              ↓ sample
Output:  "."
```

This continues until the model produces a special `<|end|>` token or hits the max token limit.

### Why This Matters

The model is not retrieving a fact. It is **predicting the statistically likely continuation** of your input based on patterns in its training data. This is why:

- It can hallucinate plausible-sounding but false information (it optimizes for likelihood, not truth)
- It can answer differently on repeated identical prompts (sampling is stochastic by default)
- It performs better when you frame your prompt the way the training data was framed

:::note
"The model understands language" is a useful shorthand but technically wrong. It has learned statistical patterns that approximate understanding well enough to be useful — but it has no world model, no beliefs, and no intent.
:::

---

## 2. Tokens & Tokenization

### What Is a Token?

A token is a chunk of text — usually 3–4 characters for English. Tokens are the atomic unit of input and output for all LLMs.

```
"Hello, world!"   →  ["Hello", ",", " world", "!"]           = 4 tokens
"ChatGPT"         →  ["Chat", "G", "PT"]                     = 3 tokens
"unbelievable"    →  ["un", "believ", "able"]                 = 3 tokens
"日本語"           →  ["日", "本", "語"]                        = 3 tokens (more per char)
```

### How BPE (Byte-Pair Encoding) Works

Most LLMs use Byte-Pair Encoding to build their vocabulary:

1. Start with individual characters as the vocabulary
2. Count the most frequent pair of adjacent tokens in the training corpus
3. Merge that pair into a new token, add it to the vocabulary
4. Repeat ~50,000–100,000 times

The result: common words become single tokens, rare/new words get split into subword pieces. This is why unusual words, names, or code cost more tokens than plain English prose.

### Why Tokenization Matters

**Cost:** APIs charge per token. 1,000 tokens ≈ 750 English words.

**Context:** Your context window budget is in tokens, not words. A 128K-token window can hold roughly 96K words of English — but significantly less if your text is code or multilingual.

**Surprises to watch for:**

| Input | Tokens | Why |
|-------|--------|-----|
| `"2024-01-15"` | 5 | Date formats split at hyphens |
| `" hello"` vs `"hello"` | Different tokens | Leading space changes the token |
| `"GPT-4"` | 3–4 | Hyphenated terms split |
| Code with indentation | More than expected | Spaces in code are their own tokens |

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")

texts = ["Hello world", "hello world", " hello world", "2024-01-15", "GPT-4"]
for text in texts:
    tokens = enc.encode(text)
    print(f"{text!r:25s} → {len(tokens)} tokens: {tokens}")
```

:::tip
Always count tokens before sending a request. The `tiktoken` library (for OpenAI models) or `anthropic.count_tokens()` gives you the exact count. Budget surprises at runtime are the most avoidable cost bugs.
:::

---

## 3. Pretraining

### What the Model Learned

During pretraining, the model saw hundreds of billions of tokens of text from the internet, books, and code. It learned to predict the next token across all of this — which forced it to build internal representations of:

- Grammar and syntax
- Factual associations ("Paris is the capital of France")
- Code patterns and conventions
- Reasoning chains that appear in text
- Stylistic and tonal patterns

### What It Didn't Learn (And Cannot Know)

| Limitation | Why | Implication |
|-----------|-----|-------------|
| Events after training cutoff | Not in training data | Stale facts, wrong dates |
| Private/internal information | Not in training data | Never assume it knows your docs |
| Mathematical ground truth | Optimizes for likely text, not correctness | Arithmetic errors, especially large numbers |
| Whether its output is true | No feedback signal on factual accuracy | Always hallucinate-check critical claims |

### Why LLMs Hallucinate

When the model encounters a question where the "correct" answer was rare in training data, it still produces a probable-sounding continuation. It has no internal signal for "I don't know." The output distribution just assigns probability to whatever text pattern the training data would suggest follows your prompt.

---

## 4. Context Window

### What It Is

The context window is the maximum number of tokens the model can attend to at once — both the input (prompt) and the output it has generated so far.

| Model | Context Window |
|-------|---------------|
| GPT-4o | 128K tokens |
| Claude 3.5 Sonnet | 200K tokens |
| Gemini 1.5 Pro | 1M tokens |
| Llama 3.1 70B | 128K tokens |

### What Happens at the Boundary

When your conversation exceeds the context window, the oldest tokens get **dropped** — silently, with no error in most APIs. The model simply forgets them. This causes:

- Instructions given at the start of a long conversation being ignored
- The model losing track of document content injected early in the prompt
- Inconsistent behavior when the same prompt produces different context lengths

### Working Within the Limit

**Token budget per request:**

```
max_tokens = context_window - prompt_tokens - safety_buffer
```

**Strategies when you exceed the limit:**

| Strategy | How | Trade-off |
|----------|-----|-----------|
| Sliding window | Drop oldest messages | Loses early context |
| Summarization | Replace old messages with a summary | Lossy but compact |
| RAG | Don't put docs in context; retrieve on demand | Requires retrieval infra |
| Chunking | Split large documents; process in pieces | Multiple calls, more complex |

:::note
"Bigger context = better" is not always true. Attention scales quadratically with sequence length (O(n²)), so 128K-token contexts are significantly slower and more expensive than 8K-token contexts. The model's ability to use context buried deep in a very long prompt is also empirically weaker than context near the end — this is called the **"lost in the middle"** problem.
:::

---

## 5. The Full Generation Pipeline

From your prompt to the output token — step by step:

```
Your prompt: "The cat sat on the"
       │
       ▼
┌─────────────────────┐
│   Tokenizer         │  "The", " cat", " sat", " on", " the"
└─────────────────────┘       [464, 3797, 6275, 319, 262]
       │
       ▼
┌─────────────────────┐
│   Embedding layer   │  Each token ID → 4096-dim vector
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│   Transformer layers│  Self-attention + feed-forward × N layers
│   (e.g. 32 layers)  │  Each token attends to all prior tokens
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│   LM head           │  Final hidden state → logit per vocab token
└─────────────────────┘  shape: [vocab_size] e.g. [50,257]
       │
       ▼
┌─────────────────────┐
│   Sampling          │  Logits → softmax → sample
└─────────────────────┘  e.g. " mat" with p=0.42
       │
       ▼
Output token: " mat"  →  appended to sequence  →  repeat
```

---

## 6. Sampling Strategies

### Temperature

Divides all logits by a constant before softmax. Controls the "sharpness" of the probability distribution:

```python
import numpy as np

def sample_with_temp(logits, temperature=1.0, n=5):
    probs = np.exp(logits / temperature) / np.exp(logits / temperature).sum()
    return np.random.choice(len(logits), size=n, p=probs)

logits = np.array([3.0, 1.5, 0.8, 0.3, -0.5])  # 5 candidate tokens

# T=0.1: nearly deterministic, always picks token 0
# T=1.0: standard, follows the learned distribution
# T=2.0: flatter, more variety/randomness
```

| Temperature | Effect | Use when |
|------------|--------|---------|
| 0.0–0.3 | Near-deterministic, repeatable | Structured data extraction, code |
| 0.7–1.0 | Balanced | General purpose chat, summarization |
| 1.2–2.0 | Creative, varied | Brainstorming, creative writing |

### Top-k Sampling

Zero out all tokens except the top-k most probable, then renormalize:

```python
def top_k_sample(logits, k=50):
    # Keep only top k logits, set rest to -inf
    kth_value = np.partition(logits, -k)[-k]
    masked = np.where(logits >= kth_value, logits, -np.inf)
    probs = np.exp(masked) / np.exp(masked[np.isfinite(masked)]).sum()
    return np.random.choice(len(logits), p=probs)
```

**Problem with top-k:** A fixed k treats a peaked distribution and a flat one the same way.

### Top-p (Nucleus) Sampling

Select the smallest set of tokens whose cumulative probability exceeds p:

```python
def top_p_sample(logits, p=0.9):
    probs = np.exp(logits) / np.exp(logits).sum()
    sorted_idx = np.argsort(probs)[::-1]
    cumulative = np.cumsum(probs[sorted_idx])
    # Keep tokens until cumulative probability exceeds p
    cutoff = np.searchsorted(cumulative, p) + 1
    kept = sorted_idx[:cutoff]
    renorm = probs[kept] / probs[kept].sum()
    return np.random.choice(kept, p=renorm)
```

Top-p adapts to the distribution: when the model is confident, only 2–3 tokens may be in the nucleus; when uncertain, 50+ tokens might be included.

### Deterministic vs Stochastic

| Setting | Behavior | Note |
|---------|----------|------|
| `temperature=0` | Argmax (always top token) | Not truly deterministic — floating point and hardware differences can cause variation |
| `temperature>0` | Stochastic | Set a `seed` for reproducibility when supported |
| `top_p=1, top_k=0` | No truncation, full distribution | The model's raw learned distribution |

:::tip
For production systems extracting structured data (JSON, classification), use `temperature=0` and a structured output mode (JSON mode or function calling). For conversational use cases, `temperature=0.7, top_p=0.9` is a sensible default.
:::

---

## Mental Model

An LLM is a **very sophisticated autocomplete**. Every behavior — from answering questions to writing code to hallucinating facts — is the model continuing your text in the most statistically likely way given its training. Your prompt is the beginning of a document; the model writes the rest. Understanding this reframes every problem: if the output is wrong, ask "what kind of document would continue this way?" and adjust your prompt to steer toward the continuation you want.

---

## Common Mistakes

| Mistake | Why it happens | Fix |
|---------|----------------|-----|
| "LLMs understand text" | Anthropomorphizing the model | Think "pattern matching on tokens" not "comprehension" |
| Using `temperature=0` and expecting identical outputs | Floating point non-determinism | Set `seed` if supported; accept minor variation |
| Context window surprises mid-conversation | Not tracking token count | Add a token counter to your message loop |
| Sending the whole document in the prompt | "It has a 200K context!" | Long contexts are slow, expensive, and have attention dilution; use RAG instead |
| Assuming the model knows current events | Training cutoff | Always tell the model the current date when it's relevant |

---

## Quiz

> **Q: You ask an LLM "What is 847 × 293?" with `temperature=0`. You get a confident, wrong answer. Why, and what can you do about it?**
>
> <details><summary>Show Answer</summary>
>
> The model doesn't compute arithmetic — it predicts the most likely text continuation of "847 × 293 = ". The training data contained many correct multiplication results, but the model has no arithmetic engine. For large numbers, the most probable continuation may still be wrong.
>
> **Fixes:**
> 1. Use **tool calling / function calling** to route math to a Python interpreter
> 2. Ask the model to "think step by step" (Chain of Thought) — this often helps for medium-complexity arithmetic
> 3. For production use cases requiring exact math, never rely on raw LLM output; always use code execution
>
> </details>

---

## Summary Table

| Concept | What it is | When to use |
|---------|-----------|-------------|
| Autoregressive generation | Predicting one token at a time | Understanding all LLM behavior |
| BPE tokenization | Subword vocab built from merge rules | Token counting, cost estimation |
| Context window | Max tokens model can attend to | Sizing prompts, managing conversation history |
| Temperature | Logit scaling before softmax | Controlling output randomness |
| Top-k | Keep top k tokens only | Preventing low-probability token selection |
| Top-p (nucleus) | Keep tokens until cumulative p | Adaptive truncation of the distribution |
| Hallucination | Plausible but false generation | Always verify critical factual claims |

---

## Next Steps

→ [Hands-On: LLM Fundamentals Exercises](./hands-on)
