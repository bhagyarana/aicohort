---
sidebar_position: 2
title: "Overview"
description: Embeddings, self-attention, feed-forward layers, KV cache, positional embeddings, and scaling laws — how transformers actually work.
---

# Transformer & LLM Internals — Deep Dive

Every production LLM problem — slow decode, unexpected context forgetting, inconsistent long-document answers, ballooning inference cost — has a root cause inside the architecture. This module gives you the mechanical understanding to trace those problems to their source instead of guessing at prompt tweaks.

---

## 1. Embeddings

### From Token IDs to Vectors

The tokenizer produces integer IDs. Before any computation can happen, each ID must become a continuous vector the model can do math on. The embedding layer is simply a large lookup table:

```
Token ID: 3797  ("cat")
     ↓
Embedding table[3797]  →  [0.21, -0.84, 0.13, 0.57, ..., -0.02]
                                      768 floats (GPT-2)
```

Each row of the embedding table is a **learned, dense float vector**. The table is trained alongside the rest of the model — those numbers are not hand-designed; they emerge from gradient descent on next-token prediction.

### Embedding Dimensions Across Models

| Model | Embedding dimension (d_model) |
|-------|-------------------------------|
| GPT-2 | 768 |
| GPT-3 | 12,288 |
| Llama 3 8B | 4,096 |
| Llama 3 70B | 8,192 |

Larger d_model means more capacity to encode fine-grained distinctions — but also more parameters and memory.

### Semantic Geometry

The key property of trained embeddings: **similar concepts cluster together**. Distance in the embedding space reflects semantic similarity.

```
High-dimensional embedding space (shown as 2D projection):

         queen •
                    king •
   woman •
                  man •

         france •   paris •

   germany •  berlin •
```

Cosine similarity measures the angle between two vectors — it ranges from -1 (opposite) to 1 (identical direction):

```python
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Example: if you have embedding vectors for two words
# sim = cosine_similarity(embedding["king"], embedding["queen"])
# Expect: > 0.7 for closely related words
```

### The Classic Analogy

The famous arithmetic: **king − man + woman ≈ queen**

This works because the model learned a "gender direction" in embedding space. Subtracting the man vector and adding the woman vector navigates that direction, landing near queen. This is not magic — it is a consequence of training on text where gender relationships appear consistently.

:::note
The embedding table is trained end-to-end alongside the transformer layers. The model learns which geometric structure is most useful for predicting the next token across all of the training data. You cannot inspect a single embedding dimension and know what "feature" it represents — the representations are distributed across all dimensions.
:::

---

## 2. Self-Attention: The Core Mechanism

### The Problem Attention Solves

For a model to answer "What does *it* refer to in 'The trophy didn't fit in the suitcase because *it* was too big'?", it needs to match "it" with context elsewhere in the sentence. In older recurrent networks (RNNs/LSTMs), information had to travel step-by-step through the sequence — a bottleneck that degraded over long distances. Attention solves this by letting every position directly query every other position in a single operation.

### The Soft Database Lookup Model

Think of attention as a differentiable key-value lookup:

- **Query (Q):** "What am I looking for?" — derived from the current token
- **Key (K):** "What do I contain?" — derived from every token in the sequence
- **Value (V):** "What do I return if you match me?" — derived from every token in the sequence

For each query, the model computes how well it matches every key (the attention score), then returns a weighted average of the values:

```
Score = Q · Kᵀ / √d_k          (scaled dot product)
Weights = softmax(Score)         (turn scores into a probability distribution)
Output = Weights · V             (weighted average of values)
```

### Step-by-Step with Matrices

Given a sequence of n tokens, each with embedding dimension d_model:

```
Input X: shape (n, d_model)

Learned weight matrices (one set per attention head):
  W_Q: (d_model, d_k)
  W_K: (d_model, d_k)
  W_V: (d_model, d_v)

Compute:
  Q = X · W_Q      shape: (n, d_k)
  K = X · W_K      shape: (n, d_k)
  V = X · W_V      shape: (n, d_v)

Scores = Q · Kᵀ   shape: (n, n)   — every query against every key
Scores /= √d_k    — scaling to prevent vanishing gradients

Weights = softmax(Scores, axis=-1)   shape: (n, n)
Output  = Weights · V                 shape: (n, d_v)
```

ASCII diagram — attention score matrix for a 4-token sequence:

```
            Keys:
          t1   t2   t3   t4
Queries: ┌────┬────┬────┬────┐
  t1     │0.7 │0.1 │0.1 │0.1 │  ← t1 mostly attends to itself
  t2     │0.2 │0.6 │0.1 │0.1 │
  t3     │0.1 │0.4 │0.4 │0.1 │  ← t3 attends to t2 and itself
  t4     │0.05│0.05│0.4 │0.5 │
         └────┴────┴────┴────┘
         (rows sum to 1.0 after softmax)
```

:::tip
The √d_k scaling is not cosmetic. When d_k is large (e.g., 128), the raw dot products grow large in magnitude, pushing softmax into near-zero gradient territory (all weight concentrated on one token). Dividing by √d_k keeps the scores in a range where softmax gradients remain healthy during training.
:::

### Multi-Head Attention

Running attention once with large matrices learns one type of relationship. Multi-head attention runs H independent attention operations in parallel, each with different weight matrices, then concatenates and projects the results:

```
for head h in 1..H:
    Q_h = X · W_Q_h    (d_model → d_k = d_model / H)
    K_h = X · W_K_h
    V_h = X · W_V_h
    head_h = attention(Q_h, K_h, V_h)

Output = concat(head_1, ..., head_H) · W_O
```

Different heads specialize: one might track syntactic dependencies, another coreference, another positional proximity. GPT-3 has 96 heads; Llama 3 8B has 32.

### Causal (Masked) Attention

During text generation, each token can only attend to *previous* tokens — not future ones (which haven't been generated yet). This is enforced by adding −∞ to all positions above the diagonal before softmax:

```
Causal mask for 4 tokens (upper triangle = −∞):

    t1    t2    t3    t4
t1 [  0   -∞   -∞   -∞ ]
t2 [  0    0   -∞   -∞ ]
t3 [  0    0    0   -∞ ]
t4 [  0    0    0    0 ]

After softmax, the -∞ entries become 0 — those positions contribute nothing.
```

### Why Attention Replaced RNNs

| Property | RNN/LSTM | Transformer Attention |
|----------|----------|-----------------------|
| Processing order | Sequential (step by step) | Parallel (all positions at once) |
| Max path length between tokens | O(n) steps | O(1) — direct |
| Long-range dependencies | Degrades with distance | Equally easy at any distance |
| Training parallelism | Poor (depends on previous step) | Excellent (all positions simultaneously) |
| GPU utilization | Low | High |

---

## 3. Feed-Forward Layers

### What They Do

After the attention sub-layer, each token position passes through a small two-layer MLP independently. Crucially, there is **no interaction between positions** in this step — it is purely a per-token transformation.

```
Structure (per token position):

  x: (d_model,)
     ↓
  Linear:  (d_model → 4 × d_model)   — expand
     ↓
  GELU activation
     ↓
  Linear:  (4 × d_model → d_model)   — contract
     ↓
  output: (d_model,)
```

The expansion factor of 4× is the standard (sometimes 8/3 for SwiGLU variants used in Llama). This "bottleneck then expand then contract" structure provides non-linearity and transformation capacity.

### Attention vs Feed-Forward: Roles

| Component | What it does | Analogy |
|-----------|-------------|---------|
| Self-Attention | Routes information between token positions | "Who should I listen to?" |
| Feed-Forward | Transforms each token's representation | "Now what does that information mean?" |

Mechanistic interpretability research suggests attention layers retrieve factual associations while FFN layers store and transform them — but this is still an active research area.

---

## 4. The Full Transformer Block

One transformer block stacks the two sub-layers with residual connections and layer normalization:

```
Input x
  │
  ├──────────────────────────────────────┐
  ↓                                      │
LayerNorm(x)                             │
  ↓                                      │
Self-Attention                           │
  ↓                                      │
  + ◄────────────────────────────────────┘  (residual add)
  │
  ├──────────────────────────────────────┐
  ↓                                      │
LayerNorm                                │
  ↓                                      │
Feed-Forward (MLP)                       │
  ↓                                      │
  + ◄────────────────────────────────────┘  (residual add)
  │
Output x'
```

### Residual Connections

The `+ (residual add)` step adds the input directly to the sub-layer's output. This is not optional — without residual connections, training very deep networks fails because gradients vanish or explode. Residuals create a "highway" for gradients to flow backward through hundreds of layers during training.

### Layer Count Across Models

| Model | Layers (transformer blocks) | d_model | Approx. parameters |
|-------|----------------------------|---------|-------------------|
| GPT-2 | 12 | 768 | 117M |
| GPT-3 | 96 | 12,288 | 175B |
| Llama 3 8B | 32 | 4,096 | 8B |
| Llama 3 70B | 80 | 8,192 | 70B |

### Parameter Count Approximation

The dominant cost is the weight matrices. A rough formula:

```
Parameters ≈ 12 × d_model² × n_layers
```

For Llama 3 8B: 12 × 4096² × 32 ≈ 6.4B (the remainder comes from embeddings and other components).

---

## 5. KV Cache

### The Problem Without Caching

Without any optimization, generating token N requires running the full attention computation over all N tokens — recomputing Q, K, and V for every prior token every single time. Generating 1,000 tokens means running the attention stack 1,000 times over growing sequences: O(n²) total computation.

### The Solution: Cache K and V

The Keys and Values for all past tokens do not change — only the new token's Query is new. The KV cache stores all previously computed K and V matrices so they never need to be recomputed:

```
Without KV cache (token 500):
  Compute Q, K, V for tokens 1..500 → attention → output token 500

With KV cache (token 500):
  Load cached K, V for tokens 1..499 from memory
  Compute Q, K, V only for token 500
  Append new K_500, V_500 to cache
  → attention → output token 500
```

This reduces generation from O(n²) to O(n) in compute — but shifts the bottleneck to **memory bandwidth** (reading the entire cache from GPU memory on every step).

### Memory Impact

```
KV cache size = 2 × n_layers × n_heads × d_head × seq_len × bytes_per_param

Example: Llama 3 8B, 4K context, fp16:
  = 2 × 32 × 32 × 128 × 4096 × 2 bytes
  = ~2.1 GB just for the KV cache
```

### What Breaks the Cache

| Cache invalidator | Why |
|-------------------|-----|
| Changing the system prompt mid-conversation | All K/V derived from old system prompt are invalid |
| Inserting content before prior tokens | Token positions shift; cached K/V are at wrong positions |
| Batching requests with different lengths | Padding misalignment invalidates shared prefixes |
| Switching models or quantization level | Different model = different K/V values |

:::note
KV cache is why long-context inference is memory-bound, not compute-bound. As context length grows, each generation step reads more and more data from GPU memory (the cache) but does relatively little computation with it. The bottleneck is memory bandwidth, not FLOPS. This is why FlashAttention and quantized KV caches (8-bit, 4-bit) matter so much for production systems.
:::

---

## 6. Positional Embeddings

### Why Position Must Be Injected

Attention is a set operation by default — it does not care about the order of tokens, only their content. The phrase "dog bites man" and "man bites dog" would produce identical attention scores without positional information. Position must be explicitly injected.

### Four Approaches

**Absolute Sinusoidal (original Transformer, 2017):**

Fixed sin/cos functions of position index. Position p, dimension i:

```
PE(p, 2i)   = sin(p / 10000^(2i/d_model))
PE(p, 2i+1) = cos(p / 10000^(2i/d_model))
```

Added to the token embedding before the first layer. Deterministic, requires no training — but doesn't generalize well beyond the maximum training length.

**Learned Absolute (GPT-2, GPT-3):**

One learned vector per position, stored in a table. Works well within training length, but cannot extrapolate to longer sequences (the model has never seen positions beyond its training maximum).

**Rotary Position Embeddings — RoPE (Llama, Mistral):**

Instead of adding a position vector, RoPE rotates the Q and K vectors for each token by an angle proportional to its position. The key insight: when you take the dot product Q · K, the rotation cancels in a way that the score depends on the *relative* position (j − i) rather than absolute positions. This generalizes better to longer sequences and is the dominant approach in 2024–2025 models.

**ALiBi (PaLM, MPT):**

Adds a fixed, position-dependent bias (−m × |i − j|) to attention scores. Requires no positional computation at embedding time. Empirically generalizes to longer contexts than training length.

### Comparison

| Method | How | Extrapolation | Used in |
|--------|-----|--------------|---------|
| Absolute sinusoidal | Add fixed sin/cos to embedding | Poor | Original Transformer |
| Learned absolute | Add learned position vector | None (hard cutoff) | GPT-2, GPT-3 |
| RoPE | Rotate Q and K by position angle | Good (with YaRN/NTK scaling) | Llama, Mistral, Qwen |
| ALiBi | Bias attention scores by distance | Excellent | PaLM, MPT |

---

## 7. Scaling Laws

### Kaplan et al. 2020 (OpenAI)

Loss scales as a **power law** with model size (N), dataset size (D), and compute (C):

```
L(N) ∝ N^(-α)    (loss decreases as you add parameters)
L(D) ∝ D^(-β)    (loss decreases as you add training tokens)
L(C) ∝ C^(-γ)    (loss decreases as you add compute)
```

This means you can predict model quality before training — a powerful property when training runs cost millions of dollars.

### Chinchilla (Hoffmann et al. 2022, DeepMind)

Kaplan et al. over-emphasized model size. Chinchilla showed the compute-optimal ratio is approximately **20 training tokens per model parameter**:

```
Compute-optimal training:
  70B parameter model → train on ~1.4 trillion tokens
  7B parameter model  → train on ~140 billion tokens
```

Before Chinchilla, many large models were "undertrained" — too few tokens for their size. After Chinchilla: Llama 2 7B was trained on 2T tokens (more than Chinchilla-optimal), trading training compute for better inference efficiency.

### Practical Implications

```
For a fixed compute budget C:
  Before Chinchilla: train the largest model you can afford
  After Chinchilla:  train a smaller model on proportionally more data

Why this matters for practitioners:
  A smaller, well-trained model is often better AND cheaper to run than
  a larger, undertrained model with similar training cost.
```

:::tip
Scaling laws let companies predict model quality before training, saving enormous amounts of compute on failed experiments. If your extrapolated loss curve doesn't hit target before you run out of budget, you know to adjust *before* spending the money.
:::

---

## Common Mistakes

| Mistake | Why it happens | Fix |
|---------|----------------|-----|
| "Attention = understanding" | Anthropomorphizing the mechanism | Attention is a weighted average of values based on query-key similarity — a mathematical operation, not comprehension |
| Ignoring KV cache in production | Performance oversight | Monitor cache hit rate; avoid mutating system prompts mid-session |
| Treating all layers equally | Oversimplification of the architecture | Early layers handle syntax and local patterns; later layers handle semantics, reasoning, and long-range dependencies |
| Assuming bigger always means better | Misreading scaling laws | Chinchilla shows compute-optimal models are smaller and better-trained; a 7B model on 2T tokens often beats a 70B model on 200B tokens |
| Confusing context length and memory | Architectural confusion | KV cache grows with context length; longer context does not mean better retention — attention can still fail to retrieve content in the middle |

---

## Quiz

> **Q1: Why does the attention formula divide by √d_k?**
>
> <details><summary>Show Answer</summary>
>
> When d_k is large (e.g., 128), the dot product Q · K grows proportionally in magnitude. Large values push the softmax function into near-saturation — the output distribution becomes nearly one-hot, and the gradient of softmax approaches zero. Dividing by √d_k keeps the scores in a regime where softmax gradients remain healthy during training.
>
> </details>

---

> **Q2: A user complains that a long-context response is slow even though the first token arrives quickly. What is likely happening?**
>
> <details><summary>Show Answer</summary>
>
> Time-to-first-token (TTFT) is fast because the KV cache stores the prior context — prefill is a single large matrix multiplication that GPUs are good at. Decode speed is slower because each new token requires reading the entire KV cache from GPU memory to compute attention. As the cache grows (long context), the memory bandwidth required per token increases linearly, slowing the decode rate (tokens per second). The bottleneck is memory bandwidth, not FLOPS.
>
> </details>

---

> **Q3: Why can't a standard transformer process a 200K-token context window as fast as an 8K one?**
>
> <details><summary>Show Answer</summary>
>
> Two reasons: (1) The attention score matrix is n × n, so attention computation scales as O(n²). A 200K sequence is 625× more attention computation than an 8K sequence (200,000² / 8,000² = 625). (2) The KV cache for 200K tokens is 625× larger, requiring proportionally more GPU memory to store and more memory bandwidth to read on every decode step. Both effects compound: 200K context is not just 25× more expensive than 8K — it is hundreds of times more expensive.
>
> </details>

---

## Summary Table

| Concept | What it is | Why it matters |
|---------|-----------|----------------|
| Embedding table | Learned lookup: token ID → float vector | Foundation of all transformer computation |
| Self-attention | Weighted average of values based on query-key similarity | Enables direct token-to-token routing at any distance |
| Multi-head attention | H parallel attention operations, concatenated | Different heads specialize in different relationship types |
| Causal mask | −∞ added to future positions before softmax | Ensures generation is left-to-right; prevents future peeking |
| Feed-forward layer | Per-position 2-layer MLP (expand then contract) | Non-linearity + feature transformation after attention |
| Residual connection | Input added directly to sub-layer output | Critical for gradient flow through deep networks |
| KV cache | Stores past K and V to avoid recomputation | Reduces generation from O(n²) to O(n) in compute |
| RoPE | Rotates Q/K by position angle | Encodes relative position; generalizes to longer contexts |
| Scaling laws | Loss scales predictably with N, D, C | Predict model quality before training; guide resource allocation |

---

## Next Steps

→ [Hands-On: Transformer Internals Exercises](./hands-on)
