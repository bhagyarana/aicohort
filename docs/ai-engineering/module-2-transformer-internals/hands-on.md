---
sidebar_position: 3
title: "Hands-On"
description: Word embedding geometry, attention visualization with BertViz, KV cache benchmarking, and building scaled dot-product attention from scratch.
---

# Hands-On: Transformer Internals

These four exercises go from intuition-building (embedding geometry) to implementation (building an attention head from scratch). Exercises 1 and 4 require only NumPy. Exercises 2 and 3 require a working Python environment; a GPU speeds up Exercise 3 but is not required.

---

## Exercise 1: Word Embedding Geometry (Beginner)

**Goal:** Explore how learned embeddings encode semantic relationships geometrically.
**Time:** ~25 min

### Setup

```bash
pip install numpy gensim
```

### Step 1 — Load GloVe embeddings and compute cosine similarity

```python
import numpy as np
from gensim.downloader import load as gensim_load

# Download ~66 MB GloVe vectors trained on Wikipedia + Gigaword
# This takes a minute the first time; cached afterward
print("Loading GloVe vectors...")
glove = gensim_load("glove-wiki-gigaword-100")   # 100-dim vectors
print(f"Vocabulary size: {len(glove):,}")
print(f"Vector dimension: {glove.vector_size}")

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Explore pairwise similarities
pairs = [
    ("king",   "queen"),
    ("king",   "man"),
    ("queen",  "woman"),
    ("paris",  "france"),
    ("berlin", "germany"),
    ("cat",    "dog"),
    ("cat",    "automobile"),
    ("happy",  "sad"),
]

print("\nCosine similarities:")
print(f"{'Pair':30s}  {'Similarity':>10}")
print("-" * 44)
for w1, w2 in pairs:
    sim = cosine_similarity(glove[w1], glove[w2])
    print(f"{w1 + ' — ' + w2:30s}  {sim:10.4f}")
```

### Step 2 — Word arithmetic: King − Man + Woman ≈ Queen

```python
def find_similar(positive: list[str], negative: list[str], topn: int = 5):
    """
    Compute: sum(positive vectors) - sum(negative vectors)
    Return the top-n closest words to the result.
    """
    result_vector = (
        sum(glove[w] for w in positive) -
        sum(glove[w] for w in negative)
    )
    # gensim's most_similar does this internally; we do it manually for clarity
    exclude = set(positive + negative)
    similarities = []
    for word in glove.key_to_index:
        if word not in exclude:
            sim = cosine_similarity(result_vector, glove[word])
            similarities.append((word, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:topn]

print("\nWord arithmetic results:")
print("king - man + woman:")
for word, sim in find_similar(["king", "woman"], ["man"]):
    print(f"  {word:15s} {sim:.4f}")

print("\nparis - france + germany:")
for word, sim in find_similar(["paris", "germany"], ["france"]):
    print(f"  {word:15s} {sim:.4f}")

print("\nfast - speed + slow:")
for word, sim in find_similar(["fast", "slow"], ["speed"]):
    print(f"  {word:15s} {sim:.4f}")
```

### Step 3 — Similarity heatmap for related words

```python
words = ["king", "queen", "man", "woman", "prince", "princess",
         "dog", "cat", "paris", "london"]

n = len(words)
matrix = np.zeros((n, n))
for i, w1 in enumerate(words):
    for j, w2 in enumerate(words):
        matrix[i][j] = cosine_similarity(glove[w1], glove[w2])

# Print as ASCII heatmap (no matplotlib needed)
print("\nSimilarity heatmap (higher = more similar):")
print(f"{'':12s}", end="")
for w in words:
    print(f"{w[:5]:>7}", end="")
print()

for i, w1 in enumerate(words):
    print(f"{w1:12s}", end="")
    for j in range(n):
        val = matrix[i][j]
        # Map to symbols: ░ < 0.3, ▒ 0.3–0.6, ▓ 0.6–0.8, █ > 0.8
        if val > 0.8:
            sym = " ████"
        elif val > 0.6:
            sym = " ▓▓▓▓"
        elif val > 0.3:
            sym = " ▒▒▒▒"
        else:
            sym = " ░░░░"
        print(sym, end="")
    print()
```

### Step 4 — Test semantic clusters

```python
# Verify that semantic clusters are visible in the embedding space
clusters = {
    "royalty":   ["king", "queen", "prince", "princess", "throne"],
    "geography": ["paris", "london", "berlin", "rome", "tokyo"],
    "animals":   ["dog", "cat", "horse", "lion", "tiger"],
    "colors":    ["red", "blue", "green", "yellow", "purple"],
}

print("\nIntra-cluster vs inter-cluster similarity:")
for cluster_name, cluster_words in clusters.items():
    intra_sims = []
    for i, w1 in enumerate(cluster_words):
        for j, w2 in enumerate(cluster_words):
            if i < j:
                intra_sims.append(cosine_similarity(glove[w1], glove[w2]))
    print(f"  {cluster_name:12s}: avg intra-cluster similarity = {np.mean(intra_sims):.4f}")
```

### What to verify

- `king — queen` similarity is > 0.7
- `king — man` similarity is greater than `queen — man` similarity (king is closer to man than queen is)
- Word arithmetic `king - man + woman` returns "queen" in the top 3 results
- Intra-cluster similarities are consistently higher than the similarity between different clusters

---

## Exercise 2: Visualizing Attention (Intermediate)

**Goal:** See real attention patterns in a transformer and understand what different heads learn.
**Time:** ~30 min

### Setup

```bash
pip install transformers torch bertviz
```

### Step 1 — Load BERT and visualize attention for an ambiguous sentence

```python
from bertviz import head_view
from transformers import BertTokenizer, BertModel
import torch

# Load BERT — small enough to run on CPU
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
model.eval()

def get_attention(sentence: str):
    """Returns tokens and attention tensors for BertViz."""
    inputs = tokenizer.encode_plus(sentence, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    with torch.no_grad():
        outputs = model(**inputs)

    attention = outputs.attentions   # tuple of (1, n_heads, seq_len, seq_len) per layer
    return tokens, attention

# Classic disambiguation sentence
sentence1 = "The bank can guarantee deposits will eventually cover future tuition costs"
tokens1, attention1 = get_attention(sentence1)
print(f"Tokens: {tokens1}")
print(f"Number of layers: {len(attention1)}")
print(f"Attention shape per layer: {attention1[0].shape}")  # (1, 12, seq_len, seq_len)

# Visualize in a Jupyter notebook or save HTML:
# head_view(attention1, tokens1)
```

### Step 2 — Extract and print attention weights for specific heads

```python
import numpy as np

def print_attention_head(tokens, attention, layer: int, head: int):
    """Print the attention weight matrix for one head as a table."""
    attn = attention[layer][0, head].detach().numpy()
    n = len(tokens)

    print(f"\nLayer {layer}, Head {head} attention weights:")
    print(f"{'':12s}", end="")
    for t in tokens:
        print(f"{t[:5]:>7}", end="")
    print()

    for i, t_from in enumerate(tokens):
        print(f"{t_from:12s}", end="")
        for j in range(n):
            val = attn[i, j]
            if val > 0.3:
                sym = f"{val:.2f} "
            else:
                sym = "  .   "
            print(f"{sym:>7}", end="")
        print()

# Inspect a few heads in early vs late layers
# Early layers (0-3) often show diagonal (each token attends to itself)
# Later layers (8-11) show longer-range dependencies
print_attention_head(tokens1, attention1, layer=0,  head=0)
print_attention_head(tokens1, attention1, layer=10, head=5)
```

### Step 3 — Compare attention for pronoun resolution

```python
# "it" is ambiguous — which entity does it refer to?
sentence2 = "The trophy did not fit in the suitcase because it was too big"
sentence3 = "The trophy did not fit in the suitcase because it was too small"

tokens2, attention2 = get_attention(sentence2)
tokens3, attention3 = get_attention(sentence3)

def attention_from_token(tokens, attention, token_str: str, layer: int = 10):
    """Show what a specific token attends to, averaged across all heads."""
    idx = next(i for i, t in enumerate(tokens) if t == token_str)
    # Average across heads
    avg_attn = attention[layer][0].mean(dim=0).detach().numpy()
    attn_row = avg_attn[idx]

    print(f"\n'{token_str}' attention (layer {layer}, avg heads):")
    for i, (tok, weight) in enumerate(zip(tokens, attn_row)):
        bar = "█" * int(weight * 50)
        print(f"  {tok:12s} {weight:.3f} {bar}")

# Compare where "it" attends in the two sentences
# In sentence2 "it was too big" → "it" should refer to the trophy (big trophy)
# In sentence3 "it was too small" → "it" should refer to the suitcase (small suitcase)
attention_from_token(tokens2, attention2, "it")
attention_from_token(tokens3, attention3, "it")
```

### Step 4 — Quantify head specialization

```python
def head_diagonal_score(attention, layer: int, head: int) -> float:
    """
    A head is 'diagonal' if each token mostly attends to itself.
    Returns the average weight on the diagonal (0 = no self-attention, 1 = pure self-attention).
    """
    attn = attention[layer][0, head].detach().numpy()
    return float(np.mean(np.diag(attn)))

print("\nDiagonal (self-attention) score per head in layer 0:")
print("Higher = each token mostly attends to itself")
for head in range(12):
    score = head_diagonal_score(attention1, layer=0, head=head)
    bar = "█" * int(score * 40)
    print(f"  Head {head:2d}: {score:.3f} {bar}")
```

### What to verify

- In layer 0, some heads show strong diagonal patterns (high self-attention scores)
- In later layers (8–11), heads show long-range dependencies
- The token `[CLS]` often receives high attention in later layers (it aggregates sequence information)
- "it" attends differently between the "too big" and "too small" sentences in at least some heads

---

## Exercise 3: KV Cache Benchmark (Intermediate)

**Goal:** Measure the inference speedup from the KV cache and observe how memory scales.
**Time:** ~25 min (faster with a GPU; CPU is slow but works)

### Setup

```bash
pip install transformers torch
```

### Step 1 — Load GPT-2 and define a timed generation function

```python
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"   # ~500 MB, manageable on CPU
print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Using device: {device}")

def generate_timed(prompt: str, max_new_tokens: int, use_cache: bool) -> tuple[str, float]:
    """Generate text and return (output, wall_time_seconds)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    n_input = inputs["input_ids"].shape[1]

    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache,
            do_sample=False,          # greedy for reproducibility
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - start

    output_text = tokenizer.decode(output_ids[0][n_input:], skip_special_tokens=True)
    return output_text, elapsed
```

### Step 2 — Compare generation speed with and without cache

```python
prompt = "The transformer architecture was introduced in 2017 and has since"
max_new_tokens = 200

print(f"\nGenerating {max_new_tokens} tokens...")
print(f"Prompt: '{prompt}'")

_, time_cached    = generate_timed(prompt, max_new_tokens, use_cache=True)
_, time_no_cache  = generate_timed(prompt, max_new_tokens, use_cache=False)

tps_cached    = max_new_tokens / time_cached
tps_no_cache  = max_new_tokens / time_no_cache
speedup       = time_no_cache / time_cached

print(f"\nWith KV cache:    {time_cached:.2f}s  ({tps_cached:.1f} tokens/sec)")
print(f"Without KV cache: {time_no_cache:.2f}s  ({tps_no_cache:.1f} tokens/sec)")
print(f"Speedup:          {speedup:.1f}×")
```

### Step 3 — Plot tokens-per-second vs context length

```python
# Measure decode speed at different context lengths (simulated by varying prompt length)
context_lengths = [16, 64, 128, 256, 512]
results = []

for ctx_len in context_lengths:
    # Create a prompt of approximately ctx_len tokens
    base = "The history of artificial intelligence is long and fascinating. "
    tokens_so_far = tokenizer(base, return_tensors="pt")["input_ids"].shape[1]
    repeats = max(1, ctx_len // tokens_so_far)
    padded_prompt = base * repeats

    _, t_cache    = generate_timed(padded_prompt, max_new_tokens=50, use_cache=True)
    _, t_no_cache = generate_timed(padded_prompt, max_new_tokens=50, use_cache=False)

    actual_len = tokenizer(padded_prompt, return_tensors="pt")["input_ids"].shape[1]
    tps_c  = 50 / t_cache
    tps_nc = 50 / t_no_cache
    results.append((actual_len, tps_c, tps_nc))
    print(f"Context {actual_len:4d} tokens | cache: {tps_c:5.1f} t/s | no cache: {tps_nc:5.1f} t/s")

# ASCII plot
print("\nTokens/sec vs context length:")
print("  (C) = with cache, (N) = without cache\n")
for ctx, tps_c, tps_nc in results:
    bar_c  = "█" * int(tps_c / 2)
    bar_nc = "░" * int(tps_nc / 2)
    print(f"  {ctx:4d}t  C: {bar_c} {tps_c:.1f}")
    print(f"         N: {bar_nc} {tps_nc:.1f}")
    print()
```

### Step 4 — Estimate KV cache memory usage

```python
def kv_cache_size_mb(
    n_layers: int,
    n_heads: int,
    d_head: int,
    seq_len: int,
    bytes_per_param: int = 2,   # fp16
) -> float:
    """Estimate KV cache memory in MB."""
    total_bytes = 2 * n_layers * n_heads * d_head * seq_len * bytes_per_param
    return total_bytes / (1024 ** 2)

# GPT-2 architecture
gpt2_config = dict(n_layers=12, n_heads=12, d_head=64)

print("\nKV cache memory vs sequence length (GPT-2, fp16):")
print(f"{'Seq len':>10} {'Cache (MB)':>12}")
print("-" * 25)
for seq_len in [512, 1024, 2048, 4096, 8192, 16384]:
    mb = kv_cache_size_mb(**gpt2_config, seq_len=seq_len)
    bar = "█" * int(mb * 5)
    print(f"{seq_len:>10,} {mb:>10.2f} MB  {bar}")
```

### What to verify

- KV cache provides a measurable speedup (expect 2–5× on CPU, more on GPU)
- Without cache, decode speed degrades as context length increases (O(n²) effect)
- With cache, decode speed degrades more slowly (O(n) memory reads dominate)
- KV cache memory grows linearly with sequence length

---

## Exercise 4: Build a Minimal Attention Head (Intermediate-Advanced)

**Goal:** Implement scaled dot-product attention from scratch in NumPy to understand every step.
**Time:** ~35 min

### Setup

```bash
pip install numpy   # only dependency
```

### The full implementation

```python
import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over last axis."""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scaled dot-product attention.

    Args:
        Q: (seq_len, d_k) — queries
        K: (seq_len, d_k) — keys
        V: (seq_len, d_v) — values
        mask: (seq_len, seq_len) optional mask; -inf at positions to block

    Returns:
        output:  (seq_len, d_v) — weighted combination of values
        weights: (seq_len, seq_len) — attention weight matrix
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)          # (seq_len, seq_len)

    if mask is not None:
        scores = scores + mask                 # add -inf to masked positions

    weights = softmax(scores)                  # (seq_len, seq_len)
    output  = weights @ V                      # (seq_len, d_v)
    return output, weights

# Example: 4-token sequence, d_k = d_v = 8
np.random.seed(42)
seq_len, d_k, d_v = 4, 8, 8

Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_v)

output, weights = attention(Q, K, V)

print("Bidirectional attention weights (row = query, col = key):")
print(np.round(weights, 3))
print("\nRow sums (should all be 1.0):", np.round(weights.sum(axis=-1), 6))
print("\nOutput shape:", output.shape)
```

### Add causal masking

```python
# Causal mask: block all future positions (upper triangle) with -inf
causal_mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
print("\nCausal mask:")
print(causal_mask)

causal_output, causal_weights = attention(Q, K, V, mask=causal_mask)

print("\nCausal attention weights (lower-triangular):")
print(np.round(causal_weights, 3))
print("\nRow sums (should all be 1.0):", np.round(causal_weights.sum(axis=-1), 6))

# Verify: upper triangle should be exactly 0.0
upper_triangle = np.triu(causal_weights, k=1)
print(f"\nMax value in upper triangle (should be 0.0): {upper_triangle.max():.10f}")
```

### Extend to multi-head attention

```python
def multi_head_attention(
    X: np.ndarray,
    W_Q: list[np.ndarray],
    W_K: list[np.ndarray],
    W_V: list[np.ndarray],
    W_O: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Multi-head attention.

    Args:
        X:   (seq_len, d_model)
        W_Q: list of H weight matrices, each (d_model, d_k)
        W_K: list of H weight matrices, each (d_model, d_k)
        W_V: list of H weight matrices, each (d_model, d_v)
        W_O: output projection (H * d_v, d_model)
        mask: optional causal or padding mask

    Returns:
        output: (seq_len, d_model)
    """
    H = len(W_Q)
    head_outputs = []

    for h in range(H):
        Q_h = X @ W_Q[h]
        K_h = X @ W_K[h]
        V_h = X @ W_V[h]
        head_out, _ = attention(Q_h, K_h, V_h, mask=mask)
        head_outputs.append(head_out)   # (seq_len, d_v)

    # Concatenate all heads along the feature dimension
    concat = np.concatenate(head_outputs, axis=-1)   # (seq_len, H * d_v)
    return concat @ W_O                              # (seq_len, d_model)

# Test multi-head attention: 4-token sequence, d_model=16, H=2 heads
np.random.seed(0)
d_model, H, d_k, d_v_head = 16, 2, 8, 8

X = np.random.randn(seq_len, d_model)
W_Qs = [np.random.randn(d_model, d_k)    for _ in range(H)]
W_Ks = [np.random.randn(d_model, d_k)    for _ in range(H)]
W_Vs = [np.random.randn(d_model, d_v_head) for _ in range(H)]
W_O  = np.random.randn(H * d_v_head, d_model)

mha_output = multi_head_attention(X, W_Qs, W_Ks, W_Vs, W_O)
print(f"\nMulti-head attention output shape: {mha_output.shape}")  # (4, 16)
print("First token output vector (first 8 dims):")
print(np.round(mha_output[0, :8], 4))
```

### Verify the implementation matches expectations

```python
print("\n=== Verification Summary ===")

# Check 1: attention weights sum to 1
row_sums = weights.sum(axis=-1)
assert np.allclose(row_sums, 1.0, atol=1e-6), "Row sums are not 1.0"
print("PASS: Bidirectional attention weights sum to 1.0 per row")

# Check 2: causal weights are lower-triangular
upper = np.triu(causal_weights, k=1)
assert np.allclose(upper, 0.0, atol=1e-10), "Causal weights have non-zero upper triangle"
print("PASS: Causal attention weights are lower-triangular")

# Check 3: multi-head output has correct shape
assert mha_output.shape == (seq_len, d_model), f"Wrong shape: {mha_output.shape}"
print(f"PASS: Multi-head attention output shape is ({seq_len}, {d_model})")

# Check 4: scaling by sqrt(d_k) matters
scores_unscaled = Q @ K.T
scores_scaled   = Q @ K.T / np.sqrt(d_k)
print(f"\nUnscaled score std: {scores_unscaled.std():.3f}")
print(f"Scaled score std:   {scores_scaled.std():.3f}")
print(f"  (scaling reduces std by factor ~√{d_k} = {np.sqrt(d_k):.2f})")
```

### What to verify

- All rows in the attention weight matrix sum to exactly 1.0
- Causal attention weights are strictly lower-triangular (zero in the upper triangle)
- Multi-head attention output shape is `(seq_len, d_model)`
- Unscaled scores have higher variance than scaled scores (demonstrating why √d_k scaling matters)

---

## Checklist

- [ ] Completed Exercise 1: King − Man + Woman returns Queen in top results; intra-cluster similarities are higher than inter-cluster
- [ ] Completed Exercise 2: Identified at least one diagonal attention head and one long-range dependency head
- [ ] Completed Exercise 3: Measured the KV cache speedup; confirmed KV cache memory grows linearly with sequence length
- [ ] Completed Exercise 4: Implemented attention from scratch; all verification checks pass; causal weights are lower-triangular
- [ ] (Optional) Extended Exercise 4 to include a simple feed-forward layer (Linear → GELU → Linear) after attention

---

**Next →** [Resources: Transformer Internals](./resources)
