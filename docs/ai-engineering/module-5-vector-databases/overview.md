---
sidebar_position: 2
title: "Overview"
description: ANN search internals, HNSW and IVF algorithms, metadata filtering, hybrid search with BM25, quantization, and a production vector database comparison.
---

# Vector Databases & Search Systems — Deep Dive

Every RAG pipeline, semantic search product, and recommendation engine has a vector database at its core. When you called `collection.query()` in Module 4, you triggered a chain of non-trivial engineering: graph traversal, cluster lookup, score computation, and metadata filtering. This module explains each step — so you can tune it, debug it, and choose the right database for the job.

---

## 1. What Vector Databases Actually Do

A vector database has four jobs:

```
1. STORE    — Persist high-dimensional float vectors (and their associated metadata)
2. INDEX    — Build a data structure that enables fast similarity queries
3. QUERY    — Given a query vector, return the top-K most similar stored vectors
4. FILTER   — Restrict results to records matching metadata predicates (e.g., date > 2024)
```

A regular relational database (Postgres, MySQL) can store float arrays but cannot do step 2 efficiently. A naive linear scan through 10 million 1536-dimensional vectors would take seconds per query — too slow for interactive use. Vector databases exist to solve step 2.

---

## 2. Why Exact Search Doesn't Scale

### The Brute-Force Approach

Exact nearest-neighbor search: compute the similarity between the query vector and every vector in the index, return the top-K.

```
Query vector:  [0.1, 0.4, 0.9, ...]   (1536 dimensions)
Index size:    10,000,000 vectors

Operations:    10,000,000 × 1536 dot products per query
At 10ns/op:   ~150 ms per query

At 1 billion vectors: ~15,000 ms (15 seconds) per query
```

Acceptable for 10K vectors, unusable at 100M+. The solution: **Approximate Nearest Neighbor (ANN)** — trade a small amount of accuracy for massive speed gains.

### The Recall-Speed Tradeoff

```
Exact search:   recall = 1.00 (perfect)  |  latency = high
ANN search:     recall = 0.95–0.99       |  latency = 1–10ms

Recall@K: fraction of true top-K results that appear in ANN top-K results
```

At recall = 0.95, the ANN index returns 95% of what exact search would — meaning 1 in 20 queries has a slightly wrong best result. In practice, this is imperceptible for RAG (the retrieved chunks are still highly relevant).

---

## 3. HNSW — The Most Widely-Used ANN Index

### Conceptual Walkthrough

HNSW stands for **Hierarchical Navigable Small World**. It builds a multi-layer graph where:
- **Upper layers**: sparse, long-range connections — fast coarse navigation
- **Lower layers**: dense, short-range connections — precise local search
- **Bottom layer**: every vector is present; this is where final candidates come from

```
Layer 2 (sparse):     A ————————————————— G
                                |
Layer 1 (medium):     A — B — C — D — E — F — G
                              |
Layer 0 (dense):   A-B-C-D-E-F-G-H-I-J-K-L-M-N

Query: find nearest neighbor to Q
  1. Enter at top layer at arbitrary entry point
  2. Greedily move to the neighbor closest to Q
  3. Drop to next layer at current position
  4. Repeat until bottom layer
  5. Collect candidates in bottom layer neighborhood
```

### HNSW Parameters That Matter

| Parameter | What it controls | Default | Tuning direction |
|-----------|----------------|---------|-----------------|
| `M` | Max edges per node (graph connectivity) | 16 | ↑ = higher recall, more memory, slower build |
| `ef_construction` | Candidate list size during index build | 200 | ↑ = higher recall at build time, slower indexing |
| `ef_search` | Candidate list size during query | 50 | ↑ = higher recall at query time, slower search |
| `space` | Distance metric (`cosine`, `l2`, `ip`) | `cosine` | Match your embedding model's output |

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection(
    "my-index",
    metadata={
        "hnsw:space": "cosine",
        "hnsw:construction_ef": 200,   # higher = better recall during indexing
        "hnsw:search_ef": 100,         # higher = better recall during queries (tunable at runtime)
        "hnsw:M": 16,                  # connections per node
    },
)
```

### HNSW Memory Usage

```
Memory per vector ≈ (dimension × 4 bytes) + (M × 2 × 8 bytes)

For text-embedding-3-small (1536 dims), M=16:
  = (1536 × 4) + (16 × 2 × 8)
  = 6144 + 256
  = ~6.4 KB per vector

At 10M vectors: ~64 GB RAM required
```

This is the primary cost of HNSW — it's an in-memory index. For datasets > 100M vectors, consider IVF or quantization (covered in sections 4 and 6).

---

## 4. IVF — The Memory-Efficient Alternative

### How Inverted File Index Works

IVF takes a different approach: instead of a graph, it clusters vectors and builds an inverted index over those clusters.

```
TRAINING PHASE (offline):
  1. Run k-means clustering on a sample of vectors → get C cluster centroids
  2. For each vector, assign it to its nearest centroid
  3. Store vectors grouped by cluster

QUERY PHASE:
  1. Compute distance from query to all C centroids (fast — C << N)
  2. Select nprobe closest centroids
  3. Search only the vectors in those nprobe clusters
  4. Return top-K from that subset
```

### HNSW vs IVF Tradeoffs

| Dimension | HNSW | IVF |
|-----------|------|-----|
| **Query latency** | Faster (1–5ms) | Slower (5–50ms for large nprobe) |
| **Memory** | High (full index in RAM) | Lower (can use disk for vectors) |
| **Build time** | Fast (no training phase) | Slower (k-means training required) |
| **Recall control** | `ef_search` parameter | `nprobe` parameter |
| **Best for** | < 100M vectors, latency-critical | > 100M vectors, memory-constrained |
| **Used by** | ChromaDB, Qdrant (default) | FAISS (classic), Pinecone (internally) |

:::tip Choosing Between Them
For most RAG applications with < 10M chunks, HNSW is the right default — it's faster and simpler to tune. Use IVF when your dataset exceeds 100M vectors or your RAM budget is tight.
:::

---

## 5. Metadata Filtering

### The Problem With Naive Filtering

Many RAG applications need to filter by metadata: "only retrieve documents from 2024", "only from department=engineering", "only file_type=contract". The naive approach — retrieve top-K vectors, then filter by metadata — has a fatal flaw:

```
Retrieve top-10 vectors
Filter: keep only those where year == 2024

If only 2 of the top-10 match the filter, you get 2 results instead of 10.
If 0 of the top-10 match, you get 0 results — even if matches exist.
```

### How Production Vector DBs Handle Filtering

| Approach | How it works | When to use |
|----------|-------------|------------|
| **Pre-filter** | Filter the index to only consider matching records before search | High-selectivity filters (< 1% of data matches) |
| **Post-filter** | Retrieve oversized top-K (e.g., top-100), then filter | Low-selectivity filters (> 50% of data matches) |
| **In-filter (HNSW)** | Visit only graph nodes that match the filter during traversal | Medium selectivity; Qdrant does this well |

Qdrant's payload filtering is particularly well-designed — it integrates filter predicates into the HNSW graph traversal rather than as a post-processing step.

```python
# Qdrant example: filter by metadata during vector search
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchRequest

client = QdrantClient(":memory:")

results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    query_filter=Filter(
        must=[
            FieldCondition(key="year", match=MatchValue(value=2024)),
            FieldCondition(key="department", match=MatchValue(value="engineering")),
        ]
    ),
    limit=5,
)
```

---

## 6. Hybrid Search

### When Vector Search Alone Fails

Semantic similarity captures meaning, but misses exact keyword matches. Consider:

- Query: "What is our Q3 2024 EBITDA?"
- Document contains: "Q3 2024 EBITDA was $42.3M"

The word "EBITDA" is rare enough that semantic similarity may not rank this document first. **BM25 keyword search** would rank it first instantly — it rewards exact term overlap.

The solution: combine both signals.

### BM25 — Keyword Relevance Scoring

BM25 (Best Match 25) is the standard keyword retrieval algorithm used in search engines:

```
BM25 score = Σ IDF(t) × (tf(t,d) × (k1+1)) / (tf(t,d) + k1 × (1-b+b×|d|/avgdl))

Where:
  t    = query term
  d    = document
  tf   = term frequency in document
  IDF  = inverse document frequency (rare terms score higher)
  |d|  = document length
  k1, b = tuning parameters (k1=1.5, b=0.75 are standard defaults)
```

In plain terms: BM25 scores high when query terms appear frequently in the document but are rare across the corpus. This is what search engines have used for decades.

### Reciprocal Rank Fusion (RRF)

RRF combines ranked lists from multiple retrieval sources without needing to normalize their scores (which have different scales and distributions):

```
RRF_score(d) = Σ 1 / (k + rank_i(d))

Where:
  k           = smoothing constant (typically 60)
  rank_i(d)   = rank of document d in retrieval system i
```

A document that ranks 1st in vector search and 3rd in BM25 scores much higher than one that ranks 5th in both — even if their raw similarity scores differ dramatically.

```python
from rank_bm25 import BM25Okapi

def hybrid_search(
    query: str,
    corpus: list[str],
    vector_results: list[tuple[int, float]],  # (doc_idx, similarity)
    top_k: int = 5,
    k: int = 60,
) -> list[int]:
    # BM25 retrieval
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)

    # Build rank maps
    vector_rank = {doc_idx: rank + 1 for rank, (doc_idx, _) in enumerate(vector_results)}
    bm25_rank = {doc_idx: rank + 1 for rank, (doc_idx, _) in enumerate(bm25_ranked[:100])}

    # Collect all candidate docs
    all_docs = set(vector_rank.keys()) | set(bm25_rank.keys())

    # Compute RRF scores
    rrf_scores = {
        doc_idx: (1 / (k + vector_rank.get(doc_idx, 1000))) +
                 (1 / (k + bm25_rank.get(doc_idx, 1000)))
        for doc_idx in all_docs
    }

    # Return top-K by RRF score
    return [doc_idx for doc_idx, _ in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]]
```

### When Hybrid Search Wins vs Vector-Only

| Scenario | Winner | Why |
|----------|--------|-----|
| Exact product codes, names, IDs | BM25 | Semantic embedding of "SKU-8834" is arbitrary |
| Conceptual/paraphrase queries | Vector | "Show me something about memory" → no exact keyword |
| Technical jargon (EBITDA, API names) | BM25 | Rare terms get high IDF score |
| General Q&A | Hybrid (RRF) | Best of both worlds |

---

## 7. Quantization

### The Size Problem

Storing 10 million vectors in float32 (4 bytes per dimension, 1536 dimensions):

```
10M × 1536 × 4 bytes = 61.4 GB RAM
```

That's expensive. Quantization reduces precision to reduce size:

| Precision | Bytes/dim | Size for 10M × 1536 | Recall impact |
|-----------|----------|---------------------|---------------|
| FP32 (default) | 4 | 61.4 GB | Baseline |
| FP16 | 2 | 30.7 GB | ~0% loss |
| INT8 (scalar quant) | 1 | 15.4 GB | ~1–2% loss |
| INT4 | 0.5 | 7.7 GB | ~3–5% loss |
| Binary | 0.125 | 1.9 GB | ~5–15% loss |

### Scalar Quantization (INT8)

The simplest approach: scale each float to the range [0, 255] and store as a uint8.

```python
import numpy as np

def quantize_int8(vectors: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Quantize float32 vectors to int8. Returns (quantized, min_val, scale)."""
    min_val = vectors.min()
    max_val = vectors.max()
    scale = (max_val - min_val) / 255.0
    quantized = ((vectors - min_val) / scale).clip(0, 255).astype(np.uint8)
    return quantized, min_val, scale

def dequantize_int8(quantized: np.ndarray, min_val: float, scale: float) -> np.ndarray:
    return quantized.astype(np.float32) * scale + min_val

# Example
original = np.random.randn(100, 1536).astype(np.float32)
quantized, min_val, scale = quantize_int8(original)

size_original = original.nbytes / (1024 ** 2)
size_quantized = quantized.nbytes / (1024 ** 2)
print(f"Original: {size_original:.1f} MB → Quantized: {size_quantized:.1f} MB")
print(f"Compression ratio: {size_original / size_quantized:.1f}×")
```

### Product Quantization (PQ) — Conceptual

PQ divides the vector into M sub-vectors and quantizes each independently using a learned codebook. It achieves much higher compression ratios (64–512×) at the cost of more setup complexity.

```
Original vector:    [d₁, d₂, ..., d₁₅₃₆]   (FP32, 6144 bytes)

Split into 8 sub-vectors of 192 dims each:
  Sub-vector 1:    [d₁...d₁₉₂]    → nearest codebook entry (index 1–256)  → 1 byte
  Sub-vector 2:    [d₁₉₃...d₃₈₄]  → nearest codebook entry                → 1 byte
  ...
  Sub-vector 8:    → 1 byte

Compressed vector: [i₁, i₂, ..., i₈]   (8 bytes instead of 6144 — 768× compression)
```

Qdrant, FAISS, and Pinecone all support PQ. For most RAG applications, scalar INT8 quantization is sufficient.

---

## 8. Production Database Comparison

Choosing a vector database is an infrastructure decision, not just a technical one:

| Database | Best for | Hosting | Filtering | Hybrid search | Notes |
|----------|----------|---------|-----------|---------------|-------|
| **Pinecone** | Managed production; no ops burden | Cloud-only | Good (namespaces + metadata) | Via sparse-dense | Can't self-host; expensive at scale |
| **Qdrant** | Self-host or cloud; excellent filtering | Both | Excellent (payload filtering) | Built-in | Best overall for RAG use cases |
| **Weaviate** | Multi-modal, GraphQL interface | Both | Good | BM25 built-in | Module-based architecture |
| **pgvector** | Already on Postgres; < 5M vectors | Self-host | Native SQL | via SQL | Degrades at scale; no ANN beyond HNSW |
| **ChromaDB** | Local dev, prototyping, notebooks | Local/embedded | Basic | No | Not production-grade; data durability issues |
| **Milvus** | Very large scale (> 100M vectors) | Self-host/cloud | Good | No native | Complex ops; best for billion-scale |

### Decision Framework

```
Are you prototyping or building an internal tool?
  → ChromaDB (zero setup)

Are you already on Postgres and < 5M vectors?
  → pgvector (no new infrastructure)

Do you need managed hosting with zero ops?
  → Pinecone

Do you need self-hosting + excellent filtering + hybrid search?
  → Qdrant

Are you building a multi-modal system or need GraphQL?
  → Weaviate

Are you operating at billion-vector scale?
  → Milvus or Pinecone enterprise
```

---

## Common Mistakes

1. **Using ChromaDB in production** — it lacks durability guarantees and degrades beyond a few million vectors. Use Qdrant or Pinecone for production.
2. **Post-filtering without over-retrieval** — if you filter after retrieving top-5, you might get 0 results. Always retrieve 3–10× your target K when filtering.
3. **Ignoring `ef_search`** — the default `ef_search` in HNSW trades recall for speed. Increase it if retrieval quality is poor.
4. **Not using hybrid search for keyword-heavy queries** — exact identifiers (SKUs, names, codes) do not embed well. Add BM25.
5. **Skipping quantization at scale** — at 10M+ vectors, INT8 quantization halves your RAM requirement with < 2% recall loss.
6. **Not benchmarking recall** — tune `ef_search` and `M` against a recall benchmark on your actual data, not just defaults.

---

## Quiz

<details>
<summary>Q1: You have 50 million vectors and need query latency under 20ms. Should you use HNSW or IVF, and why?</summary>

HNSW is likely the better choice if you have sufficient RAM. At 50M vectors with 1536 dimensions and INT8 quantization, the index fits in roughly 75GB — within reach of a large cloud instance. HNSW queries are consistently 1–10ms. IVF can achieve similar latency only with a large `nprobe` (many clusters searched), which loses its memory advantage. If RAM is truly constrained, IVF with product quantization is the alternative.

</details>

<details>
<summary>Q2: A user asks "Show me all support tickets about JIRA-4421 from Q1 2024." Your vector search returns unrelated results. What two systems should you combine?</summary>

BM25 keyword search + vector search (hybrid with RRF). BM25 will rank exact matches on "JIRA-4421" first — embedding-based similarity is poor at matching exact alphanumeric identifiers. Also add a metadata filter for `date >= 2024-01-01 AND date < 2024-04-01` to enforce the Q1 2024 constraint.

</details>

<details>
<summary>Q3: What is the key difference between scalar quantization and product quantization?</summary>

Scalar quantization maps each individual float dimension to an integer (e.g., 4 bytes → 1 byte), achieving ~4× compression. Product quantization splits the vector into sub-vectors, then maps each sub-vector to a learned codebook entry (a single byte index), achieving much higher compression (64–512×) at the cost of: (1) a training phase to learn the codebooks, and (2) slightly higher reconstruction error.

</details>

---

## Next Steps

→ **[Hands-On: Vector Databases](./hands-on)** — Index 10K documents and measure recall, implement BM25 + vector hybrid search with RRF, benchmark HNSW vs flat search, and apply 8-bit quantization.

→ **[Module 6: Model Optimization](/learn/ai-engineering/module-6-model-optimization)** — Having mastered the retrieval layer, learn how to optimize the generation layer: quantization for LLMs, batching, streaming, speculative decoding, and the cost/quality tradeoffs between model sizes.
