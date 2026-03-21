---
sidebar_position: 3
title: "Hands-On"
description: Index 10K documents and measure recall, implement BM25+vector hybrid search with RRF, benchmark HNSW vs flat, and apply 8-bit quantization.
---

# Hands-On: Vector Databases & Search Systems

Four exercises that progressively expose the internals of vector databases. All exercises can run locally using ChromaDB and free libraries — no paid vector database account required (though Exercise 3 shows a Qdrant alternative for comparison).

---

## Exercise 1: Index 10K Documents and Measure Recall (Beginner)

**Goal:** Build a 10K-document index, query it at different top-K values, and measure how recall changes.
**Time:** ~35 min

### Setup

```bash
pip install chromadb openai numpy scikit-learn datasets
```

### Step 1 — Generate or load 10K documents

```python
import numpy as np
from datasets import load_dataset
from openai import OpenAI
import chromadb

openai_client = OpenAI()
chroma_client = chromadb.Client()

# Load a public dataset with ~10K short texts
# Using the AG News dataset (news article titles + descriptions)
dataset = load_dataset("ag_news", split="train[:10000]")
documents = [f"{row['text']}" for row in dataset]

print(f"Loaded {len(documents)} documents")
print(f"Sample: {documents[0][:200]}")
```

### Step 2 — Embed and index

```python
def embed_batch(texts: list[str]) -> list[list[float]]:
    response = openai_client.embeddings.create(
        input=texts,
        model="text-embedding-3-small",
    )
    return [item.embedding for item in response.data]

collection = chroma_client.create_collection(
    "ag-news-10k",
    metadata={"hnsw:space": "cosine"},
)

BATCH_SIZE = 100
all_embeddings = []

for i in range(0, len(documents), BATCH_SIZE):
    batch = documents[i : i + BATCH_SIZE]
    batch_embs = embed_batch(batch)
    all_embeddings.extend(batch_embs)
    if i % 500 == 0:
        print(f"Embedded {i}/{len(documents)}")

collection.add(
    ids=[str(i) for i in range(len(documents))],
    embeddings=all_embeddings,
    documents=documents,
    metadatas=[{"label": row["label"], "idx": i} for i, row in enumerate(dataset)],
)

print(f"Indexed {collection.count()} documents")
```

### Step 3 — Measure recall at different top-K values

```python
def get_ground_truth(query_idx: int, embeddings: list[list[float]], top_k: int) -> list[int]:
    """Compute exact nearest neighbors using brute force (ground truth)."""
    query_emb = np.array(embeddings[query_idx])
    all_embs = np.array(embeddings)
    similarities = all_embs @ query_emb  # dot product (unit-normalized = cosine)
    similarities[query_idx] = -1  # exclude self
    return list(np.argsort(similarities)[-top_k:][::-1])

def measure_recall(
    query_indices: list[int],
    embeddings: list[list[float]],
    collection: chromadb.Collection,
    top_k: int,
) -> float:
    """Compare ANN results against brute-force ground truth."""
    hits = 0
    total = 0

    for q_idx in query_indices:
        # Ground truth (exact)
        true_neighbors = set(get_ground_truth(q_idx, embeddings, top_k))

        # ANN result
        results = collection.query(
            query_embeddings=[embeddings[q_idx]],
            n_results=top_k,
        )
        ann_neighbors = set(int(id_) for id_ in results["ids"][0])

        hits += len(true_neighbors & ann_neighbors)
        total += top_k

    return hits / total

# Test on 50 random queries
np.random.seed(42)
query_indices = list(np.random.choice(len(documents), 50, replace=False))
all_embeddings_array = all_embeddings  # already computed above

print("Measuring recall at different top-K values...")
print(f"{'top-K':>6} {'Recall@K':>10}")
print("-" * 18)
for k in [1, 3, 5, 10, 20]:
    recall = measure_recall(query_indices, all_embeddings_array, collection, k)
    print(f"{k:>6} {recall:>10.4f}")
```

**Expected output:** ChromaDB HNSW with default settings typically achieves recall > 0.95 at top-10. If you see significantly lower recall, increase `hnsw:search_ef`.

### Step 4 — Effect of ef_search on recall

```python
# ChromaDB doesn't expose ef_search tuning directly at query time
# But you can compare collections with different construction parameters

col_low_ef = chroma_client.create_collection(
    "ag-news-low-ef",
    metadata={"hnsw:space": "cosine", "hnsw:construction_ef": 40, "hnsw:M": 8},
)
col_high_ef = chroma_client.create_collection(
    "ag-news-high-ef",
    metadata={"hnsw:space": "cosine", "hnsw:construction_ef": 400, "hnsw:M": 32},
)

for col in [col_low_ef, col_high_ef]:
    col.add(
        ids=[str(i) for i in range(len(documents))],
        embeddings=all_embeddings_array,
        documents=documents,
    )

recall_low = measure_recall(query_indices, all_embeddings_array, col_low_ef, 10)
recall_high = measure_recall(query_indices, all_embeddings_array, col_high_ef, 10)

print(f"\nRecall@10 with low ef (ef=40, M=8):   {recall_low:.4f}")
print(f"Recall@10 with high ef (ef=400, M=32): {recall_high:.4f}")
```

---

## Exercise 2: Hybrid Search — BM25 + Vector with RRF (Intermediate)

**Goal:** Implement hybrid search combining BM25 keyword retrieval with vector retrieval, fused using Reciprocal Rank Fusion. Measure when hybrid beats each component.
**Time:** ~35 min

### Setup

```bash
pip install rank-bm25
```

### Implement BM25 retrieval

```python
from rank_bm25 import BM25Okapi
import re

def tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return re.sub(r"[^\w\s]", "", text.lower()).split()

# Build BM25 index over all documents
tokenized_corpus = [tokenize(doc) for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

def bm25_search(query: str, top_k: int = 100) -> list[tuple[int, float]]:
    """Return (doc_idx, score) pairs sorted by BM25 score descending."""
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

# Test BM25
sample_q = "technology company acquisition merger"
bm25_results = bm25_search(sample_q, top_k=5)
print("BM25 top-5:")
for idx, score in bm25_results:
    print(f"  [{score:.3f}] {documents[idx][:100]}")
```

### Implement vector retrieval

```python
def vector_search(query: str, top_k: int = 100) -> list[tuple[int, float]]:
    """Return (doc_idx, distance) pairs sorted by cosine distance ascending."""
    emb = openai_client.embeddings.create(input=query, model="text-embedding-3-small").data[0].embedding
    results = collection.query(
        query_embeddings=[emb],
        n_results=top_k,
        include=["distances"],
    )
    return [
        (int(id_), dist)
        for id_, dist in zip(results["ids"][0], results["distances"][0])
    ]

# Test vector search
vector_results = vector_search(sample_q, top_k=5)
print("\nVector top-5:")
for idx, dist in vector_results:
    print(f"  [dist={dist:.3f}] {documents[idx][:100]}")
```

### Implement RRF fusion

```python
def rrf_fusion(
    bm25_ranked: list[tuple[int, float]],
    vector_ranked: list[tuple[int, float]],
    top_k: int = 10,
    k: int = 60,
) -> list[tuple[int, float]]:
    """Combine two ranked lists using Reciprocal Rank Fusion."""
    bm25_rank = {doc_idx: rank + 1 for rank, (doc_idx, _) in enumerate(bm25_ranked)}
    vector_rank = {doc_idx: rank + 1 for rank, (doc_idx, _) in enumerate(vector_ranked)}

    all_docs = set(bm25_rank.keys()) | set(vector_rank.keys())

    rrf_scores = {
        doc_idx: (1 / (k + bm25_rank.get(doc_idx, 1_000_000))) +
                 (1 / (k + vector_rank.get(doc_idx, 1_000_000)))
        for doc_idx in all_docs
    }

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

def hybrid_search(query: str, top_k: int = 5) -> list[int]:
    bm25_ranked = bm25_search(query, top_k=100)
    vector_ranked = vector_search(query, top_k=100)
    fused = rrf_fusion(bm25_ranked, vector_ranked, top_k=top_k)
    return [doc_idx for doc_idx, _ in fused]
```

### Compare all three on test queries

```python
test_queries = [
    # Keyword-heavy queries (BM25 should win)
    "Apple AAPL stock price earnings Q3",
    "FIFA World Cup 2006 Germany Brazil",
    # Semantic queries (vector should win)
    "latest advances in generative artificial intelligence",
    "problems with corporate restructuring and layoffs",
    # Mixed (hybrid should win)
    "Google antitrust lawsuit government regulation technology",
]

print(f"{'Query':<50} {'BM25 top1':>40} {'Vector top1':>40} {'Hybrid top1':>40}")
print("-" * 175)

for query in test_queries:
    bm25_top = bm25_search(query, top_k=5)
    vec_top = vector_search(query, top_k=5)
    hyb_top = hybrid_search(query, top_k=5)

    bm25_doc = documents[bm25_top[0][0]][:35] if bm25_top else "N/A"
    vec_doc = documents[vec_top[0][0]][:35] if vec_top else "N/A"
    hyb_doc = documents[hyb_top[0]][:35] if hyb_top else "N/A"

    print(f"{query[:48]:<50} {bm25_doc:<40} {vec_doc:<40} {hyb_doc:<40}")
```

**What to look for:** Keyword-heavy queries (exact stock tickers, event names) should show BM25 beating vector on top-1 relevance. For paraphrase/semantic queries, vector should win. Hybrid should never be worse than either alone.

---

## Exercise 3: HNSW vs Flat Index Latency Benchmark (Intermediate)

**Goal:** Measure query latency for exact (flat) search vs HNSW at different dataset sizes.
**Time:** ~25 min

```python
import time
import numpy as np

def generate_random_embeddings(n: int, dim: int = 1536) -> np.ndarray:
    """Generate unit-normalized random vectors (simulates real embeddings)."""
    vecs = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms

def flat_search(query: np.ndarray, index: np.ndarray, top_k: int) -> list[int]:
    """Exact nearest neighbor search via brute-force dot product."""
    scores = index @ query
    return list(np.argsort(scores)[-top_k:][::-1])

def benchmark_flat(n_docs: int, n_queries: int = 10, top_k: int = 5) -> float:
    """Return average query latency in ms for flat search."""
    index = generate_random_embeddings(n_docs)
    queries = generate_random_embeddings(n_queries)
    times = []
    for q in queries:
        start = time.perf_counter()
        flat_search(q, index, top_k)
        times.append(time.perf_counter() - start)
    return np.mean(times) * 1000

def benchmark_hnsw(n_docs: int, n_queries: int = 10, top_k: int = 5) -> float:
    """Return average query latency in ms for HNSW search via ChromaDB."""
    client = chromadb.Client()
    col = client.create_collection(f"bench-{n_docs}", metadata={"hnsw:space": "cosine"})
    vecs = generate_random_embeddings(n_docs)
    col.add(
        ids=[str(i) for i in range(n_docs)],
        embeddings=vecs.tolist(),
        documents=[f"doc {i}" for i in range(n_docs)],
    )
    queries = generate_random_embeddings(n_queries)
    times = []
    for q in queries:
        start = time.perf_counter()
        col.query(query_embeddings=[q.tolist()], n_results=top_k)
        times.append(time.perf_counter() - start)
    client.delete_collection(f"bench-{n_docs}")
    return np.mean(times) * 1000


print(f"{'N docs':>10} {'Flat (ms)':>12} {'HNSW (ms)':>12} {'Speedup':>10}")
print("-" * 48)

for n in [1_000, 10_000, 50_000, 100_000]:
    flat_ms = benchmark_flat(n)
    hnsw_ms = benchmark_hnsw(n)
    speedup = flat_ms / hnsw_ms
    print(f"{n:>10,} {flat_ms:>12.2f} {hnsw_ms:>12.2f} {speedup:>9.1f}×")
```

**Expected findings:**
- At 1K docs: flat and HNSW are similar (HNSW may even be slower due to graph overhead)
- At 10K docs: HNSW starts winning
- At 100K docs: HNSW is 10–50× faster than exact search

---

## Exercise 4: 8-bit Quantization — Size vs Accuracy (Advanced)

**Goal:** Apply scalar INT8 quantization to a batch of embeddings and measure the size reduction and recall impact.
**Time:** ~30 min

```python
import numpy as np
from openai import OpenAI

openai_client = OpenAI()

# Generate embeddings for 1000 sample texts
sample_texts = [documents[i] for i in range(1000)]

print("Generating embeddings for quantization experiment...")
all_embs = []
for i in range(0, len(sample_texts), 100):
    batch = sample_texts[i : i + 100]
    embs = openai_client.embeddings.create(input=batch, model="text-embedding-3-small")
    all_embs.extend([e.embedding for e in embs.data])

embeddings_fp32 = np.array(all_embs, dtype=np.float32)
print(f"Shape: {embeddings_fp32.shape}")
print(f"FP32 size: {embeddings_fp32.nbytes / 1024:.1f} KB")
```

### Implement and evaluate quantization

```python
class ScalarQuantizer:
    def __init__(self):
        self.min_val = None
        self.scale = None

    def fit(self, vectors: np.ndarray):
        self.min_val = vectors.min()
        self.max_val = vectors.max()
        self.scale = (self.max_val - self.min_val) / 255.0
        return self

    def quantize(self, vectors: np.ndarray) -> np.ndarray:
        return ((vectors - self.min_val) / self.scale).clip(0, 255).astype(np.uint8)

    def dequantize(self, quantized: np.ndarray) -> np.ndarray:
        return quantized.astype(np.float32) * self.scale + self.min_val


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between all pairs of rows in a and b."""
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_norm @ b_norm.T


def recall_at_k(
    original: np.ndarray,
    reconstructed: np.ndarray,
    n_queries: int = 50,
    k: int = 10,
) -> float:
    """Compare top-K neighbors from original vs reconstructed embeddings."""
    query_indices = np.random.choice(len(original), n_queries, replace=False)
    hits = 0

    for q_idx in query_indices:
        q_orig = original[q_idx : q_idx + 1]
        q_recon = reconstructed[q_idx : q_idx + 1]

        # True neighbors (exclude self)
        orig_sims = cosine_similarity_matrix(q_orig, original)[0]
        orig_sims[q_idx] = -1
        true_neighbors = set(np.argsort(orig_sims)[-k:])

        # Reconstructed neighbors
        recon_sims = cosine_similarity_matrix(q_recon, reconstructed)[0]
        recon_sims[q_idx] = -1
        recon_neighbors = set(np.argsort(recon_sims)[-k:])

        hits += len(true_neighbors & recon_neighbors)

    return hits / (n_queries * k)


# Fit quantizer and measure
np.random.seed(42)
sq = ScalarQuantizer().fit(embeddings_fp32)
embeddings_int8 = sq.quantize(embeddings_fp32)
embeddings_reconstructed = sq.dequantize(embeddings_int8)

# Compare FP16 (just cast, minimal quality loss)
embeddings_fp16 = embeddings_fp32.astype(np.float16)
embeddings_fp16_recon = embeddings_fp16.astype(np.float32)

print("\n=== Quantization Comparison ===")
print(f"{'Precision':<12} {'Size (KB)':>10} {'Compression':>12} {'Recall@10':>10}")
print("-" * 48)

fp32_kb = embeddings_fp32.nbytes / 1024
int8_kb = embeddings_int8.nbytes / 1024
fp16_kb = embeddings_fp16.nbytes / 1024

recall_fp16 = recall_at_k(embeddings_fp32, embeddings_fp16_recon)
recall_int8 = recall_at_k(embeddings_fp32, embeddings_reconstructed)
recall_fp32 = 1.0  # baseline

print(f"{'FP32 (base)':<12} {fp32_kb:>10.1f} {'1.0×':>12} {recall_fp32:>10.4f}")
print(f"{'FP16':<12} {fp16_kb:>10.1f} {'2.0×':>12} {recall_fp16:>10.4f}")
print(f"{'INT8':<12} {int8_kb:>10.1f} {'4.0×':>12} {recall_int8:>10.4f}")
```

### Visualize the reconstruction error

```python
# Compute element-wise reconstruction error
fp16_error = np.abs(embeddings_fp32 - embeddings_fp16_recon).mean()
int8_error = np.abs(embeddings_fp32 - embeddings_reconstructed).mean()

print(f"\nMean absolute reconstruction error:")
print(f"  FP16: {fp16_error:.6f}")
print(f"  INT8: {int8_error:.6f}")

# Check distribution of errors per dimension
dim_errors_int8 = np.abs(embeddings_fp32 - embeddings_reconstructed).mean(axis=0)
worst_dims = np.argsort(dim_errors_int8)[-5:]
print(f"\nHighest-error dimensions (INT8): {worst_dims}")
print(f"Their error values: {dim_errors_int8[worst_dims]}")
```

**Expected findings:**
- FP16: ~2× compression, < 0.001% recall loss — essentially free precision reduction
- INT8: ~4× compression, 1–3% recall loss — acceptable for most production RAG use cases

---

## Mini-Project: Hybrid Search Evaluation System

**Goal:** Build a system that queries a corpus with BM25, vector search, and hybrid, then scores each approach against human relevance judgments (or GPT-4 as judge).

```python
from openai import OpenAI

openai_client = OpenAI()

EVAL_QUERIES = [
    {
        "query": "Apple quarterly revenue earnings report 2024",
        "relevant_keywords": ["apple", "revenue", "earnings", "quarterly"],
        "type": "keyword-heavy",
    },
    {
        "query": "companies facing economic headwinds and workforce reductions",
        "relevant_keywords": ["layoff", "cut", "restructur", "downsiz"],
        "type": "semantic",
    },
    {
        "query": "Microsoft Azure cloud computing growth",
        "relevant_keywords": ["microsoft", "azure", "cloud"],
        "type": "mixed",
    },
]

def keyword_precision(result_docs: list[str], relevant_keywords: list[str]) -> float:
    """Simple precision proxy: fraction of results containing at least one keyword."""
    hits = sum(
        1 for doc in result_docs
        if any(kw.lower() in doc.lower() for kw in relevant_keywords)
    )
    return hits / len(result_docs) if result_docs else 0.0


print(f"\n{'Query':<45} {'Type':<14} {'BM25 P':>8} {'Vec P':>8} {'Hyb P':>8}")
print("-" * 88)

for item in EVAL_QUERIES:
    query = item["query"]
    keywords = item["relevant_keywords"]

    bm25_top = bm25_search(query, top_k=5)
    bm25_docs = [documents[idx] for idx, _ in bm25_top]
    bm25_p = keyword_precision(bm25_docs, keywords)

    vec_top = vector_search(query, top_k=5)
    vec_docs = [documents[idx] for idx, _ in vec_top]
    vec_p = keyword_precision(vec_docs, keywords)

    hyb_top = hybrid_search(query, top_k=5)
    hyb_docs = [documents[idx] for idx in hyb_top]
    hyb_p = keyword_precision(hyb_docs, keywords)

    print(f"{query[:43]:<45} {item['type']:<14} {bm25_p:>8.2f} {vec_p:>8.2f} {hyb_p:>8.2f}")
```

---

## Self-Assessment Checklist

- [ ] Exercise 1: Indexed 10K documents and computed Recall@K at multiple K values
- [ ] Exercise 1: Understand why recall degrades at smaller top-K
- [ ] Exercise 2: Implemented BM25 retrieval and understand what IDF rewards
- [ ] Exercise 2: Implemented RRF fusion and can explain the formula
- [ ] Exercise 2: Identified at least one query where BM25 beats vector and one where vector beats BM25
- [ ] Exercise 3: Measured HNSW vs flat latency and can explain the crossover point
- [ ] Exercise 4: Applied INT8 quantization and measured both size reduction and recall impact
- [ ] Mini-project: Hybrid evaluation system runs and produces results across all three retrieval methods
