---
sidebar_position: 2
title: "Overview"
description: The full RAG pipeline — from chunking and embedding to retrieval, reranking, and context injection — with failure modes, evaluation, and code.
---

# RAG Systems — Deep Dive

A language model's knowledge is frozen at training time. It cannot access your internal documents, your company's knowledge base, or anything published after its cutoff. Retrieval-Augmented Generation (RAG) fixes this by giving the model access to a dynamic, queryable store of external information at inference time. Instead of asking "what do you know?", you ask "what do you know, given these documents?".

---

## 1. Why RAG Exists

### The Core Problems RAG Solves

| Problem | Without RAG | With RAG |
|---------|------------|---------|
| **Stale knowledge** | Model answers based on training cutoff | Retrieved documents can be from today |
| **Private data** | Model has no access to internal docs | Index your docs, retrieve on demand |
| **Hallucination** | Model invents plausible-sounding facts | Model is grounded in retrieved text |
| **Attribution** | "The model said so" | "See retrieved chunk from doc X, paragraph Y" |
| **Context window limits** | Can't fit an entire knowledge base | Retrieve only the relevant 3–5 chunks |

RAG does not eliminate hallucination entirely — a model can still ignore retrieved context or synthesize incorrectly — but it provides a factual anchor that dramatically reduces fabrication rates.

### When RAG Is the Right Tool

```
Does the system need access to documents that change over time?  → RAG
Does the system need to answer from private/internal docs?       → RAG
Is the model hallucinating facts it should be able to look up?  → RAG
Does the task require citing sources?                           → RAG

Does the model need to learn a new task behavior or style?      → Fine-tuning (Module 7)
Is the knowledge truly static and always available?             → Few-shot prompting (Module 3)
```

---

## 2. Vector Representations

### How Text Becomes Numbers

An embedding model converts text into a dense numeric vector — a list of floating-point numbers (e.g., 1536 numbers for `text-embedding-3-small`). The key property: semantically similar texts produce vectors that are geometrically close.

```python
from openai import OpenAI

client = OpenAI()

def embed(text: str) -> list[float]:
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small",
    )
    return response.data[0].embedding

vec_a = embed("How do I reset my password?")
vec_b = embed("I forgot my login credentials")
vec_c = embed("What is the capital of France?")

# vec_a and vec_b will be geometrically close
# vec_c will be far from both
```

### Cosine Similarity vs Dot Product

Two distance measures are used in vector search. Choosing the wrong one gives subtly wrong results:

| Measure | Formula | When to use |
|---------|---------|------------|
| **Cosine similarity** | `(A · B) / (‖A‖ × ‖B‖)` | When vectors are **not** normalized — measures angle only, ignores magnitude |
| **Dot product** | `A · B` | When vectors **are** unit-normalized (length 1) — equivalent to cosine, but faster |
| **L2 distance** | `‖A - B‖` | Rarely used in RAG; better for clustering tasks |

OpenAI embedding models return unit-normalized vectors, so dot product and cosine similarity are equivalent. Most vector databases default to cosine similarity.

```python
import numpy as np

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# For unit-normalized vectors (OpenAI embeddings), this simplifies to:
def dot_product_similarity(a: list[float], b: list[float]) -> float:
    return float(np.dot(np.array(a), np.array(b)))
```

:::note Semantic ≠ Lexical
Cosine similarity captures *meaning*, not word overlap. "password reset" and "forgot credentials" score high similarity. "Paris" and "reset password" score near zero — even if they share no words, and even if one document literally contains "Paris" and the query is "Paris". Lexical overlap is handled by BM25 (covered in Module 5).
:::

---

## 3. The Full RAG Pipeline

```
                    INDEXING (done once, offline)
                    ─────────────────────────────
Documents → [Loader] → [Chunker] → [Embedder] → [Vector Store]


                    QUERYING (done per request, online)
                    ────────────────────────────────────
User Query → [Embed Query] → [Vector Search: top-K chunks]
           → [Reranker] → [Select top-N]
           → [Format context into prompt]
           → [LLM generates answer]
           → Response
```

Each stage has tunable parameters with real quality impact.

---

## 4. Chunking Strategies

### Why Chunking Matters

You cannot embed an entire 200-page document as a single vector — the embedding would average out all meaning into a useless blur. You split the document into chunks, embed each chunk independently, and retrieve individual chunks.

The chunk is the atomic unit of retrieval. If your chunk is too large, the embedding is diffuse and retrieval precision suffers. If it's too small, you lose context and the answer lacks coherence.

### Strategy Comparison

| Strategy | How it works | Best for | Risk |
|----------|-------------|----------|------|
| **Fixed-size (token count)** | Split every N tokens, regardless of content | Quick prototyping | Cuts mid-sentence, breaks tables |
| **Sentence-based** | Split on sentence boundaries (`.`, `!`, `?`) | Clean narrative prose | Uneven chunk sizes; very short sentences |
| **Recursive character** | Try `\n\n`, then `\n`, then `.`, then ` ` | General-purpose default | Requires tuning separators per doc type |
| **Semantic** | Embed sentences, cluster similar adjacent sentences together | High-quality academic/technical docs | Slow to index; needs embeddings at chunk time |
| **Document-structure-aware** | Respect headers, sections, tables (Markdown/HTML parsing) | Structured docs (wikis, manuals, code) | Requires per-format parsers |

### Recursive Character Chunking (the Default)

LangChain's `RecursiveCharacterTextSplitter` is the most widely-used chunker in production. It tries separators in order, preserving structure where possible:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,       # max tokens (or chars) per chunk
    chunk_overlap=64,     # overlap between adjacent chunks
    separators=["\n\n", "\n", ". ", " ", ""],
)

chunks = splitter.split_text(document_text)
```

### The Role of Overlap

Without overlap, a sentence split across two chunks may be retrievable from neither. With overlap, each chunk shares content with its neighbors:

```
Chunk 1: [A B C D E F]
Chunk 2:         [E F G H I J]
Chunk 3:                 [I J K L M N]
```

Overlap of 10–15% of chunk size is a good starting point. More overlap = better context continuity but larger index and more redundant retrieval.

:::tip Rule of Thumb
Start with `chunk_size=512, chunk_overlap=64` for general text. If your documents have clear heading structure (Markdown, HTML), use a document-structure-aware splitter instead.
:::

---

## 5. Indexing

### What Happens at Index Time

```python
import chromadb
from openai import OpenAI

client = OpenAI()
chroma = chromadb.Client()
collection = chroma.create_collection("knowledge-base")

def index_chunks(chunks: list[str], doc_id: str):
    embeddings = [
        client.embeddings.create(input=chunk, model="text-embedding-3-small")
        .data[0].embedding
        for chunk in chunks
    ]
    collection.add(
        ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
        embeddings=embeddings,
        documents=chunks,
        metadatas=[{"doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))],
    )
```

Key design decisions at index time:
- **Embedding model**: larger models (e.g., `text-embedding-3-large`) produce higher-quality embeddings at higher cost
- **Metadata**: store doc source, section, date — enables filtered retrieval later
- **Batch processing**: embed in batches of 100–1000 for efficiency

---

## 6. Retrieval

### Vector Search

At query time, embed the user's question and search for the top-K most similar chunks:

```python
def retrieve(query: str, top_k: int = 5) -> list[dict]:
    query_embedding = (
        client.embeddings.create(input=query, model="text-embedding-3-small")
        .data[0].embedding
    )
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances", "metadatas"],
    )
    return [
        {"text": doc, "distance": dist, "metadata": meta}
        for doc, dist, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        )
    ]
```

### Choosing top-K

| top-K | Effect |
|-------|--------|
| Too small (1–2) | May miss the best chunk if retrieval is imperfect |
| Good range (3–7) | Enough signal without overwhelming the context window |
| Too large (20+) | Context window fills up; model attention dilutes; slower |

Start with K=5. After adding a reranker (next section), increase retrieval K to 20 and re-rank to top 3–5.

---

## 7. Reranking

### Why Retrieval Alone Is Not Enough

Vector search finds chunks that are *semantically similar* to the query. But similarity ≠ relevance. A chunk about "password policy documents" is semantically similar to "how do I reset my password?" but may not directly answer the question.

A **cross-encoder reranker** takes each (query, chunk) pair and scores their relevance as a classification task — much more accurate than embedding-distance alone.

```
Vector retrieval:   embed(query) → find nearby chunks (fast, approximate)
Reranking:          score(query, chunk) → reorder by true relevance (slower, precise)
```

### Using a Cross-Encoder Reranker

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, candidates: list[dict], top_n: int = 3) -> list[dict]:
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [candidate for _, candidate in ranked[:top_n]]

# Full pipeline
candidates = retrieve(query, top_k=20)   # broad retrieval
top_chunks = rerank(query, candidates, top_n=3)  # precise selection
```

:::tip Two-Stage Retrieval
This pattern — wide retrieval + narrow reranking — is standard in production RAG. Retrieve 20, rerank to 3–5. The cost of reranking 20 pairs is tiny compared to the quality improvement.
:::

---

## 8. Context Injection

### Formatting Matters

Where you put the retrieved context in the prompt affects how much the model uses it:

```python
def build_rag_prompt(query: str, chunks: list[dict]) -> list[dict]:
    context = "\n\n---\n\n".join(
        f"[Source: {c['metadata']['doc_id']}]\n{c['text']}"
        for c in chunks
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that answers questions using only the provided context. "
                "If the answer is not in the context, say 'I don't have information about that.' "
                "Always cite the source document when you reference specific facts."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        },
    ]
```

**Key rules for context injection:**
1. **Context before question** — models attend better to context placed early in the prompt
2. **Delimiter chunks** — separate chunks with a visible delimiter (`---`) so the model can distinguish sources
3. **Include source metadata** — enables citation in the answer
4. **Explicit grounding instruction** — tell the model to only use the provided context; do not rely on its parametric knowledge

### The "Lost in the Middle" Problem

Research shows LLMs are better at using context at the **beginning** and **end** of the context window than in the middle. If you have 10 retrieved chunks, the most important one should be first or last — not buried in position 5.

---

## 9. RAG Failure Modes

Every RAG system fails in predictable ways. Knowing the failure mode tells you where to fix it:

| Failure | Symptom | Root Cause | Fix |
|---------|---------|-----------|-----|
| **Wrong chunks retrieved** | Answer is off-topic | Embedding mismatch or bad chunking | Try different chunking; add reranker |
| **Right chunks, wrong answer** | Chunks are relevant but answer ignores them | Context buried in prompt; model overrides context | Put context before question; add explicit grounding instruction |
| **Hallucination despite retrieval** | Model adds facts not in retrieved context | Insufficient instruction | Use stricter system prompt: "only use the provided context" |
| **No relevant chunks exist** | Question is unanswerable | Coverage gap in the index | Add a confidence threshold; graceful "I don't know" response |
| **Slow responses** | Latency > 5 seconds | Reranker on hot path; top-K too large | Cache frequent queries; reduce top-K; async retrieval |
| **Outdated answers** | Correct answer exists but old version is returned | Stale index | Re-index on document update; add last-modified metadata filter |

---

## 10. Evaluating RAG

### Why "It Looks Good" Is Not Evaluation

RAG systems fail silently. A confident-sounding wrong answer looks identical to a correct one. You need automated metrics:

**RAGAS Framework** (the standard):

| Metric | What it measures | How it's computed |
|--------|----------------|------------------|
| **Context Relevance** | Are retrieved chunks relevant to the question? | LLM judges chunk vs query |
| **Answer Faithfulness** | Does the answer stick to retrieved context (no hallucination)? | LLM checks if each claim is supported by context |
| **Answer Relevance** | Does the answer actually address the question? | Embed answer + question, measure similarity |

```python
# pip install ragas
from ragas import evaluate
from ragas.metrics import context_relevance, answer_faithfulness, answer_relevance
from datasets import Dataset

data = {
    "question": ["How do I reset my password?"],
    "answer": ["Go to Settings > Security > Reset Password and follow the prompts."],
    "contexts": [["To reset your password, navigate to Settings, then Security, then Reset Password."]],
    "ground_truth": ["Navigate to Settings > Security > Reset Password."],
}

result = evaluate(Dataset.from_dict(data), metrics=[
    context_relevance, answer_faithfulness, answer_relevance
])
print(result)
```

:::note Thresholds to Target
- Context Relevance > 0.8
- Answer Faithfulness > 0.85
- Answer Relevance > 0.80

Below these thresholds, your RAG pipeline needs tuning before shipping.
:::

---

## Common Mistakes

1. **Chunking too large** — A 2,000-token chunk blurs the embedding signal. Stay under 512 tokens for most use cases.
2. **No overlap** — Adjacent chunks miss context at the boundary. Use 10–15% overlap.
3. **Skipping reranking** — Vector retrieval alone leaves quality on the table. A cross-encoder reranker almost always improves precision.
4. **Context after question** — Putting context after the user's question reduces how much the model uses it.
5. **No grounding instruction** — Without explicitly telling the model to stick to the context, it will mix in parametric knowledge and hallucinate.
6. **Not evaluating** — Building RAG without RAGAS or a similar framework means you are flying blind on quality.
7. **Re-indexing on every request** — Index once offline; query online. Re-index only when source documents change.

---

## Quiz

<details>
<summary>Q1: Why is overlap between chunks important?</summary>

Overlap ensures that content near chunk boundaries is fully represented in at least one chunk. Without overlap, a sentence split across two chunk boundaries may be partially missing from both — making it harder to retrieve the relevant information.

</details>

<details>
<summary>Q2: You retrieve the top-5 chunks for a query. The answer is in chunk 4. What can you do to improve the chance the model uses it?</summary>

Add a reranker: retrieve top-20, then rerank by true relevance. The correct chunk may rank higher under a cross-encoder than under cosine similarity. Also ensure your system prompt explicitly instructs the model to use all provided context, and consider ordering the most relevant chunk first or last (avoiding the "lost in the middle" problem).

</details>

<details>
<summary>Q3: What does Answer Faithfulness measure, and why does it matter more than Answer Relevance for high-stakes applications?</summary>

Answer Faithfulness measures whether every claim in the model's answer is supported by the retrieved context (no hallucination). Answer Relevance measures whether the answer addresses the question. In high-stakes applications (legal, medical, financial), a relevant-but-hallucinated answer is dangerous — it sounds correct but asserts facts not in the source documents. Faithfulness is the more critical constraint.

</details>

---

:::info See Also
Want to build this pipeline in code? See **[Module 2: Advanced LangChain](/learn/modules/module-2)** in the internal training track — hands-on RAG implementation with ChromaDB, FAISS, and LangChain's retriever API.
:::

## Next Steps

→ **[Hands-On: RAG Systems](./hands-on)** — Build the full pipeline from scratch, compare chunking strategies, add reranking, and run RAGAS evaluation.

→ **[Module 5: Vector Databases](/learn/ai-engineering/module-5-vector-databases)** — Go deeper on the storage layer: HNSW indexing, hybrid search, filtering, quantization, and when to use Pinecone vs Qdrant vs pgvector.
