---
sidebar_position: 3
title: "Hands-On"
description: Build a RAG pipeline from scratch, compare chunking strategies, add cross-encoder reranking, and evaluate with RAGAS.
---

# Hands-On: RAG Systems

Four exercises that build on each other. By the end you will have a working, evaluated RAG pipeline. Exercises 1 and 2 can be completed locally with ChromaDB (no paid API needed for the vector store). Exercises 3 and 4 require an OpenAI API key for embeddings and completions.

---

## Exercise 1: RAG Pipeline from Scratch (Beginner)

**Goal:** Build a complete RAG pipeline: load a document → chunk → embed → store → query → generate an answer.
**Time:** ~40 min

### Setup

```bash
pip install openai chromadb langchain langchain-community pypdf tiktoken
```

### Step 1 — Load and chunk a PDF

```python
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Use any PDF — e.g., download a short public document
# For testing: https://arxiv.org/pdf/2005.11401  (the original RAG paper)
PDF_PATH = "rag_paper.pdf"

loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_documents(pages)

print(f"Document pages: {len(pages)}")
print(f"Chunks created: {len(chunks)}")
print(f"\nSample chunk:\n{chunks[10].page_content[:300]}")
```

### Step 2 — Embed and store in ChromaDB

```python
import chromadb
from openai import OpenAI

openai_client = OpenAI()   # reads OPENAI_API_KEY from environment
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(
    "rag-demo",
    metadata={"hnsw:space": "cosine"},
)

def embed_batch(texts: list[str]) -> list[list[float]]:
    response = openai_client.embeddings.create(
        input=texts,
        model="text-embedding-3-small",
    )
    return [item.embedding for item in response.data]

# Embed in batches of 100 (API limit is 2048)
BATCH_SIZE = 100
all_texts = [c.page_content for c in chunks]
all_embeddings = []

for i in range(0, len(all_texts), BATCH_SIZE):
    batch = all_texts[i : i + BATCH_SIZE]
    all_embeddings.extend(embed_batch(batch))
    print(f"Embedded {min(i + BATCH_SIZE, len(all_texts))}/{len(all_texts)} chunks")

collection.add(
    ids=[str(i) for i in range(len(chunks))],
    embeddings=all_embeddings,
    documents=all_texts,
    metadatas=[{"page": c.metadata.get("page", 0)} for c in chunks],
)
print(f"\nIndexed {collection.count()} chunks into ChromaDB")
```

### Step 3 — Retrieve and generate

```python
def ask(question: str, top_k: int = 5) -> str:
    # Embed the query
    query_embedding = (
        openai_client.embeddings.create(
            input=question,
            model="text-embedding-3-small",
        )
        .data[0]
        .embedding
    )

    # Retrieve top-K chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances", "metadatas"],
    )

    retrieved_chunks = results["documents"][0]
    distances = results["distances"][0]

    # Format context
    context = "\n\n---\n\n".join(
        f"[Page {meta['page']+1}]\n{doc}"
        for doc, meta in zip(retrieved_chunks, results["metadatas"][0])
    )

    # Generate answer
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer questions using ONLY the provided context. "
                    "If the answer is not in the context, say 'I don't have information about that.' "
                    "Cite the page number when referencing specific facts."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
        temperature=0,
    )

    answer = response.choices[0].message.content
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")
    print(f"\nTop chunk distances: {[round(d, 3) for d in distances]}")
    return answer


# Test with questions about the document
ask("What is retrieval-augmented generation?")
ask("What datasets were used to evaluate the RAG model?")
ask("What are the limitations of the approach?")
```

### Step 4 — Inspect retrieval quality

```python
def inspect_retrieval(question: str, top_k: int = 5):
    """Show which chunks were retrieved and their similarity scores."""
    query_embedding = (
        openai_client.embeddings.create(input=question, model="text-embedding-3-small")
        .data[0].embedding
    )
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances", "metadatas"],
    )
    print(f"\nQuery: {question}\n{'='*60}")
    for i, (doc, dist, meta) in enumerate(zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0],
    )):
        print(f"\n[Rank {i+1}] Distance: {dist:.4f} | Page {meta['page']+1}")
        print(doc[:200] + "..." if len(doc) > 200 else doc)

inspect_retrieval("limitations of retrieval augmented generation")
```

**Expected outcome:** You should see the most relevant chunks at the top (lowest cosine distance). If unrelated chunks appear at rank 1–2, your chunking strategy may need adjustment.

---

## Exercise 2: Chunking Strategy Comparison (Intermediate)

**Goal:** Measure how different chunking strategies affect retrieval quality on the same document and query set.
**Time:** ~35 min

### Setup

Use the same PDF and ChromaDB client from Exercise 1. We'll create separate collections for each strategy.

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    NLTKTextSplitter,
)

# Load raw text once
full_text = "\n\n".join(p.page_content for p in pages)

# Strategy 1: Fixed-size character split (simplest, most naive)
fixed_splitter = CharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=0,    # no overlap — worst case
    separator="",
)
fixed_chunks = fixed_splitter.split_text(full_text)

# Strategy 2: Recursive character split (general-purpose default)
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ". ", " ", ""],
)
recursive_chunks = recursive_splitter.split_text(full_text)

# Strategy 3: Large chunks (fewer, more context per chunk)
large_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=128,
)
large_chunks = large_splitter.split_text(full_text)

print(f"Fixed-size:  {len(fixed_chunks)} chunks")
print(f"Recursive:   {len(recursive_chunks)} chunks")
print(f"Large:       {len(large_chunks)} chunks")
```

### Build an index per strategy and compare

```python
def build_index(name: str, chunks: list[str]) -> chromadb.Collection:
    col = chroma_client.get_or_create_collection(
        name, metadata={"hnsw:space": "cosine"}
    )
    if col.count() == 0:
        embeddings = []
        for i in range(0, len(chunks), BATCH_SIZE):
            embeddings.extend(embed_batch(chunks[i : i + BATCH_SIZE]))
        col.add(
            ids=[str(i) for i in range(len(chunks))],
            embeddings=embeddings,
            documents=chunks,
        )
    return col

idx_fixed = build_index("fixed", fixed_chunks)
idx_recursive = build_index("recursive", recursive_chunks)
idx_large = build_index("large", large_chunks)

# Test queries with known expected answers
test_queries = [
    "What is the architecture of the RAG model?",
    "How does the retrieval component work in RAG?",
    "What is the role of the generator in RAG?",
]

def get_top1(collection: chromadb.Collection, query: str) -> str:
    emb = openai_client.embeddings.create(input=query, model="text-embedding-3-small").data[0].embedding
    res = collection.query(query_embeddings=[emb], n_results=1)
    return res["documents"][0][0][:200]

print("\n=== Chunking Strategy Comparison ===")
for query in test_queries:
    print(f"\nQuery: {query}")
    print(f"  Fixed:     {get_top1(idx_fixed, query)[:100]}...")
    print(f"  Recursive: {get_top1(idx_recursive, query)[:100]}...")
    print(f"  Large:     {get_top1(idx_large, query)[:100]}...")
```

**What to observe:**
- Fixed-size chunks often cut mid-sentence — the top result may be incomplete
- Recursive chunks respect paragraph boundaries — results are usually more coherent
- Large chunks have more context but the embedding signal is more diffuse

**Self-assessment:** Which strategy returned the most complete, relevant top-1 chunk for each query? Write down your findings — you'll use this in Exercise 4.

---

## Exercise 3: Add a Cross-Encoder Reranker (Intermediate)

**Goal:** Implement two-stage retrieval — wide vector search followed by precise cross-encoder reranking — and measure the precision improvement.
**Time:** ~30 min

### Setup

```bash
pip install sentence-transformers
```

### Implement the reranker

```python
from sentence_transformers import CrossEncoder
import time

# Load a small, fast cross-encoder (runs locally, no API needed)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def retrieve_and_rerank(
    query: str,
    collection: chromadb.Collection,
    retrieve_k: int = 20,
    rerank_top_n: int = 3,
) -> list[dict]:
    # Stage 1: broad vector retrieval
    query_emb = (
        openai_client.embeddings.create(input=query, model="text-embedding-3-small")
        .data[0].embedding
    )
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=retrieve_k,
        include=["documents", "distances"],
    )
    candidates = [
        {"text": doc, "vector_distance": dist}
        for doc, dist in zip(results["documents"][0], results["distances"][0])
    ]

    # Stage 2: cross-encoder reranking
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True,
    )
    return [
        {"text": c["text"], "rerank_score": float(s), "vector_distance": c["vector_distance"]}
        for s, c in ranked[:rerank_top_n]
    ]


# Compare: vector-only vs reranked
query = "What evaluation metrics were used to assess RAG performance?"

print("=== Vector-Only (top 3) ===")
query_emb = openai_client.embeddings.create(input=query, model="text-embedding-3-small").data[0].embedding
raw = idx_recursive.query(query_embeddings=[query_emb], n_results=3)
for i, (doc, dist) in enumerate(zip(raw["documents"][0], raw["distances"][0])):
    print(f"\n[Rank {i+1}] Distance: {dist:.4f}")
    print(doc[:200])

print("\n\n=== After Reranking (retrieve 20, rerank to 3) ===")
reranked = retrieve_and_rerank(query, idx_recursive)
for i, r in enumerate(reranked):
    print(f"\n[Rank {i+1}] Rerank score: {r['rerank_score']:.4f} | Vector dist: {r['vector_distance']:.4f}")
    print(r["text"][:200])
```

### Measure latency cost of reranking

```python
def time_retrieval(query: str, collection: chromadb.Collection, use_reranker: bool, k: int = 5):
    start = time.perf_counter()
    if use_reranker:
        results = retrieve_and_rerank(query, collection, retrieve_k=20, rerank_top_n=k)
    else:
        emb = openai_client.embeddings.create(input=query, model="text-embedding-3-small").data[0].embedding
        raw = collection.query(query_embeddings=[emb], n_results=k)
        results = raw["documents"][0]
    elapsed = time.perf_counter() - start
    return elapsed

queries = [
    "What is retrieval augmented generation?",
    "How does the model handle open-domain questions?",
    "What are the training details of the retrieval model?",
]

print(f"{'Query':<50} {'Vector-only':>12} {'With reranker':>14}")
print("-" * 78)
for q in queries:
    t_vec = time_retrieval(q, idx_recursive, use_reranker=False)
    t_rer = time_retrieval(q, idx_recursive, use_reranker=True)
    print(f"{q[:48]:<50} {t_vec*1000:>10.0f}ms {t_rer*1000:>12.0f}ms")
```

**Expected finding:** Reranking adds ~100–500ms locally. In production, this is usually acceptable given the quality improvement.

---

## Exercise 4: RAG Evaluator (Advanced)

**Goal:** Implement a simple RAG evaluation harness that measures Context Relevance and Answer Faithfulness using RAGAS (or an LLM-as-judge fallback).
**Time:** ~45 min

### Setup

```bash
pip install ragas datasets
```

### Build a small evaluation dataset

```python
# You need: question, expected answer (ground truth), and your RAG pipeline

EVAL_QUESTIONS = [
    {
        "question": "What is the main contribution of the RAG paper?",
        "ground_truth": "RAG combines parametric memory (LLM weights) with non-parametric memory (retrieved documents) for knowledge-intensive NLP tasks.",
    },
    {
        "question": "What retrieval model is used in RAG?",
        "ground_truth": "RAG uses DPR (Dense Passage Retrieval) with a bi-encoder architecture.",
    },
    {
        "question": "On which benchmark did RAG outperform the previous state-of-the-art?",
        "ground_truth": "RAG outperformed previous state-of-the-art on Open-domain QA benchmarks including Natural Questions, WebQuestions, and CuratedTrec.",
    },
]
```

### Run your pipeline on the eval set

```python
def rag_pipeline(question: str, collection: chromadb.Collection) -> tuple[str, list[str]]:
    """Returns (answer, retrieved_contexts)."""
    reranked = retrieve_and_rerank(question, collection, retrieve_k=20, rerank_top_n=3)
    contexts = [r["text"] for r in reranked]

    context_str = "\n\n---\n\n".join(contexts)
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using ONLY the provided context. "
                    "If the answer is not present, say 'Not in context.' "
                    "Be concise."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context_str}\n\nQuestion: {question}",
            },
        ],
        temperature=0,
    )
    return response.choices[0].message.content, contexts


# Collect results
eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
for item in EVAL_QUESTIONS:
    answer, contexts = rag_pipeline(item["question"], idx_recursive)
    eval_data["question"].append(item["question"])
    eval_data["answer"].append(answer)
    eval_data["contexts"].append(contexts)
    eval_data["ground_truth"].append(item["ground_truth"])
    print(f"Q: {item['question']}\nA: {answer}\n")
```

### Run RAGAS evaluation

```python
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_relevance,
    answer_faithfulness,
    answer_relevance,
)

dataset = Dataset.from_dict(eval_data)
result = evaluate(
    dataset,
    metrics=[context_relevance, answer_faithfulness, answer_relevance],
)

print("\n=== RAGAS Evaluation Results ===")
print(result)

# Print per-question breakdown
result_df = result.to_pandas()
for _, row in result_df.iterrows():
    print(f"\nQ: {row['question'][:60]}")
    print(f"  Context Relevance: {row.get('context_relevance', 'N/A'):.3f}")
    print(f"  Faithfulness:      {row.get('answer_faithfulness', 'N/A'):.3f}")
    print(f"  Answer Relevance:  {row.get('answer_relevance', 'N/A'):.3f}")
```

### LLM-as-judge fallback (if RAGAS is unavailable)

```python
def judge_faithfulness(question: str, answer: str, contexts: list[str]) -> float:
    """Score 0.0–1.0: does the answer contain only facts from the context?"""
    context_str = "\n\n".join(contexts)
    prompt = f"""You are an evaluator. Score whether the ANSWER is faithful to the CONTEXT.
Faithful means: every factual claim in the answer is supported by the context.
Return only a number between 0.0 and 1.0.

CONTEXT:
{context_str}

ANSWER:
{answer}

Faithfulness score (0.0 = completely hallucinated, 1.0 = fully grounded):"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=5,
    )
    try:
        return float(response.choices[0].message.content.strip())
    except ValueError:
        return 0.0

# Test
for item in eval_data["question"][:2]:
    idx = eval_data["question"].index(item)
    score = judge_faithfulness(
        eval_data["question"][idx],
        eval_data["answer"][idx],
        eval_data["contexts"][idx],
    )
    print(f"Faithfulness: {score:.2f} | Q: {item[:60]}")
```

---

## Mini-Project: Document Q&A System

**Goal:** Combine all four exercises into a reusable `DocumentQA` class that indexes any PDF and answers questions with cited sources and quality scores.

```python
class DocumentQA:
    def __init__(self, model: str = "gpt-4o-mini", embed_model: str = "text-embedding-3-small"):
        self.openai = OpenAI()
        self.chroma = chromadb.Client()
        self.model = model
        self.embed_model = embed_model
        self.collection = None
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def index(self, pdf_path: str, collection_name: str = "qa-index"):
        """Load, chunk, embed, and store a PDF."""
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
        chunks = splitter.split_documents(pages)

        self.collection = self.chroma.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"}
        )
        texts = [c.page_content for c in chunks]
        metas = [{"page": c.metadata.get("page", 0)} for c in chunks]

        for i in range(0, len(texts), 100):
            batch = texts[i : i + 100]
            embs = [
                e.embedding
                for e in self.openai.embeddings.create(input=batch, model=self.embed_model).data
            ]
            self.collection.add(
                ids=[str(i + j) for j in range(len(batch))],
                embeddings=embs,
                documents=batch,
                metadatas=metas[i : i + 100],
            )
        print(f"Indexed {self.collection.count()} chunks from {pdf_path}")

    def ask(self, question: str) -> dict:
        """Return answer + sources + faithfulness score."""
        # Retrieve and rerank
        emb = self.openai.embeddings.create(input=question, model=self.embed_model).data[0].embedding
        raw = self.collection.query(query_embeddings=[emb], n_results=20, include=["documents", "metadatas"])
        pairs = [(question, doc) for doc in raw["documents"][0]]
        scores = self.reranker.predict(pairs)
        top3 = sorted(zip(scores, raw["documents"][0], raw["metadatas"][0]), reverse=True)[:3]

        contexts = [doc for _, doc, _ in top3]
        sources = [f"Page {meta['page']+1}" for _, _, meta in top3]

        context_str = "\n\n---\n\n".join(
            f"[{src}]\n{ctx}" for src, ctx in zip(sources, contexts)
        )
        response = self.openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Answer using ONLY the provided context. Cite sources."},
                {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {question}"},
            ],
            temperature=0,
        )
        answer = response.choices[0].message.content
        faithfulness = judge_faithfulness(question, answer, contexts)

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "faithfulness": faithfulness,
        }


# Usage
qa = DocumentQA()
qa.index("rag_paper.pdf")
result = qa.ask("What problem does RAG solve?")
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
print(f"Faithfulness: {result['faithfulness']:.2f}")
```

---

## Self-Assessment Checklist

- [ ] Exercise 1: Built and tested a working end-to-end RAG pipeline
- [ ] Exercise 1: Inspected retrieval results and understood distance scores
- [ ] Exercise 2: Compared at least two chunking strategies on the same queries
- [ ] Exercise 2: Can explain why recursive splitting outperforms fixed-size splitting
- [ ] Exercise 3: Implemented two-stage retrieval with a cross-encoder reranker
- [ ] Exercise 3: Measured latency overhead of reranking
- [ ] Exercise 4: Ran RAGAS evaluation and can interpret all three metrics
- [ ] Mini-project: `DocumentQA` class works end-to-end on a new PDF
