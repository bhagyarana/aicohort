---
sidebar_position: 1
title: "Module 5: Vector Databases"
description: Understand the storage layer behind all semantic search — HNSW, IVF, hybrid search, quantization, and when to use Pinecone vs Qdrant vs pgvector.
---

# Module 5: Vector Databases & Search Systems

In Module 4, you used ChromaDB as a convenient black box. This module opens that box. Vector databases are not just "fast dictionaries for embeddings" — they implement sophisticated indexing algorithms (HNSW, IVF) that trade exactness for speed, support complex metadata filtering that interacts with vector search in non-obvious ways, and offer hybrid search that combines keyword and semantic retrieval for better recall.

Understanding what happens inside a vector database lets you tune it correctly, diagnose retrieval failures, and choose the right tool for your production requirements.

## What You'll Learn

- What vector databases actually do: store, index, and query high-dimensional vectors
- Why exact nearest-neighbor search is too slow at scale and what ANN gives up
- How HNSW (Hierarchical Navigable Small World) builds a layered graph for fast search
- How IVF (Inverted File Index) clusters vectors for memory-efficient search
- When to choose HNSW vs IVF and what the real tradeoffs are
- How metadata filtering interacts with vector search (and why it's tricky)
- Hybrid search: combining BM25 keyword search with vector search using RRF
- Quantization: trading embedding precision for storage and speed
- When to use Pinecone, Qdrant, Weaviate, pgvector, or ChromaDB

## Prerequisites

- [Module 4: RAG Systems](/learn/ai-engineering/module-4-rag-systems) — you need to understand embeddings and similarity search before diving into how they're indexed
- Python with `chromadb`, `rank-bm25`, and `qdrant-client` installable via pip

## Time Estimate

~3.5 hours

## Module Structure

| Page | What's covered |
|------|----------------|
| [Overview](./overview) | ANN search, HNSW and IVF internals, metadata filtering, hybrid search, quantization, and a production database comparison — with diagrams, tables, and code |
| [Hands-On](./hands-on) | Index 10K documents and measure recall, implement BM25 + vector hybrid search with RRF, benchmark HNSW vs flat search, and apply 8-bit quantization |
| [Resources](./resources) | Papers (HNSW, IVF, SPLADE), official database docs, and videos on vector search at scale |

---

**Ready to start? →** [Overview: Vector Databases & Search Systems](./overview)
