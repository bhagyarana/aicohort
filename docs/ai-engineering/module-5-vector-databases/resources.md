---
sidebar_position: 4
title: "Resources"
description: Papers, documentation, and videos on vector database internals, ANN algorithms, hybrid search, and quantization.
---

# Resources: Vector Databases & Search Systems

---

## Papers & Research

- [Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs (Malkov & Yashunin, 2018)](https://arxiv.org/abs/1603.09320) — the original HNSW paper; introduces the layered navigable small world graph construction and search algorithm; explains the parameters `M`, `ef_construction`, and `ef_search` used in every HNSW implementation (ChromaDB, Qdrant, pgvector)
- [Billion-scale similarity search with GPUs (Johnson et al., 2017)](https://arxiv.org/abs/1702.08734) — the FAISS paper from Facebook Research; introduces IVF-PQ (Inverted File Index + Product Quantization) for billion-scale ANN search; foundational reading for understanding how large-scale vector search differs from small-scale RAG
- [Product Quantization for Nearest Neighbor Search (Jégou et al., 2011)](https://inria.hal.science/inria-00514462/document) — the original PQ paper; introduces the codebook-based compression scheme described in the overview; mathematical treatment of compression vs reconstruction error tradeoffs
- [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking (Formal et al., 2021)](https://arxiv.org/abs/2107.05720) — a learned sparse retrieval model that outperforms BM25 for keyword-heavy search; useful for understanding the space between BM25 and dense vectors; relevant if BM25 underperforms in your hybrid search experiments
- [Approximate Nearest Neighbor ANN Benchmarks](https://ann-benchmarks.com/) — not a paper, but the canonical benchmark comparing HNSW, IVF, and other ANN algorithms across recall, queries per second, and index build time; use this to compare algorithms before choosing one for production

---

## Official Documentation

- [ChromaDB HNSW Configuration](https://docs.trychroma.com/usage-guide#changing-the-distance-function) — covers the `hnsw:space`, `hnsw:construction_ef`, `hnsw:search_ef`, and `hnsw:M` parameters used in Exercise 1
- [Qdrant Documentation — Filtering](https://qdrant.tech/documentation/concepts/filtering/) — the most detailed public documentation on integrating metadata filters with HNSW traversal; covers must/should/must_not predicates and payload indexing for performance
- [Qdrant Documentation — Quantization](https://qdrant.tech/documentation/guides/quantization/) — covers scalar, product, and binary quantization options; includes guidance on when each is appropriate and how to enable them without re-indexing
- [pgvector Documentation](https://github.com/pgvector/pgvector) — the Postgres extension README covers index creation (`ivfflat` and `hnsw`), distance operators, and performance tuning; relevant if you're deploying in an existing Postgres environment
- [Pinecone Documentation — Hybrid Search](https://docs.pinecone.io/docs/hybrid-search) — Pinecone's implementation of sparse-dense hybrid search using its sparse vectors feature; covers the `alpha` parameter for weighting keyword vs vector contributions
- [FAISS Documentation](https://faiss.ai/index.html) — the comprehensive reference for all FAISS index types (Flat, IVF, HNSW, PQ and their combinations); required reading if you are building a custom vector search layer without a managed database

---

## Videos & Courses

- [Pinecone: Vector Databases Explained (2023)](https://www.youtube.com/watch?v=dN0lsF2cvm4) — a clear visual walkthrough of how HNSW and IVF indexes work, with animations of the graph traversal; the best introductory video for understanding what happens under the hood
- [James Briggs: Hybrid Search Explained](https://www.youtube.com/watch?v=lkTpSTPNMoU) — covers BM25, dense retrieval, and RRF fusion with working code examples; demonstrates on a real dataset where hybrid consistently outperforms either component alone
- [Luc Perkins: Vector Databases — a Technical Deep Dive (2024)](https://www.youtube.com/watch?v=g4fMe0RL1YE) — covers the internals of multiple production vector databases; includes a live comparison of Qdrant, Weaviate, and pgvector on the same workload

---

## Tools to Know

| Tool | What it does | When to use |
|------|-------------|-------------|
| [FAISS](https://faiss.ai/) | Facebook's ANN library: IVF, HNSW, PQ, flat indexes | Building custom vector search without a managed DB; billion-scale research |
| [Qdrant](https://qdrant.tech/) | Production vector DB with HNSW, payload filtering, built-in quantization | Production RAG with complex filtering; self-host or cloud |
| [pgvector](https://github.com/pgvector/pgvector) | Vector extension for Postgres: HNSW + IVFFlat indexes | Already on Postgres; < 5M vectors; want SQL filtering natively |
| [rank-bm25](https://github.com/dorianbrown/rank_bm25) | Pure-Python BM25Okapi implementation | Adding BM25 to any hybrid search pipeline; no external service needed |
| [Weaviate](https://weaviate.io/) | Multi-modal vector DB with built-in BM25 hybrid search | Systems that need both text and image search; GraphQL interface |
| [Milvus](https://milvus.io/) | Open-source vector DB designed for billion-scale workloads | Very large production deployments (> 100M vectors) that need to self-host |
| [ANN Benchmarks](https://ann-benchmarks.com/) | Standardized recall/QPS benchmarks for all major ANN algorithms | Comparing algorithms before committing to a database choice |
| [VectorDBBench](https://github.com/zilliztech/VectorDBBench) | End-to-end benchmark for Qdrant, Pinecone, Milvus, Weaviate | Database selection for production — tests real ingestion + query latency |

---

## What to Read Next

→ **[Module 6: Model Optimization](/learn/ai-engineering/module-6-model-optimization)** — With the retrieval layer understood, learn how to optimize the generation layer: LLM quantization, batching strategies, streaming, speculative decoding, and when to use small models instead of large ones.
