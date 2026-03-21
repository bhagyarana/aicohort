---
sidebar_position: 4
title: "Resources"
description: Foundational papers, tools, documentation, and videos for Retrieval-Augmented Generation.
---

# Resources: RAG Systems

---

## Papers & Research

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401) — the original RAG paper from Facebook AI Research; introduces the RAG-Token and RAG-Sequence model variants; uses DPR for retrieval over a Wikipedia index; the benchmarks here (Natural Questions, TriviaQA) are still the standard comparison points
- [Dense Passage Retrieval for Open-Domain Question Answering (Karpukhin et al., 2020)](https://arxiv.org/abs/2004.04906) — the DPR paper that provides the retriever backbone for the original RAG system; explains bi-encoder training with in-batch negatives; foundational reading for understanding embedding-based retrieval
- [RAGAS: Automated Evaluation of Retrieval Augmented Generation (Es et al., 2023)](https://arxiv.org/abs/2309.15217) — introduces the Context Relevance, Answer Faithfulness, and Answer Relevance metrics used in Exercise 4; explains why reference-free evaluation is necessary for RAG; the RAGAS library used in hands-on is based directly on this paper
- [Lost in the Middle: How Language Models Use Long Contexts (Liu et al., 2023)](https://arxiv.org/abs/2307.03172) — shows empirically that LLMs attend more to context at the beginning and end of the prompt, less in the middle; directly informs the context ordering advice in the overview
- [Self-RAG: Learning to Retrieve, Generate, and Critique Through Self-Reflection (Asai et al., 2023)](https://arxiv.org/abs/2310.11511) — introduces an adaptive retrieval approach where the model decides *when* to retrieve and critiques its own outputs; more advanced than standard RAG but shows the direction the field is heading
- [HyDE: Hypothetical Document Embeddings for Zero-Shot Dense Retrieval (Gao et al., 2022)](https://arxiv.org/abs/2212.10496) — use an LLM to generate a hypothetical answer, embed it, and use that embedding for retrieval instead of the raw question; often improves recall when the query and document vocabulary differ significantly

---

## Official Documentation

- [LangChain RAG Documentation](https://python.langchain.com/docs/use_cases/question_answering/) — the canonical LangChain guide for building RAG pipelines; covers document loaders, text splitters, vector stores, and retrieval chains with LCEL
- [ChromaDB Documentation](https://docs.trychroma.com/) — the embedded vector database used in hands-on exercises; covers collections, metadata filtering, and persistence; good for local development and prototyping
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings) — covers the `text-embedding-3-small` and `text-embedding-3-large` models, dimensionality reduction, and best practices for chunking and indexing
- [Sentence Transformers — Cross-Encoders](https://www.sbert.net/docs/cross_encoder/usage/usage.html) — documentation for the `CrossEncoder` class used in Exercise 3; covers available models, scoring, and fine-tuning
- [RAGAS Documentation](https://docs.ragas.io/) — the library used for evaluation in Exercise 4; covers all metrics, dataset format, and integration with LangChain and LlamaIndex

---

## Videos & Courses

- [Jerry Liu: Building Production-Ready RAG (LlamaIndex, 2023)](https://www.youtube.com/watch?v=TRjq7t2Ms5I) — the LlamaIndex co-founder covers advanced RAG patterns: query transformations, hierarchical indexing, sub-question decomposition, and evaluation; most relevant after completing the basics in this module
- [Greg Kamradt: Chunking Strategies for LLM Applications](https://www.youtube.com/watch?v=8OJC21T2SL4) — the most thorough video treatment of chunking strategies available; includes empirical comparisons across fixed-size, recursive, semantic, and document-structure chunking on real datasets; directly relevant to Exercise 2
- [Pinecone: RAG in Production](https://www.youtube.com/watch?v=dXxQ0LR-3Hg) — covers production considerations: indexing pipelines, metadata filtering, hybrid search, and monitoring retrieval quality; bridges the gap between the exercises here and real-world deployment

---

## Tools to Know

| Tool | What it does | When to use |
|------|-------------|-------------|
| [LangChain](https://python.langchain.com/) | Document loaders, text splitters, retrieval chains, RAG LCEL pipelines | Building RAG pipelines quickly with pre-built components |
| [LlamaIndex](https://www.llamaindex.ai/) | Alternative to LangChain; stronger document parsing, hierarchical indexing, sub-question engines | Complex document structures (PDFs with tables, code, mixed formats) |
| [ChromaDB](https://www.trychroma.com/) | Embedded vector database, runs in-process, no server needed | Local development, prototyping, small datasets (< 1M vectors) |
| [Qdrant](https://qdrant.tech/) | Production vector database with excellent filtering and payload support | Production RAG with metadata filtering requirements |
| [RAGAS](https://docs.ragas.io/) | Automated RAG evaluation: Context Relevance, Faithfulness, Answer Relevance | Evaluating and monitoring RAG pipeline quality |
| [sentence-transformers](https://www.sbert.net/) | Cross-encoder rerankers and bi-encoder embeddings (local, free) | Adding a reranker without API costs; local embedding generation |
| [Cohere Rerank](https://cohere.com/rerank) | Hosted cross-encoder reranker API | Production reranking without serving your own model |
| [Unstructured](https://unstructured.io/) | Parse PDFs, Word docs, HTML, images into structured text for indexing | Complex document parsing (scanned PDFs, invoices, presentations) |

---

## What to Read Next

→ **[Module 5: Vector Databases](/learn/ai-engineering/module-5-vector-databases)** — Understand the storage layer under your RAG system: how HNSW and IVF indexes work, how to combine keyword and vector search (hybrid search), quantization for scale, and how to choose between Pinecone, Qdrant, Weaviate, and pgvector for production.
