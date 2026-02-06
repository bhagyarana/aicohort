---
sidebar_position: 1
title: "Module 2: Advanced LangChain"
description: Master embeddings, RAG, and complex chains
---

# Module 2: Advanced LangChain

Welcome to Module 2! This module covers advanced LangChain concepts including embeddings, vector stores, and Retrieval Augmented Generation (RAG).

## Learning Objectives

By the end of this module, you will be able to:

- Generate and use embeddings for semantic search
- Set up and query vector stores (ChromaDB, FAISS)
- Build complete RAG pipelines
- Create sequential and complex chains
- Process documents for knowledge bases

## Module Contents

| Section | Duration | Description |
|---------|----------|-------------|
| [Overview](./overview) | 30 min | Theory and architecture |
| [Hands-on](./hands-on) | 60 min | Practical exercises |
| [Resources](./resources) | - | Links and references |

## Topics Covered

### 1. Embeddings
Convert text to vector representations for semantic understanding.

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vector = embeddings.embed_query("What is RAG?")
```

### 2. Vector Stores
Store and retrieve documents based on semantic similarity.

```python
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings
)

results = vectorstore.similarity_search("query", k=3)
```

### 3. RAG Pipelines
Augment LLM responses with retrieved context.

```python
# Complete RAG chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)
```

### 4. Sequential Chains
Build multi-step processing pipelines.

```python
# Chain outputs feed into next chain
final_chain = first_chain | second_chain | third_chain
```

## Hands-on Labs

This module includes exercises on:

| Lab | Description |
|-----|-------------|
| `4_utilities/` | Embeddings and utilities |
| `6_chains/` | Sequential chain patterns |
| `7_rag/` | Retrieval Augmented Generation |
| `8_sample_project/` | Complete RAG project |

## Prerequisites

Before starting this module:

- [ ] Completed [Module 1](/learn/modules/module-1)
- [ ] Understanding of LCEL and chains
- [ ] API keys configured

## Key Takeaways

:::note What You'll Build
By the end of this module, you'll have built a complete RAG system that can answer questions based on your own documents.
:::

## Next Steps

1. Complete the [Overview](./overview)
2. Work through [Hands-on exercises](./hands-on)
3. Check [Resources](./resources)
4. Move to [Module 3: LangGraph](/learn/modules/module-3)

---

**Estimated Time**: 90 minutes

**Difficulty**: Intermediate
