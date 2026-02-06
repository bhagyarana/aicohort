---
sidebar_position: 2
title: "Overview"
description: Module 2 theory and concepts
---

# Module 2 Overview

This overview covers the theoretical foundations of embeddings, vector stores, and RAG systems.

## What is RAG?

**Retrieval Augmented Generation (RAG)** enhances LLM responses by retrieving relevant context from external knowledge sources.

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG Architecture                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────┐     ┌──────────────┐     ┌──────────────────┐   │
│   │  Query   │────▶│  Retriever   │────▶│    Retrieved     │   │
│   │          │     │              │     │    Documents     │   │
│   └──────────┘     └──────────────┘     └────────┬─────────┘   │
│                                                   │              │
│                                                   ▼              │
│   ┌──────────┐     ┌──────────────┐     ┌──────────────────┐   │
│   │ Response │◀────│     LLM      │◀────│  Query + Context │   │
│   │          │     │              │     │                  │   │
│   └──────────┘     └──────────────┘     └──────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Core Concepts

### 1. Embeddings

Embeddings convert text into dense vector representations:

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Single text
vector = embeddings.embed_query("Hello, world!")
print(f"Vector dimension: {len(vector)}")  # e.g., 1536

# Multiple texts
vectors = embeddings.embed_documents([
    "First document",
    "Second document"
])
```

**Embedding Providers:**

| Provider | Model | Dimensions | Use Case |
|----------|-------|------------|----------|
| OpenAI | text-embedding-3-small | 1536 | General purpose |
| OpenAI | text-embedding-3-large | 3072 | Higher accuracy |
| HuggingFace | all-MiniLM-L6-v2 | 384 | Local, free |
| Google | embedding-001 | 768 | Google ecosystem |

### 2. Vector Stores

Vector stores enable semantic search over embeddings:

```python
from langchain_community.vectorstores import Chroma, FAISS

# ChromaDB (persistent storage)
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# FAISS (in-memory, fast)
vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=embeddings
)
```

**Comparison:**

| Vector Store | Persistence | Speed | Best For |
|--------------|-------------|-------|----------|
| ChromaDB | Built-in | Good | Development, small-medium |
| FAISS | Manual | Fast | Large datasets, production |
| Pinecone | Cloud | Good | Production, managed |
| Weaviate | Built-in | Good | Hybrid search |

### 3. Document Processing

Before storing documents, you need to load and split them:

```python
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load documents
loader = PyPDFLoader("document.pdf")
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(documents)
```

**Chunking Strategies:**

| Strategy | Use Case |
|----------|----------|
| `RecursiveCharacterTextSplitter` | General purpose, respects structure |
| `TokenTextSplitter` | When token limits matter |
| `MarkdownTextSplitter` | Markdown documents |
| `PythonCodeTextSplitter` | Code files |

### 4. Retrievers

Retrievers fetch relevant documents for a query:

```python
# Basic retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# With score threshold
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.5}
)

# Use the retriever
docs = retriever.invoke("What is machine learning?")
```

### 5. RAG Chain

Combine retrieval with generation:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Create prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the question based on the context below.

Context: {context}

Question: {question}

Answer:
""")

# Helper function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build the RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Use it
answer = rag_chain.invoke("What is the main topic?")
```

## Sequential Chains

Chain multiple operations together:

### Simple Sequential Chain

```python
from langchain_core.runnables import RunnableSequence

# Each chain transforms the output
chain1 = prompt1 | llm | parser  # Returns string
chain2 = prompt2 | llm | parser  # Takes string, returns string

# Connect them
full_chain = chain1 | chain2
```

### Complex Sequential Processing

```python
# Multi-step processing
summarize_chain = summarize_prompt | llm | StrOutputParser()
analyze_chain = analyze_prompt | llm | StrOutputParser()
conclude_chain = conclude_prompt | llm | StrOutputParser()

# Pipeline
result = (
    {"text": RunnablePassthrough()}
    | summarize_chain
    | {"summary": RunnablePassthrough()}
    | analyze_chain
    | {"analysis": RunnablePassthrough()}
    | conclude_chain
)
```

## Best Practices

### 1. Chunking Strategy

```python
# Good: Appropriate chunk size with overlap
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Not too large, not too small
    chunk_overlap=200  # 20% overlap for context
)

# Add metadata
for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = i
    chunk.metadata["source"] = "document.pdf"
```

### 2. Retrieval Quality

```python
# Use metadata filtering
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {"source": "specific_doc.pdf"}
    }
)

# Combine with reranking
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
```

### 3. RAG Prompt Engineering

```python
# Good: Clear instructions, handle missing context
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer based on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Only use information from the context
- If the answer isn't in the context, say "I don't have that information"
- Be concise but complete

Answer:
""")
```

## Architecture Patterns

### Basic RAG

```
Query → Embed → Search → Retrieve → Prompt → Generate
```

### RAG with Reranking

```
Query → Embed → Search → Retrieve → Rerank → Prompt → Generate
```

### Hybrid Search

```
Query → [Semantic Search + Keyword Search] → Merge → Rerank → Generate
```

## Summary

| Concept | Purpose |
|---------|---------|
| Embeddings | Text to vectors |
| Vector Stores | Semantic storage |
| Chunking | Document preparation |
| Retrievers | Context fetching |
| RAG Chain | Augmented generation |

## Next Steps

Proceed to the [Hands-on exercises](./hands-on) to build your own RAG system.
