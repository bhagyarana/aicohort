---
sidebar_position: 2
title: "Overview"
description: Module 2 theory and concepts
---

# Module 2 Overview

This overview covers the theoretical foundations of embeddings, vector stores, and RAG systems.

## Memory vs State — A Critical Distinction

One of the most confused concepts in LangChain: **Memory** and **State** sound similar but serve completely different purposes.

| | Memory | State |
|--|--------|-------|
| **Lives in** | LangChain memory classes | LangGraph `TypedDict` |
| **Scope** | Conversation history between a user and LLM | Full application context across all nodes |
| **Manages** | What the LLM "remembers" about past messages | What the entire workflow knows right now |
| **Analogy** | Your personal notebook of past conversations | The whiteboard in a team meeting room |

**In short**: Memory is about *conversation history*. State is about *workflow data*.

### LangChain Memory Classes

| Class | What it stores | Token behavior |
|-------|---------------|----------------|
| `ConversationBufferMemory` | All messages verbatim | Grows forever |
| `ConversationBufferWindowMemory` | Last N messages | Fixed window |
| `ConversationTokenBufferMemory` | Messages up to token limit | Predictable cost |
| `ConversationSummaryMemory` | LLM-generated summary of history | Low & stable |

### RunnableWithMessageHistory — The Modern Approach

`RunnableWithMessageHistory` is the current recommended way to add conversation memory to any chain. Think of it as a **wrapper** that handles session management automatically.

> **Analogy**: Imagine a doctor's office. The AI model is the doctor — brilliant but forgetful. `RunnableWithMessageHistory` is the diligent administrative assistant who:
> - Maintains a file cabinet (the message store) of every patient's history
> - When a patient walks in with their Patient ID (Session ID), pulls the right file
> - Hands context to the doctor before each appointment
> - Files the notes back afterward
> - Ensures the doctor never mixes up one patient's history with another's

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# The "file cabinet" — stores histories by session ID
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Your chain (any LCEL chain)
chain = prompt | llm | StrOutputParser()

# Wrap it with history management
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# User A's session
chain_with_history.invoke(
    {"question": "What is RAG?"},
    config={"configurable": {"session_id": "user-alice"}}
)

# User B's session — completely separate history
chain_with_history.invoke(
    {"question": "How do embeddings work?"},
    config={"configurable": {"session_id": "user-bob"}}
)
```

**Key benefits**: Multi-user support, session isolation, debugging, low-latency replay — without you writing any session logic.

---

### Quiz: Memory vs State

> **Q: You're building a multi-agent research system where an Orchestrator calls a Researcher and a Writer. The Researcher fetches articles, the Writer drafts content, and the Orchestrator reviews it. Which do you use — Memory or State?**
>
> <details>
> <summary>Show Answer</summary>
>
> **State** (via LangGraph). Memory manages *conversation history between user and AI*. What you need here is *workflow state* — the fetched articles, draft content, and review status all need to flow between nodes in your graph. Use a `TypedDict` state object in LangGraph. Memory would only be relevant if you wanted the chatbot to remember what it discussed with a user across multiple sessions.
> </details>

---

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

## Naïve RAG — What Can Go Wrong

The basic RAG pipeline (chunk → embed → retrieve → generate) works surprisingly well for simple cases. But in production, it breaks in predictable ways.

> **Analogy**: Naïve RAG is like a librarian who looks for books with the *exact title you mentioned*. If you ask for "books about money management for young professionals" but the book is titled "Millennial Finance Guide," the librarian misses it entirely.

### The Four Failure Modes

| Failure | What happens | Why it hurts |
|---------|-------------|--------------|
| **Limited contextual understanding** | Similarity search fails on complex multi-part questions | Wrong chunks retrieved, answer is off-topic |
| **Inconsistent quality** | Retriever and generator work independently | Poor retrieval cascades into poor generation |
| **Scaling issues** | Large document sets slow similarity search | Latency spikes in production |
| **Hallucination** | Retrieved chunks are misaligned with question | Model fills gaps with made-up information |

### Advanced RAG — Fix It Before It Breaks

Instead of fixing retrieval *after* the fact, Advanced RAG improves the *query itself* before hitting the vector store.

#### Strategy 1: Query Expansion with a Generated Answer

**Idea**: Ask the LLM to generate a *hypothetical answer* to your question first, then use that hypothetical answer to search the vector store. Similar documents will be closer to what a real answer looks like than to the raw question.

```
Original query: "What are the tax implications of early 401k withdrawal?"
                           │
                           ▼
           LLM generates hypothetical answer:
    "Early 401k withdrawal before age 59½ typically incurs a 10%
     penalty plus ordinary income tax. Exceptions include..."
                           │
                           ▼
         Search vector store with the HYPOTHETICAL ANSWER
         (retrieves more relevant chunks)
```

```python
from langchain_core.output_parsers import StrOutputParser

# Step 1: Generate a hypothetical answer
hypo_prompt = ChatPromptTemplate.from_template(
    "Generate a hypothetical answer to this question for retrieval purposes: {question}"
)
hypo_chain = hypo_prompt | llm | StrOutputParser()

# Step 2: Use the hypothetical answer to retrieve
def rag_with_hypo(question: str) -> str:
    hypothetical = hypo_chain.invoke({"question": question})
    docs = retriever.invoke(hypothetical)          # search with hypo answer
    context = format_docs(docs)
    return rag_chain.invoke({"context": context, "question": question})
```

#### Strategy 2: Query Expansion with Multiple Sub-Queries

**Idea**: Generate 5 related sub-questions from the original query, retrieve documents for each, then merge and deduplicate before sending to the LLM. Ensures comprehensive coverage even for vague original questions.

```
Original: "Tell me about LangChain"
                    │
                    ▼
    LLM generates 5 sub-queries:
    1. "What is LangChain and what problems does it solve?"
    2. "What are the core components of LangChain?"
    3. "How does LangChain compare to other LLM frameworks?"
    4. "What are real-world use cases for LangChain?"
    5. "How do you get started with LangChain?"
                    │
                    ▼
    Retrieve docs for each → Merge → Deduplicate → Generate
```

> **Analogy**: Advanced RAG is the research assistant who first imagines what the answer might look like AND asks five different follow-up questions — ensuring they find every relevant book in the library, not just the obvious ones.

### Production-Ready Conversational RAG — Full Architecture

When you're ready to ship a RAG chatbot, it needs more than just the chain:

```
┌─────────────────────────────────────────────────────────────┐
│              Production RAG Chatbot Architecture             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Streamlit / React Frontend                                  │
│       │  (upload files, chat UI)                             │
│       ▼                                                      │
│  FastAPI Backend                                             │
│   ├── POST /chat      → RunnableWithMessageHistory chain     │
│   ├── POST /upload    → Document ingestion pipeline          │
│   ├── GET  /files     → List indexed documents               │
│   └── DELETE /files   → Remove from vector store            │
│       │                                                      │
│       ▼                                                      │
│  Pydantic Models → Strict request/response validation        │
│  Session IDs     → Per-user conversation isolation          │
│  ChromaDB        → Vector store with persistence            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

Key production additions beyond the basic chain:
- **Pydantic request/response models** — validate inputs before they hit the LLM
- **Multi-user sessions** — unique Session IDs per user via `RunnableWithMessageHistory`
- **File management API** — upload, list, delete documents without restarting
- **Streaming responses** — use `chain.astream()` for real-time token display

---

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

### Knowledge Graph: When Relationships Matter More Than Similarity

Vector RAG finds *similar* content. But sometimes you need to traverse *relationships*.

**Example — a legal compliance agent:**

```
Vector RAG: "Find text similar to 'GDPR data retention requirements'"
Knowledge Graph: "Find all contracts → that cite clause 7b → which is subject to
                  GDPR Article 17 → with review dates before 2026"
```

A knowledge graph stores entities (contracts, clauses, regulations) and their relationships — so the agent can reason about *connections*, not just similarity.

**When to use Knowledge Graph:**

| Scenario | Best Tool |
|---------|-----------|
| "Find documents about topic X" | Vector RAG |
| "Find how entity A relates to entity B" | Knowledge Graph |
| "Find all items affected by a cascade change" | Knowledge Graph |
| "Search across millions of text chunks" | Vector RAG |

> Most teams don't need a knowledge graph until dealing with structured, highly relational data (legal, compliance, financial, supply chain). If you're not sure you need it — you probably don't yet.

---

:::warning Context Window ≠ Memory
The context window is a **sliding window of recent tokens** — not a persistent store. RAG retrieves external knowledge *into* the context window for each request. If retrieved content is too large, it pushes out earlier conversation history. Engineer your chunk sizes and retrieval counts to leave room for conversation history, not just documents.
:::

## Summary

| Concept | Purpose |
|---------|---------|
| Embeddings | Text to vectors |
| Vector Stores | Semantic storage |
| Chunking | Document preparation |
| Retrievers | Context fetching |
| RAG Chain | Augmented generation |
| Knowledge Graph | Relationship traversal |

## Test Your Understanding

---

> **Q1: Your RAG system retrieves 4 documents but the final answer is completely wrong. What's the most likely culprit in Naïve RAG?**
>
> <details>
> <summary>Show Answer</summary>
>
> **Poor retrieval quality**. In Naïve RAG, the retriever and generator work *independently* — there's no feedback loop. If the similarity search retrieved chunks that look relevant (same keywords) but lack the actual answer, the LLM will either hallucinate or say "I don't know." This is the "inconsistent quality" failure mode. Fix: try Advanced RAG with query expansion, or add a re-ranker.
> </details>

---

> **Q2: What's the key architectural difference between Naïve RAG and Advanced RAG with Query Expansion?**
>
> <details>
> <summary>Show Answer</summary>
>
> **Where the LLM is involved**. In Naïve RAG, the LLM is only called *after* retrieval (to generate the answer). In Advanced RAG with query expansion, the LLM is called *before* retrieval too — to either generate a hypothetical answer or multiple sub-queries. This pre-retrieval LLM call dramatically improves which documents get fetched.
> </details>

---

> **Q3: Two users are chatting with your `RunnableWithMessageHistory` bot simultaneously. User A asks about stocks, User B asks about cooking. What prevents the bot from mixing up their conversation histories?**
>
> <details>
> <summary>Show Answer</summary>
>
> **The Session ID**. Each user has a unique `session_id` passed in the config. `get_session_history(session_id)` retrieves the right `ChatMessageHistory` object for each user. The two histories are stored separately in the `store` dict. Without session IDs, all users would share one conversation history — a serious bug in any multi-user system.
> </details>

---

> **Q4: Your document corpus has 10 million chunks. Your RAG pipeline is slow. Which Naïve RAG failure mode is this, and what's the fix direction?**
>
> <details>
> <summary>Show Answer</summary>
>
> **Scaling & Robustness Issues**. At 10M chunks, brute-force similarity search is expensive. Fix direction: use approximate nearest neighbor (ANN) indexing (FAISS with IVF, Pinecone, Weaviate), add metadata filtering to narrow the search space before semantic search, and consider hierarchical retrieval (coarse → fine).
> </details>

---

## Next Steps

Proceed to the [Hands-on exercises](./hands-on) to build your own RAG system.
