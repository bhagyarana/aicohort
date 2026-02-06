---
sidebar_position: 4
title: "Resources"
description: Module 2 additional resources and references
---

# Module 2 Resources

Additional learning materials for embeddings, vector stores, and RAG.

## Official Documentation

| Resource | Description |
|----------|-------------|
| [LangChain Retrieval](https://python.langchain.com/docs/concepts/#retrieval) | Retrieval concepts |
| [Vector Stores Guide](https://python.langchain.com/docs/integrations/vectorstores/) | All vector store integrations |
| [Text Splitters](https://python.langchain.com/docs/how_to/#text-splitters) | Document chunking |
| [RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/) | Official RAG tutorial |

## Vector Store Documentation

| Store | Documentation |
|-------|--------------|
| ChromaDB | [docs.trychroma.com](https://docs.trychroma.com/) |
| FAISS | [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki) |
| Pinecone | [docs.pinecone.io](https://docs.pinecone.io/) |
| Weaviate | [weaviate.io/developers](https://weaviate.io/developers/weaviate) |

## Embedding Models

| Provider | Model | Use Case |
|----------|-------|----------|
| OpenAI | text-embedding-3-small | Cost-effective |
| OpenAI | text-embedding-3-large | High accuracy |
| Cohere | embed-multilingual-v3.0 | Multilingual |
| HuggingFace | sentence-transformers | Local, free |

## Sample Code

### Multi-Source RAG

```python
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader
)

# Load from multiple sources
pdf_docs = PyPDFLoader("doc.pdf").load()
text_docs = TextLoader("notes.txt").load()
web_docs = WebBaseLoader("https://example.com").load()

# Combine all documents
all_docs = pdf_docs + text_docs + web_docs

# Process and index
chunks = splitter.split_documents(all_docs)
vectorstore = Chroma.from_documents(chunks, embeddings)
```

### Hybrid Search

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# Keyword-based retriever
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 4

# Vector retriever
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Combine both
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)
```

### Conversational RAG

```python
from langchain_core.prompts import MessagesPlaceholder

conversational_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the context to answer questions."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Context: {context}\n\nQuestion: {question}")
])

# Include chat history in chain
chain = (
    {
        "context": retriever | format_docs,
        "question": lambda x: x["question"],
        "chat_history": lambda x: x["chat_history"]
    }
    | conversational_prompt
    | llm
    | StrOutputParser()
)
```

## Module Files

Available in `module/Module 2/`:

```
Module 2/
├── 4_utilities/
│   ├── embeddings.ipynb
│   └── vector_stores.ipynb
├── 6_chains/
│   ├── sequential_chains.ipynb
│   └── simple_sequential.ipynb
├── 7_rag/
│   └── rag_basics.ipynb
├── 8_sample_project/
│   ├── pdf_qa.ipynb
│   └── data/
└── data/
    └── state_of_the_union.txt
```

## Common Issues

### Memory Issues with Large Documents

```python
# Process in batches
batch_size = 100
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    vectorstore.add_documents(batch)
```

### Slow Retrieval

```python
# Use FAISS for faster search
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")

# Load later
vectorstore = FAISS.load_local("faiss_index", embeddings)
```

## Next Module

Continue to [Module 3: LangGraph](/learn/modules/module-3) to learn:
- Stateful agent workflows
- Graph-based agent design
- Memory and state management
- Complex multi-step agents
