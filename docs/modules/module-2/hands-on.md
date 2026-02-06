---
sidebar_position: 3
title: "Hands-on"
description: Module 2 practical exercises
---

# Module 2 Hands-on Exercises

Build a complete RAG system through guided exercises.

## Exercise 1: Working with Embeddings

Learn to generate and use embeddings.

### Setup

```python
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```

### Generate Embeddings

```python
# Single query embedding
query = "What is machine learning?"
query_vector = embeddings.embed_query(query)
print(f"Vector dimensions: {len(query_vector)}")

# Multiple document embeddings
documents = [
    "Machine learning is a subset of AI.",
    "Deep learning uses neural networks.",
    "Python is a popular programming language."
]
doc_vectors = embeddings.embed_documents(documents)
print(f"Generated {len(doc_vectors)} vectors")
```

### Semantic Similarity

```python
import numpy as np
from scipy.spatial.distance import cosine

def similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

# Compare query to documents
query_vec = embeddings.embed_query("Tell me about AI")

for i, doc in enumerate(documents):
    doc_vec = doc_vectors[i]
    sim = similarity(query_vec, doc_vec)
    print(f"'{doc[:30]}...' - Similarity: {sim:.4f}")
```

:::tip Try It Yourself
Compare the similarity scores of related vs unrelated queries.
:::

---

## Exercise 2: Vector Store Setup

Set up ChromaDB for persistent storage.

### Create Vector Store

```python
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Create documents
documents = [
    Document(
        page_content="LangChain is a framework for LLM applications.",
        metadata={"source": "langchain_docs", "topic": "overview"}
    ),
    Document(
        page_content="Agents can use tools to accomplish tasks.",
        metadata={"source": "langchain_docs", "topic": "agents"}
    ),
    Document(
        page_content="RAG combines retrieval with generation.",
        metadata={"source": "langchain_docs", "topic": "rag"}
    ),
]

# Create vector store
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
```

### Query Vector Store

```python
# Similarity search
results = vectorstore.similarity_search(
    "How do AI agents work?",
    k=2
)

for doc in results:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print("---")
```

### Search with Scores

```python
# Get similarity scores
results_with_scores = vectorstore.similarity_search_with_score(
    "What is RAG?",
    k=3
)

for doc, score in results_with_scores:
    print(f"Score: {score:.4f} - {doc.page_content[:50]}...")
```

### Metadata Filtering

```python
# Filter by metadata
results = vectorstore.similarity_search(
    "Tell me about agents",
    k=2,
    filter={"topic": "agents"}
)
```

---

## Exercise 3: Document Processing

Load and process real documents.

### Load PDF Documents

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load PDF
loader = PyPDFLoader("path/to/document.pdf")
pages = loader.load()

print(f"Loaded {len(pages)} pages")
```

### Split Documents

```python
# Configure splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Split documents
chunks = splitter.split_documents(pages)
print(f"Created {len(chunks)} chunks")

# Examine a chunk
print(f"Chunk size: {len(chunks[0].page_content)}")
print(f"Metadata: {chunks[0].metadata}")
```

### Create Searchable Index

```python
# Build vector store from chunks
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./pdf_index"
)

# Test retrieval
results = vectorstore.similarity_search(
    "main topic of the document",
    k=3
)
```

---

## Exercise 4: Building a RAG Chain

Create a complete question-answering system.

### Setup Components

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)
```

### Create RAG Prompt

```python
rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant answering questions based on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Answer only based on the context provided
- If you cannot find the answer in the context, say "I don't have enough information to answer that"
- Be concise and accurate
- Quote relevant parts when helpful

Answer:
""")
```

### Build the Chain

```python
def format_docs(docs):
    """Format retrieved documents into a string."""
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )

# Complete RAG chain
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)
```

### Test the Chain

```python
# Ask questions
question = "What is the main purpose of LangChain?"
answer = rag_chain.invoke(question)
print(f"Q: {question}")
print(f"A: {answer}")
```

---

## Exercise 5: Complete RAG Project

Build a full document Q&A system.

### Project: PDF Question Answering

```python
"""
Complete RAG System for PDF Documents
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class PDFQuestionAnswering:
    def __init__(self, pdf_path: str):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.vectorstore = self._create_index(pdf_path)
        self.chain = self._create_chain()

    def _create_index(self, pdf_path: str):
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(pages)

        # Create vector store
        return Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )

    def _create_chain(self):
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

        prompt = ChatPromptTemplate.from_template("""
        Answer the question based on the following context from a PDF document.

        Context:
        {context}

        Question: {question}

        Provide a clear, accurate answer:
        """)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        return (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str) -> str:
        return self.chain.invoke(question)


# Usage
qa_system = PDFQuestionAnswering("your_document.pdf")
answer = qa_system.ask("What is the main topic?")
print(answer)
```

---

## Notebook Labs

Available in `module/Module 2/`:

| Directory | Content |
|-----------|---------|
| `4_utilities/` | Embeddings and vector store basics |
| `6_chains/` | Sequential chain patterns |
| `7_rag/` | RAG implementation |
| `8_sample_project/` | Complete RAG project |

## Checklist

Before moving to Module 3, ensure you can:

- [ ] Generate embeddings for text
- [ ] Set up and query vector stores
- [ ] Load and chunk documents
- [ ] Build RAG chains
- [ ] Handle metadata and filtering

## Next Steps

1. Review [Resources](./resources) for additional learning
2. Proceed to [Module 3: LangGraph](/learn/modules/module-3)
