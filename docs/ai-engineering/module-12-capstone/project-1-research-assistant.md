---
sidebar_position: 2
title: "Project 1: AI Research Assistant"
description: Build a research assistant that answers questions with citations from retrieved academic papers — combining RAG, semantic search, and grounded generation.
---

# Project 1: AI Research Assistant

**Difficulty:** Intermediate
**Modules:** 1, 2, 3, 4, 5
**Time:** 8–12 hours

---

## Problem Statement

Build a research assistant that:
1. Accepts a research question in natural language
2. Queries a corpus of academic papers (arXiv or a local document set)
3. Retrieves the most relevant passages using semantic search
4. Answers the question with inline citations and a confidence score
5. Distinguishes clearly between what is supported by the retrieved documents and what is not

**What makes this hard**: Faithfulness. The system must ground every claim in a specific retrieved passage. It must not hallucinate citations, invent authors, or synthesize conclusions not present in the source material.

---

## Architecture

```
User question
      ↓
[Query Expansion]
  LLM rewrites question into 2–3 search-optimized variants
      ↓
[Retrieval]
  Embed all query variants → search vector DB → union top-K results
      ↓
[Reranking]
  Cross-encoder reranker → select top 5 most relevant passages
      ↓
[Grounded Generation]
  LLM answers using ONLY the retrieved passages
  Each claim must cite a specific passage by [ref number]
      ↓
[Confidence Scoring]
  Evaluate: how well does the answer match the retrieved context?
      ↓
Answer with citations + confidence score
```

---

## Implementation Guide

### Phase 1: Document Ingestion (2–3 hours)

Build the pipeline to load, chunk, embed, and store documents.

```python
import os
import json
from pathlib import Path
import openai
import chromadb
from chromadb.utils import embedding_functions

client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Initialize ChromaDB
chroma = chromadb.PersistentClient(path="./research_db")
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name="text-embedding-3-small"
)
collection = chroma.get_or_create_collection(
    name="papers",
    embedding_function=openai_ef
)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def ingest_paper(paper_id: str, title: str, authors: str, abstract: str, body: str):
    """Ingest a single paper into the vector database."""
    # Combine abstract + body for chunking
    full_text = f"Title: {title}\nAuthors: {authors}\n\nAbstract: {abstract}\n\n{body}"
    chunks = chunk_text(full_text)

    chunk_ids = [f"{paper_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {"paper_id": paper_id, "title": title, "authors": authors, "chunk_index": i}
        for i in range(len(chunks))
    ]

    collection.add(
        documents=chunks,
        ids=chunk_ids,
        metadatas=metadatas
    )
    print(f"Ingested '{title}' — {len(chunks)} chunks")
```

**For your document corpus**, use one of:
- A folder of PDF files (use PyMuPDF to extract text)
- arXiv API to fetch papers by keyword
- Pre-downloaded `.txt` files of papers you've chosen

### Phase 2: Query and Retrieval (2 hours)

```python
def expand_query(question: str) -> list[str]:
    """Generate search-optimized query variants."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Generate 3 search query variants for this research question.
Each variant should emphasize different aspects.
Return as a JSON list of strings.

Question: {question}"""
        }],
        response_format={"type": "json_object"},
        max_tokens=200
    )
    import json
    data = json.loads(response.choices[0].message.content)
    # Handle different JSON shapes
    queries = data.get("queries", data.get("variants", [question]))
    return [question] + queries[:2]  # original + 2 variants


def retrieve_passages(question: str, n_results: int = 10) -> list[dict]:
    """Retrieve passages using query expansion + union deduplication."""
    queries = expand_query(question)
    all_results = {}

    for query in queries:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )

        for i, doc_id in enumerate(results["ids"][0]):
            if doc_id not in all_results:
                all_results[doc_id] = {
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i]
                }

    # Sort by distance (lower = more similar) and return top results
    sorted_results = sorted(all_results.values(), key=lambda x: x["distance"])
    return sorted_results[:n_results]
```

### Phase 3: Grounded Generation (2–3 hours)

```python
GROUNDED_GENERATION_PROMPT = """You are a research assistant. Answer the question based ONLY on the provided passages.

Rules:
1. Every factual claim must cite a passage using [Ref N] notation
2. If a claim is not supported by any passage, explicitly state: "This is not covered in the retrieved documents."
3. Do not synthesize conclusions not present in the passages
4. End with a "Confidence" section: rate your confidence 1-5 and explain why

Passages:
{passages}

Question: {question}

Answer with citations:"""


def generate_grounded_answer(question: str, passages: list[dict]) -> dict:
    """Generate a grounded answer with citations."""
    # Format passages with reference numbers
    formatted_passages = "\n\n".join(
        f"[Ref {i+1}] From '{p['metadata']['title']}' by {p['metadata']['authors']}:\n{p['text']}"
        for i, p in enumerate(passages)
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": GROUNDED_GENERATION_PROMPT.format(
                passages=formatted_passages,
                question=question
            )
        }],
        max_tokens=1000
    )

    return {
        "answer": response.choices[0].message.content,
        "passages_used": len(passages),
        "sources": [
            {"ref": i+1, "title": p["metadata"]["title"], "authors": p["metadata"]["authors"]}
            for i, p in enumerate(passages)
        ]
    }


def research_assistant(question: str) -> dict:
    """Full research assistant pipeline."""
    print(f"\nProcessing: {question}")

    passages = retrieve_passages(question, n_results=8)
    print(f"Retrieved {len(passages)} passages")

    result = generate_grounded_answer(question, passages[:5])  # Top 5 after retrieval
    return result
```

### Phase 4: Evaluation (1–2 hours)

Evaluate faithfulness — does the answer only use information from retrieved passages?

```python
FAITHFULNESS_EVAL_PROMPT = """You are evaluating an AI-generated answer for faithfulness to source documents.

Answer to evaluate:
{answer}

Source passages:
{passages}

For each claim in the answer, determine if it is:
- SUPPORTED: directly stated or logically implied by the passages
- NOT_SUPPORTED: not found in the passages, or contradicts them

Return JSON:
{{
  "faithfulness_score": 0.0-1.0,
  "supported_claims": ["..."],
  "unsupported_claims": ["..."],
  "verdict": "FAITHFUL | PARTIALLY_FAITHFUL | UNFAITHFUL"
}}"""


def evaluate_faithfulness(answer: str, passages: list[dict]) -> dict:
    """Use LLM-as-judge to evaluate answer faithfulness."""
    formatted_passages = "\n".join(p["text"] for p in passages)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": FAITHFULNESS_EVAL_PROMPT.format(
                answer=answer,
                passages=formatted_passages
            )
        }],
        response_format={"type": "json_object"},
        max_tokens=800
    )

    import json
    return json.loads(response.choices[0].message.content)
```

---

## Evaluation Criteria

Your research assistant is working well when:

| Criterion | Target | How to Measure |
|-----------|--------|----------------|
| Faithfulness score | > 0.85 | Run `evaluate_faithfulness` on 10 answers |
| Citation accuracy | All `[Ref N]` point to real passages | Manual spot check |
| Retrieval precision | Top 5 passages are all relevant | Human relevance judgment on 10 queries |
| "Not found" rate | Correctly says "not in documents" for out-of-scope questions | Test 5 out-of-scope questions |
| Response coherence | Answer reads naturally with citations | Human evaluation |

---

## Stretch Goals

1. **Add arXiv live search**: Query arXiv API in real-time instead of a static corpus
2. **Multi-hop reasoning**: For questions requiring synthesis across multiple papers, chain two retrieval steps
3. **Confidence calibration**: Compare stated confidence scores against actual accuracy
4. **Export to markdown**: Generate a formatted research summary with bibliography
5. **Conversation mode**: Allow follow-up questions that maintain context from previous turns

---

## Common Failure Modes to Investigate

- **Hallucinated citations**: Model invents `[Ref 3]` when only 2 passages exist → add validation
- **Wrong answer despite correct retrieval**: Model ignores retrieved context → move context higher in prompt
- **Over-retrieval**: Top-K too high → context window overflow or diluted attention → reduce K or add reranker
- **Query expansion goes wrong**: Expanded queries retrieve unrelated content → tune expansion prompt
