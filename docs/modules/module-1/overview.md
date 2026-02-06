---
sidebar_position: 2
title: "Overview"
description: Module 1 theory and concepts
---

# Module 1 Overview

This overview covers the theoretical foundations you need before diving into the hands-on exercises.

## What is LangChain?

LangChain is a framework for developing applications powered by large language models (LLMs). It provides:

- **Modular Components**: Building blocks for LLM applications
- **Chains**: Pre-built sequences of components
- **Memory**: State management across interactions
- **Agents**: LLMs that can use tools and make decisions

```
┌─────────────────────────────────────────────────────────┐
│                    LangChain Ecosystem                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│   │  Models  │  │ Prompts  │  │  Chains  │  │ Agents │ │
│   └──────────┘  └──────────┘  └──────────┘  └────────┘ │
│                                                          │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│   │  Memory  │  │  Tools   │  │ Retrieval│  │Callbacks│ │
│   └──────────┘  └──────────┘  └──────────┘  └────────┘ │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Core Concepts

### 1. Chat Models

Chat models are the interface to LLMs. LangChain supports multiple providers:

| Provider | Model | Use Case |
|----------|-------|----------|
| OpenAI | gpt-4o, gpt-4o-mini | General purpose, high quality |
| Google | gemini-1.5-pro | Long context, multimodal |
| Groq | llama-3.1-70b | Fast inference, cost-effective |
| Anthropic | claude-3-opus | Complex reasoning |

```python
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# Different providers, same interface
openai_llm = ChatOpenAI(model="gpt-4o-mini")
groq_llm = ChatGroq(model="llama-3.1-8b-instant")
google_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
```

### 2. Messages

LangChain uses a message-based interface:

```python
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

messages = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content="What is machine learning?"),
    AIMessage(content="Machine learning is..."),
    HumanMessage(content="Can you give an example?")
]
```

**Message Types:**
- `SystemMessage`: Sets the AI's behavior and context
- `HumanMessage`: User input
- `AIMessage`: Model responses
- `ToolMessage`: Results from tool execution

### 3. Prompt Templates

Prompt templates create reusable, parameterized prompts:

```python
from langchain_core.prompts import ChatPromptTemplate

# Simple template
template = ChatPromptTemplate.from_template(
    "Explain {concept} in simple terms."
)

# With system message
template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {domain}."),
    ("human", "{question}")
])

# Use the template
prompt = template.invoke({
    "domain": "machine learning",
    "question": "What are neural networks?"
})
```

### 4. LangChain Expression Language (LCEL)

LCEL is the declarative way to compose chains:

```python
from langchain_core.output_parsers import StrOutputParser

# The pipe operator composes components
chain = prompt_template | llm | StrOutputParser()

# Invoke the chain
result = chain.invoke({"topic": "AI agents"})
```

**LCEL Benefits:**
- Streaming support out of the box
- Async execution
- Parallel processing
- Fallbacks and retries
- Tracing and debugging

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Prompt    │────▶│     LLM     │────▶│   Parser    │
│  Template   │     │   (Model)   │     │  (Output)   │
└─────────────┘     └─────────────┘     └─────────────┘
                          │
                    LCEL Chain
                    chain = prompt | llm | parser
```

### 5. Output Parsers

Transform LLM text into structured data:

```python
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser
)
from pydantic import BaseModel

# String parser (default)
str_parser = StrOutputParser()

# JSON parser
json_parser = JsonOutputParser()

# Structured output with Pydantic
class MovieReview(BaseModel):
    title: str
    rating: int
    summary: str

# LLM with structured output
structured_llm = llm.with_structured_output(MovieReview)
```

## Architecture Patterns

### Simple Chain

```python
# Basic Q&A chain
chain = prompt | llm | parser
response = chain.invoke({"question": "What is AI?"})
```

### Chain with Multiple Steps

```python
# Multi-step processing
from langchain_core.runnables import RunnablePassthrough

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)
```

### Branching Logic

```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: "code" in x, code_chain),
    (lambda x: "math" in x, math_chain),
    default_chain  # fallback
)
```

## Best Practices

### 1. Prompt Engineering

```python
# Good: Specific, structured prompt
prompt = """You are a technical writer.

Task: Explain the following concept.
Concept: {concept}

Requirements:
- Use simple language
- Include one example
- Keep it under 100 words

Response:"""

# Bad: Vague prompt
prompt = "Tell me about {concept}"
```

### 2. Temperature Settings

| Temperature | Use Case |
|-------------|----------|
| 0.0 | Factual, deterministic responses |
| 0.3-0.5 | Balanced creativity and consistency |
| 0.7-1.0 | Creative, varied responses |

### 3. Error Handling

```python
from langchain_core.runnables import RunnableWithFallbacks

# Chain with fallback
robust_chain = main_llm.with_fallbacks([backup_llm])
```

## Summary

| Concept | Purpose |
|---------|---------|
| Chat Models | Interface to LLMs |
| Messages | Structured communication |
| Prompt Templates | Reusable prompts |
| LCEL | Composable chains |
| Output Parsers | Structured responses |

## Next Steps

Now that you understand the concepts, proceed to the [Hands-on exercises](./hands-on) to practice what you've learned.
