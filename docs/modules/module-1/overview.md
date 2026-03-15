---
sidebar_position: 2
title: "Overview"
description: Module 1 theory and concepts
---

# Module 1 Overview

This overview covers the theoretical foundations you need before diving into the hands-on exercises.

## Prompt Engineering — Before You Write a Single Line of Code

Prompt Engineering is the **art and science of designing effective instructions** for LLMs. The difference between a great prompt and a mediocre one can mean the difference between a useful AI application and a frustrating one.

> "It's not about talking to AI — it's about asking the right questions."

### Anatomy of a Great Prompt

Every effective prompt has these five ingredients:

| Component | What it is | Example |
|-----------|-----------|---------|
| **Persona** | What role should the AI play? | "You are a senior financial analyst..." |
| **Instructions** | What should the model do? | "Summarize the key risks in..." |
| **Input Content** | The text/data to process | "Here is the earnings report: {text}" |
| **Format** | Expected output structure | "Return as bullet points" |
| **Context** | Background/additional info | "The audience is non-technical" |

### The Three Prompt Categories

```
Zero-Shot        → No examples | Uses what the model already knows
Few-Shot         → 2-5 examples | Guides the model toward your format
Chain-of-Thought → Step-by-step guidance | Forces structured reasoning
```

**When to use which?**
- Zero-Shot: Simple tasks where the model is likely to get it right
- Few-Shot: Specific output formats, domain-specific language
- Chain-of-Thought: Math problems, multi-step reasoning, complex analysis

### Vague vs Specific — The Golden Rule

```
Bad:  "Improve this email."

Good: "Rewrite this email to sound more confident and concise when
       asking a stakeholder for feedback on a technical proposal.
       Keep it under 100 words."
```

The model can only work with what you give it. Be your own editor first.

### Temperature & Other Parameters That Matter

| Parameter | What it controls | Sweet spot |
|-----------|-----------------|------------|
| **Temperature** | Creativity vs predictability | 0 for facts, 0.7 for creativity |
| **Max Tokens** | Length of response | Set based on expected output |
| **Top P** | Limits word choices to top % of likely words | Usually leave at default |
| **Stop Sequence** | Where to stop generating | Useful for structured outputs |

:::tip Quick Self-Check
After writing a prompt, ask: "Could a new employee understand exactly what I want from this?" If not, add more specificity.
:::

---

### Quiz: Prompt Engineering Fundamentals

> **Q: You're building a customer support bot. A user asks: "My order is late." Your prompt says "You are a helpful assistant." What's the biggest problem?**
>
> <details>
> <summary>Show Answer</summary>
>
> **Missing persona, instructions, and context.** The model has no idea it's a customer support agent, what company it represents, what tone to use, or what actions it can take. A better prompt would define the role ("You are a customer support agent for AcmeCorp"), the constraints ("Only recommend solutions we can actually execute: refund, reship, or escalate"), and the tone ("Be empathetic and solution-oriented").
>
> Vague prompts produce generic answers. Specific prompts produce useful ones.
> </details>

---

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

### The Essential Runnable Types

LCEL works because every component implements a common `Runnable` interface — meaning you can `invoke()` anything and pipe the output into the next step. Three special Runnables are worth knowing cold:

#### RunnablePassthrough — "Don't touch it"

Passes input through unchanged. Most useful when you need to keep the original question alongside retrieved context.

```python
from langchain_core.runnables import RunnablePassthrough

# Classic RAG pattern — question flows through unchanged
rag_chain = (
    {
        "context": retriever | format_docs,   # retriever transforms the query
        "question": RunnablePassthrough()     # question stays as-is
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

Think of it as the **"do nothing"** component — it exists purely to maintain data flow.

#### RunnableLambda — "Wrap any Python function"

Turns any Python function into a Runnable so it can participate in a chain.

```python
from langchain_core.runnables import RunnableLambda

def add_context(question: str) -> str:
    return f"Context: Financial domain. Question: {question}"

# Wrap it so it works in LCEL
chain = RunnableLambda(add_context) | prompt | llm | StrOutputParser()
```

#### RunnableParallel — "Do two things at once with the same input"

Runs multiple chains simultaneously on the same input. Like having two assistants read the same document and each give you a different analysis.

```python
from langchain_core.runnables import RunnableParallel

parallel = RunnableParallel(
    sentiment=sentiment_chain,    # analyzes sentiment
    summary=summary_chain,        # generates summary
)

# Both chains get the same input, output is a dict
result = parallel.invoke("The product launch was a massive success...")
# result = {"sentiment": "positive", "summary": "Product launch exceeded expectations..."}
```

### Advanced LCEL Composition

```python
# Nested chains — run sub-chains inside a chain
research_chain = search_chain | summarize_chain
full_chain = research_chain | format_chain | llm

# Adding keys on the fly — build context progressively
chain = (
    RunnablePassthrough.assign(
        summary=summary_chain,       # adds "summary" key to dict
        keywords=keyword_chain       # adds "keywords" key to dict
    )
    | final_prompt | llm
)

# Chain combination — combine multiple chains with the pipe
final = chain_a | chain_b | chain_c
```

:::note Why Runnable Matters
The `|` pipe operator only works because every component (prompt, LLM, parser, and the special Runnables) implements the same `Runnable` interface. Output of `invoke()` from one flows as input to the next. This is the backbone of all LangChain applications.
:::

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

## Memory Types — How LangChain Remembers

When your chain needs to remember previous messages in a conversation, LangChain gives you four memory strategies. Choosing the right one matters:

| Memory Type | How it works | Token Cost | Best For |
|-------------|-------------|------------|----------|
| **Conversation Buffer** | Stores all messages verbatim | Grows unboundedly | Short sessions, debugging |
| **Buffer Window** | Keeps last N message pairs | Fixed | Chatbots with sliding context |
| **Token Buffer** | Keeps messages up to token limit | Fixed | When you need predictable API costs |
| **Conversation Summary** | LLM summarizes old messages | Low | Long conversations, support agents |

### Real-World Analogy

Imagine you're a doctor seeing patients:
- **Buffer**: You read the patient's entire medical file every visit (comprehensive, expensive)
- **Buffer Window**: You only read the last 3 visits (fast, loses old context)
- **Token Buffer**: You read until you hit your page limit (predictable)
- **Summary**: Your nurse gives you a one-paragraph briefing before you walk in (efficient, loses detail)

---

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

## Test Your Understanding

These questions target the spots where most people get tripped up. Think carefully before revealing the answer.

---

> **Q1: What does `RunnablePassthrough` actually do in this chain?**
> ```python
> chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
> ```
>
> <details>
> <summary>Show Answer</summary>
>
> It passes the **original user question** through unchanged as the `"question"` key. Without it, the question would only flow to the retriever — you'd lose it by the time you hit the prompt. This pattern is the foundation of every RAG chain.
> </details>

---

> **Q2: Temperature=0 means the model will always give the exact same answer. True or False?**
>
> <details>
> <summary>Show Answer</summary>
>
> **Mostly true, but not guaranteed.** Temperature=0 makes the model deterministic — it always picks the highest-probability next token. In practice this means nearly identical outputs for identical inputs. However, very rarely, API-level non-determinism or model updates can produce slightly different results. For production, temperature=0 is the right choice for factual tasks.
> </details>

---

> **Q3: You have a chain that needs to do THREE things simultaneously with the same input: summarize it, extract keywords, AND check sentiment. Which Runnable do you use?**
>
> <details>
> <summary>Show Answer</summary>
>
> **`RunnableParallel`**. It executes multiple chains on the same input concurrently, returning a dict with all results. This is faster than chaining them sequentially and avoids making the same LLM call three times in series.
>
> ```python
> parallel = RunnableParallel(
>     summary=summary_chain,
>     keywords=keyword_chain,
>     sentiment=sentiment_chain
> )
> ```
> </details>

---

> **Q4: Which memory type would you use for a financial advisor chatbot that has year-long client conversations?**
>
> <details>
> <summary>Show Answer</summary>
>
> **Conversation Summary Memory**. Year-long conversations would blow past any token limit with Buffer or Buffer Window. Summary memory lets the LLM compress old context into a running summary, keeping token costs manageable while preserving the key history. The tradeoff: fine details from old conversations are lost in the summary.
> </details>

---

## Next Steps

Now that you understand the concepts, proceed to the [Hands-on exercises](./hands-on) to practice what you've learned.
