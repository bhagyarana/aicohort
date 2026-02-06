---
sidebar_position: 4
title: "Resources"
description: Module 1 additional resources and references
---

# Module 1 Resources

Additional learning materials and references for Module 1.

## Official Documentation

| Resource | Description |
|----------|-------------|
| [LangChain Docs](https://python.langchain.com/docs/) | Official LangChain documentation |
| [LCEL Conceptual Guide](https://python.langchain.com/docs/concepts/#langchain-expression-language-lcel) | Understanding LCEL |
| [Chat Models](https://python.langchain.com/docs/concepts/#chat-models) | Working with chat models |
| [Prompt Templates](https://python.langchain.com/docs/concepts/#prompt-templates) | Prompt template guide |

## LLM Provider Documentation

| Provider | Documentation |
|----------|--------------|
| OpenAI | [API Reference](https://platform.openai.com/docs/api-reference) |
| Google AI | [Gemini API](https://ai.google.dev/docs) |
| Groq | [Groq Docs](https://console.groq.com/docs) |
| Anthropic | [Claude API](https://docs.anthropic.com/) |

## Jupyter Notebooks

The following notebooks are available in the `module/Module 1/` directory:

```
Module 1/
├── 1_langchain_openai.ipynb    # Basic LangChain with OpenAI
├── 2_prompt_templates.ipynb    # Prompt template engineering
├── 3_lcel.ipynb                # LangChain Expression Language
├── 4_output_parser.ipynb       # Output parsing techniques
├── sequential_chain.ipynb      # Sequential chain patterns
├── Installation.txt            # Setup instructions
├── requirements.txt            # Python dependencies
└── References.txt              # Additional references
```

## Code Snippets

### Quick Reference: LLM Initialization

```python
# OpenAI
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")

# Google
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Groq
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.1-8b-instant")

# Anthropic
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-3-sonnet-20240229")
```

### Quick Reference: Prompt Templates

```python
from langchain_core.prompts import ChatPromptTemplate

# From template string
prompt = ChatPromptTemplate.from_template("Tell me about {topic}")

# From messages
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are {role}."),
    ("human", "{question}")
])

# With placeholders
prompt = ChatPromptTemplate.from_messages([
    ("system", "Context: {context}"),
    ("placeholder", "{history}"),
    ("human", "{input}")
])
```

### Quick Reference: LCEL Patterns

```python
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda
)

# Basic chain
chain = prompt | llm | parser

# With passthrough
chain = {"query": RunnablePassthrough()} | prompt | llm

# Parallel execution
chain = RunnableParallel(
    result1=chain1,
    result2=chain2
)

# Custom function
chain = RunnableLambda(lambda x: x.upper()) | llm
```

## Video Resources

Recommended video tutorials:

| Topic | Platform | Link |
|-------|----------|------|
| LangChain Crash Course | YouTube | Search "LangChain tutorial 2024" |
| Prompt Engineering | DeepLearning.AI | ChatGPT Prompt Engineering |
| LCEL Deep Dive | LangChain | Official YouTube channel |

## Articles & Blogs

- [LangChain Blog](https://blog.langchain.dev/) - Official updates and tutorials
- [Prompt Engineering Guide](https://www.promptingguide.ai/) - Comprehensive prompt techniques
- [LLM University](https://cohere.com/llmu) - Cohere's educational content

## Practice Projects

Try building these to reinforce your learning:

### 1. FAQ Bot
```python
# Build a bot that answers questions about a specific topic
# Use prompt templates with context injection
```

### 2. Code Translator
```python
# Translate code between programming languages
# Use structured output for the result
```

### 3. Content Summarizer
```python
# Summarize long text documents
# Implement streaming for real-time output
```

## Common Issues & Solutions

### Issue: API Key Not Found

```python
# Solution: Ensure .env file is loaded
from dotenv import load_dotenv
load_dotenv()  # Call this before initializing LLMs
```

### Issue: Rate Limiting

```python
# Solution: Add exponential backoff
import time
from tenacity import retry, wait_exponential

@retry(wait=wait_exponential(min=1, max=60))
def call_llm(prompt):
    return llm.invoke(prompt)
```

### Issue: Token Limits

```python
# Solution: Use a model with larger context
llm = ChatOpenAI(model="gpt-4o")  # 128k context

# Or truncate input
from langchain.text_splitter import TokenTextSplitter
splitter = TokenTextSplitter(chunk_size=4000)
```

## Next Module

Ready to advance? Continue to [Module 2: Advanced LangChain](/learn/modules/module-2) to learn about:
- Embeddings and vector stores
- RAG (Retrieval Augmented Generation)
- Complex chain patterns
- Document processing
