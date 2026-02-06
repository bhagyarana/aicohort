---
sidebar_position: 3
title: Prerequisites
description: Required knowledge and tools before starting
---

# Prerequisites

Before starting the Agentic AI Training, ensure you have the following knowledge and tools.

## Required Knowledge

### Python Fundamentals

You should be comfortable with:

```python
# Variables and data types
name = "AI Agent"
temperature = 0.7
enabled = True

# Functions and classes
class Agent:
    def __init__(self, name: str):
        self.name = name

    def run(self, prompt: str) -> str:
        return f"Processing: {prompt}"

# List comprehensions and dictionaries
tools = [tool.name for tool in available_tools]
config = {"model": "gpt-4", "temperature": 0.7}
```

### Async Programming (Helpful)

Many AI frameworks use async patterns:

```python
import asyncio

async def process_query(query: str) -> str:
    result = await llm.ainvoke(query)
    return result

# Running async code
asyncio.run(process_query("Hello, AI!"))
```

### API Concepts

Understanding REST APIs and HTTP requests:

```python
import requests

response = requests.post(
    "https://api.openai.com/v1/chat/completions",
    headers={"Authorization": f"Bearer {api_key}"},
    json={"model": "gpt-4", "messages": messages}
)
```

## Required Tools

### Essential Software

| Tool | Version | Purpose |
|------|---------|---------|
| **Python** | 3.10+ | Primary programming language |
| **pip** | Latest | Package management |
| **Git** | 2.30+ | Version control |
| **VS Code** or **PyCharm** | Latest | Code editor |

### Verify Installation

Run these commands to verify your setup:

```bash
# Check Python version
python --version
# Expected: Python 3.10.x or higher

# Check pip
pip --version
# Expected: pip 23.x or higher

# Check git
git --version
# Expected: git version 2.30.x or higher
```

## Required Accounts

You'll need accounts with the following services:

### LLM Providers (At least one)

| Provider | Free Tier | Sign Up |
|----------|-----------|---------|
| **OpenAI** | $5 credit | [platform.openai.com](https://platform.openai.com) |
| **Google AI** | Free tier | [ai.google.dev](https://ai.google.dev) |
| **Anthropic** | Limited | [console.anthropic.com](https://console.anthropic.com) |
| **Groq** | Free tier | [console.groq.com](https://console.groq.com) |

:::tip Cost-Effective Learning
Start with **Groq** or **Google AI** for free-tier access during learning. Move to OpenAI for production use.
:::

### Optional Services

| Service | Purpose |
|---------|---------|
| **Tavily** | Web search tool for agents |
| **Pinecone** | Vector database |
| **LangSmith** | LangChain observability |

## Hardware Requirements

### Minimum Requirements

- **RAM**: 8 GB
- **Storage**: 10 GB free space
- **CPU**: Multi-core processor
- **Internet**: Stable connection

### Recommended

- **RAM**: 16 GB+
- **Storage**: SSD with 20 GB+ free
- **GPU**: Optional (for local models)

## Knowledge Assessment

Test your readiness with these quick checks:

### Python Basics
```python
# Can you explain what this does?
result = [x**2 for x in range(10) if x % 2 == 0]
```

<details>
<summary>Answer</summary>

Creates a list of squares of even numbers from 0-9: `[0, 4, 16, 36, 64]`

</details>

### API Understanding
```python
# What HTTP method would you use to create a new resource?
```

<details>
<summary>Answer</summary>

POST - Used to create new resources on a server

</details>

### Environment Variables
```python
# How would you securely access an API key in Python?
```

<details>
<summary>Answer</summary>

```python
import os
api_key = os.getenv("OPENAI_API_KEY")
```

</details>

## Self-Assessment Checklist

Before proceeding, confirm you can:

- [ ] Write and run Python scripts
- [ ] Use pip to install packages
- [ ] Create and activate virtual environments
- [ ] Understand basic API concepts
- [ ] Use Git for version control
- [ ] Access at least one LLM provider

:::info Need to Brush Up?
If you're missing any prerequisites, here are some resources:
- **Python**: [Python Tutorial](https://docs.python.org/3/tutorial/)
- **Git**: [Git Handbook](https://guides.github.com/introduction/git-handbook/)
- **APIs**: [REST API Tutorial](https://restfulapi.net/)
:::

## Next Steps

Once you've confirmed all prerequisites:

1. Proceed to [Python Setup](./python-setup) to configure your environment
2. Then [API Keys Setup](./api-keys-setup) for LLM access
