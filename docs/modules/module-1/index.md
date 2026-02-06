---
sidebar_position: 1
title: "Module 1: LangChain Fundamentals"
description: Master the core concepts of LangChain
---

# Module 1: LangChain Fundamentals

Welcome to Module 1! This module covers the foundational concepts of LangChain that you'll use throughout the training.

## Learning Objectives

By the end of this module, you will be able to:

- Connect to various LLM providers (OpenAI, Groq, Google AI)
- Create and use prompt templates effectively
- Understand and use LangChain Expression Language (LCEL)
- Parse and structure LLM outputs
- Build simple chains for common use cases

## Module Contents

| Section | Duration | Description |
|---------|----------|-------------|
| [Overview](./overview) | 30 min | Theory and core concepts |
| [Hands-on](./hands-on) | 60 min | Practical exercises |
| [Resources](./resources) | - | Links and references |

## Topics Covered

### 1. LangChain with OpenAI
Set up your first LangChain application with OpenAI's models.

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
response = llm.invoke("What is Agentic AI?")
```

### 2. Prompt Templates
Create reusable, parameterized prompts.

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant specializing in {topic}."),
    ("human", "{question}")
])
```

### 3. LangChain Expression Language (LCEL)
Build composable chains with the pipe operator.

```python
chain = prompt | llm | output_parser
result = chain.invoke({"topic": "AI", "question": "What are agents?"})
```

### 4. Output Parsers
Structure LLM responses into usable formats.

```python
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()
structured_chain = prompt | llm | parser
```

## Prerequisites

Before starting this module, ensure you have:

- [ ] Completed the [Onboarding & Setup](/learn/onboarding)
- [ ] Python environment configured
- [ ] At least one LLM API key (OpenAI or Groq recommended)

## Hands-on Labs

This module includes the following Jupyter notebooks:

| Notebook | Description |
|----------|-------------|
| `1_langchain_openai.ipynb` | Basic LangChain with OpenAI |
| `2_prompt_templates.ipynb` | Creating effective prompts |
| `3_lcel.ipynb` | LangChain Expression Language |
| `4_output_parser.ipynb` | Parsing and structuring outputs |

## Key Takeaways

After completing this module, you'll understand:

:::note Core Concepts
1. **LLMs are the Foundation** - All agent systems build on LLM capabilities
2. **Prompts are Powerful** - Well-crafted prompts dramatically improve results
3. **LCEL Enables Composition** - Chain components for complex workflows
4. **Structure Matters** - Output parsing creates reliable applications
:::

## Next Steps

Once you've completed Module 1:

1. Review the [hands-on exercises](./hands-on)
2. Check additional [resources](./resources)
3. Move on to [Module 2: Advanced LangChain](/learn/modules/module-2)

---

**Estimated Time**: 90 minutes

**Difficulty**: Beginner
