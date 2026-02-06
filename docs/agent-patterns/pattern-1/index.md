---
sidebar_position: 1
title: "Pattern 1: ReAct Agent"
description: Build a reasoning and acting agent
---

# Pattern 1: ReAct Agent

The ReAct (Reasoning + Acting) pattern is the foundation of agentic AI. Learn to build agents that think through problems systematically.

## Overview

### What is ReAct?

ReAct combines **reasoning** (thinking about what to do) with **acting** (taking actions) in an iterative loop:

```
┌─────────────────────────────────────────────────────────────────┐
│                      ReAct Loop                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │                    USER QUESTION                          │  │
│   └──────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │  THOUGHT: "I need to find out X to answer this"          │  │
│   └──────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │  ACTION: search("X")                                      │  │
│   └──────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │  OBSERVATION: "Found information about X..."              │  │
│   └──────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                    ┌─────────┴─────────┐                        │
│                    │   Need more info? │                        │
│                    └─────────┬─────────┘                        │
│                    Yes       │        No                         │
│                     ▲        │         │                         │
│                     │        │         ▼                         │
│                     └────────┘   ┌─────────────┐                │
│                                  │   ANSWER    │                │
│                                  └─────────────┘                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Learning Objectives

- Understand the ReAct reasoning pattern
- Implement a basic ReAct agent
- Handle the thought-action-observation loop
- Debug and improve agent reasoning

## Key Concepts

### The Three Components

| Component | Purpose | Example |
|-----------|---------|---------|
| **Thought** | Reasoning about what to do | "I need to look up the weather" |
| **Action** | Executing a specific action | `get_weather("Paris")` |
| **Observation** | Result of the action | "Paris: Sunny, 22°C" |

### Why ReAct Works

1. **Explicit Reasoning**: Forces the model to explain its thinking
2. **Grounded Actions**: Actions are based on reasoning
3. **Iterative Refinement**: Can adjust based on observations
4. **Interpretable**: Easy to understand agent behavior

## Implementation Preview

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

# ReAct prompt template
react_prompt = PromptTemplate.from_template("""
Answer the question using the available tools.

Question: {input}

You have access to these tools:
{tools}

Use this format:

Thought: [Your reasoning about what to do]
Action: [The action to take]
Action Input: [The input for the action]
Observation: [The result of the action]
... (repeat Thought/Action/Observation as needed)
Thought: I now know the final answer
Final Answer: [Your answer to the question]

{agent_scratchpad}
""")

# Create ReAct agent
llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_react_agent(llm, tools, react_prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run
result = executor.invoke({"input": "What's the weather in Paris?"})
```

## Next Steps

Continue to [Hands-on](./hands-on) to build your ReAct agent.
