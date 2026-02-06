---
sidebar_position: 1
title: "Pattern 2: Tool-Using Agent"
description: Build agents with custom tools
---

# Pattern 2: Tool-Using Agent

Learn to create sophisticated agents with custom tools that integrate with external services and APIs.

## Overview

### What You'll Build

A tool-using agent that can:
- Call multiple custom tools
- Handle structured inputs and outputs
- Integrate with external APIs
- Chain tool calls together

```
┌─────────────────────────────────────────────────────────────────┐
│                    Tool-Using Agent                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │                      LLM Core                             │  │
│   │           (Function Calling / Tool Use)                   │  │
│   └──────────────────────────────────────────────────────────┘  │
│                              │                                   │
│          ┌───────────────────┼───────────────────┐              │
│          │                   │                   │              │
│          ▼                   ▼                   ▼              │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐      │
│   │   Search    │     │  Database   │     │    API      │      │
│   │    Tool     │     │    Tool     │     │    Tool     │      │
│   └─────────────┘     └─────────────┘     └─────────────┘      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Learning Objectives

- Create tools with complex input schemas
- Implement API integrations
- Handle asynchronous tool execution
- Build tool chains

## Key Concepts

### Tool Categories

| Category | Examples | Use Case |
|----------|----------|----------|
| **Information** | Search, lookup, query | Getting data |
| **Computation** | Calculate, analyze, process | Data processing |
| **Communication** | Email, notify, message | Outbound actions |
| **Integration** | API calls, webhooks | External systems |

### Structured Tool Inputs

```python
from pydantic import BaseModel, Field

class EmailParams(BaseModel):
    to: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email body content")
    priority: str = Field(default="normal", description="Priority: low, normal, high")

@tool(args_schema=EmailParams)
def send_email(to: str, subject: str, body: str, priority: str = "normal") -> str:
    """Send an email to a recipient."""
    pass
```

## Implementation Preview

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Multiple specialized tools
@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    pass

@tool
def query_database(sql: str) -> str:
    """Execute SQL query on database."""
    pass

@tool
def send_notification(user: str, message: str) -> str:
    """Send notification to user."""
    pass

# Create tool-calling agent
llm = ChatOpenAI(model="gpt-4o-mini")
tools = [search_web, query_database, send_notification]

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)
```

## Next Steps

Continue to [Hands-on](./hands-on) to implement your tool-using agent.
