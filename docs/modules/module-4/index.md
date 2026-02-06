---
sidebar_position: 1
title: "Module 4: Agents, Tools & MCP"
description: Build intelligent agents with tools and Model Context Protocol
---

# Module 4: Agents, Tools & MCP

Welcome to Module 4! This module covers building sophisticated AI agents with custom tools, Google ADK, and the Model Context Protocol (MCP).

## Learning Objectives

By the end of this module, you will be able to:

- Create custom tools for AI agents
- Build agents with Google's Agent Development Kit (ADK)
- Implement the Model Context Protocol (MCP)
- Design multi-agent systems
- Handle human-in-the-loop workflows

## Module Contents

| Section | Duration | Description |
|---------|----------|-------------|
| [Overview](./overview) | 30 min | Theory and architecture |
| [Hands-on](./hands-on) | 60 min | Building agents with tools |
| [Resources](./resources) | - | Additional learning |

## Topics Covered

### 1. Custom Tools
Create tools that agents can use to interact with the world.

```python
from langchain_core.tools import tool

@tool
def search_database(query: str) -> str:
    """Search the internal database for information."""
    # Implementation
    return results
```

### 2. Google Agent Development Kit (ADK)
Build agents using Google's framework.

```python
from google_adk import Agent

agent = Agent(
    name="assistant",
    model="gemini-1.5-pro",
    tools=[search_tool, calculate_tool]
)
```

### 3. Model Context Protocol (MCP)
Connect agents to external services and data sources.

```python
from mcp import Server, Tool

server = Server("my-mcp-server")

@server.tool
def fetch_data(source: str) -> dict:
    """Fetch data from external source."""
    return data
```

### 4. Multi-Agent Systems
Coordinate multiple agents working together.

```
┌─────────────────────────────────────────────────────────┐
│                  Multi-Agent System                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│    ┌─────────┐    ┌─────────┐    ┌─────────┐           │
│    │Researcher│    │ Writer  │    │Reviewer │           │
│    │  Agent   │───▶│  Agent  │───▶│  Agent  │           │
│    └─────────┘    └─────────┘    └─────────┘           │
│         │                              │                 │
│         └──────────────────────────────┘                │
│                    Feedback Loop                         │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Hands-on Labs

This module includes:

| Directory | Content |
|-----------|---------|
| `google-adk/` | Google ADK examples |
| `langchain/` | LangChain agents |
| `googe-adk/local_mcp/` | Local MCP server |
| `googe-adk/mcp_comparison/` | MCP vs traditional agents |

## Prerequisites

Before starting this module:

- [ ] Completed Modules 1-3
- [ ] Understanding of LangGraph
- [ ] API keys configured

## Key Takeaways

:::note What You'll Build
By the end of this module, you'll have built agents that can:
- Use multiple tools dynamically
- Connect to external services via MCP
- Collaborate with other agents
- Handle complex multi-step tasks
:::

## Next Steps

1. Complete the [Overview](./overview)
2. Work through [Hands-on exercises](./hands-on)
3. Check [Resources](./resources)
4. Move to [Module 5: Production Patterns](/learn/modules/module-5)

---

**Estimated Time**: 90 minutes

**Difficulty**: Advanced
