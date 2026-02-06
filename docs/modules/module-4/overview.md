---
sidebar_position: 2
title: "Overview"
description: Module 4 theory and concepts
---

# Module 4 Overview

Understanding agents, tools, and the Model Context Protocol.

## What is an AI Agent?

An AI agent is an LLM that can:
1. **Reason** about tasks
2. **Decide** which actions to take
3. **Execute** actions using tools
4. **Learn** from results

```
┌─────────────────────────────────────────────────────────────────┐
│                        Agent Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │                        LLM Core                           │  │
│   │  (Reasoning, Planning, Decision Making)                   │  │
│   └──────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │
│   │ Search  │  │  Code   │  │Database │  │   API   │          │
│   │  Tool   │  │  Tool   │  │  Tool   │  │  Tool   │          │
│   └─────────┘  └─────────┘  └─────────┘  └─────────┘          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Core Concepts

### 1. Tools

Tools are functions that agents can call to interact with the world.

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="The search query")
    max_results: int = Field(default=5, description="Maximum results")

@tool(args_schema=SearchInput)
def search_web(query: str, max_results: int = 5) -> str:
    """Search the web for information."""
    # Implementation
    return results
```

**Tool Types:**

| Type | Use Case | Example |
|------|----------|---------|
| Search | Information retrieval | Web search, DB query |
| Calculation | Math and logic | Calculator, code execution |
| API | External services | Weather, stocks, email |
| File | Document handling | Read, write, parse |

### 2. ReAct Pattern

ReAct (Reasoning + Acting) is the most common agent pattern:

```
┌─────────────────────────────────────────────────────────────────┐
│                         ReAct Loop                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Question: "What's the weather in Paris and should I bring     │
│              an umbrella?"                                       │
│                                                                  │
│   ┌───────────────────────────────────────────────────────────┐ │
│   │ Thought: I need to get the current weather in Paris       │ │
│   │ Action: get_weather("Paris")                              │ │
│   │ Observation: Sunny, 72°F, no rain expected                │ │
│   └───────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│   ┌───────────────────────────────────────────────────────────┐ │
│   │ Thought: Weather is sunny with no rain                    │ │
│   │ Action: None needed                                       │ │
│   │ Answer: The weather in Paris is sunny at 72°F.            │ │
│   │         You don't need an umbrella today.                 │ │
│   └───────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Google Agent Development Kit (ADK)

Google's ADK provides a structured way to build agents:

```python
from google_adk import Agent, Tool

# Define tools
@Tool
def greet(name: str) -> str:
    """Greet a user by name."""
    return f"Hello, {name}!"

# Create agent
agent = Agent(
    name="greeter",
    model="gemini-1.5-pro",
    instructions="You are a friendly greeter.",
    tools=[greet]
)

# Run agent
response = agent.run("Say hello to Alice")
```

**ADK Features:**

| Feature | Description |
|---------|-------------|
| Structured Output | Define response schemas |
| Tool Chaining | Sequential tool execution |
| Multi-Agent | Agent collaboration |
| Memory | Conversation persistence |

### 4. Model Context Protocol (MCP)

MCP standardizes how AI models interact with external systems:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Model Context Protocol                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐         ┌──────────────┐                     │
│   │   AI Model   │◀───────▶│  MCP Server  │                     │
│   │   (Client)   │         │              │                     │
│   └──────────────┘         └──────────────┘                     │
│                                   │                              │
│                                   ▼                              │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    External Resources                     │   │
│   │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │   │
│   │  │ Database│  │  APIs   │  │  Files  │  │ Services│    │   │
│   │  └─────────┘  └─────────┘  └─────────┘  └─────────┘    │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**MCP Components:**

```python
from mcp import Server, Tool, Resource

# Create MCP server
server = Server("my-server")

# Define a tool
@server.tool
def query_database(sql: str) -> list:
    """Execute SQL query on database."""
    return db.execute(sql)

# Define a resource
@server.resource("users/{user_id}")
def get_user(user_id: str) -> dict:
    """Get user information."""
    return users.get(user_id)
```

### 5. Multi-Agent Systems

Multiple agents working together:

```python
# Supervisor agent delegates to specialists
class SupervisorAgent:
    def __init__(self):
        self.researcher = ResearchAgent()
        self.writer = WriterAgent()
        self.reviewer = ReviewAgent()

    def process_task(self, task):
        # 1. Research
        research = self.researcher.research(task)

        # 2. Write based on research
        draft = self.writer.write(research)

        # 3. Review and improve
        final = self.reviewer.review(draft)

        return final
```

**Patterns:**

| Pattern | Description |
|---------|-------------|
| Supervisor | One agent coordinates others |
| Sequential | Agents in pipeline |
| Parallel | Agents work simultaneously |
| Debate | Agents argue and reach consensus |

## Agent Design Best Practices

### 1. Clear Tool Descriptions

```python
# Good: Detailed description
@tool
def search_products(
    query: str,
    category: str | None = None,
    max_price: float | None = None
) -> list[dict]:
    """
    Search for products in the catalog.

    Args:
        query: Search terms (e.g., "red shoes")
        category: Filter by category (e.g., "footwear")
        max_price: Maximum price in USD

    Returns:
        List of matching products with name, price, and URL
    """
    pass

# Bad: Vague description
@tool
def search(q: str) -> list:
    """Search for stuff."""
    pass
```

### 2. Error Handling

```python
@tool
def safe_api_call(endpoint: str) -> str:
    """Call external API safely."""
    try:
        response = requests.get(endpoint, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.Timeout:
        return "Error: Request timed out"
    except requests.RequestException as e:
        return f"Error: {str(e)}"
```

### 3. Tool Composition

```python
# Break complex operations into composable tools
@tool
def fetch_data(source: str) -> dict:
    """Fetch raw data from source."""
    pass

@tool
def transform_data(data: dict, format: str) -> dict:
    """Transform data to specified format."""
    pass

@tool
def validate_data(data: dict) -> bool:
    """Validate data against schema."""
    pass

# Agent can compose: fetch → transform → validate
```

## Human-in-the-Loop

Pause agent execution for human approval:

```python
from langgraph.checkpoint.memory import MemorySaver

# Compile with interrupt
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["execute_dangerous_action"]
)

# Run until interrupt
result = app.invoke(input, config)

# Human reviews...

# Continue or abort
if approved:
    app.invoke(None, config)  # Continue
else:
    app.invoke({"abort": True}, config)  # Abort
```

## Summary

| Concept | Purpose |
|---------|---------|
| Tools | Agent capabilities |
| ReAct | Reasoning pattern |
| Google ADK | Agent framework |
| MCP | External connectivity |
| Multi-Agent | Collaboration |

## Next Steps

Proceed to [Hands-on exercises](./hands-on) to build agents with tools.
