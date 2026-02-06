---
sidebar_position: 4
title: "Resources"
description: Module 4 additional resources and references
---

# Module 4 Resources

Additional learning materials for agents, tools, and MCP.

## Official Documentation

| Resource | Description |
|----------|-------------|
| [LangChain Agents](https://python.langchain.com/docs/concepts/#agents) | Agent concepts |
| [Tool Calling](https://python.langchain.com/docs/how_to/tool_calling/) | How to use tools |
| [MCP Specification](https://modelcontextprotocol.io/) | Model Context Protocol |
| [Google ADK](https://cloud.google.com/agents) | Google Agent Development Kit |

## Module Code Files

Available in `module/Module 4/`:

```
Module 4/
├── Agent with Tools/
│   ├── google-adk/
│   │   ├── 1_basic/           # Basic agent
│   │   ├── 2_tool/            # With tools
│   │   ├── 3_structured_output/  # Structured responses
│   │   └── 4_multi_agent/     # Multi-agent patterns
│   │       ├── loop_agent/
│   │       ├── runner_agent/
│   │       ├── sequential_agent/
│   │       └── tool_agent/
│   └── langchain/
│       └── 10_langchain_agents/
├── MCP/
│   ├── googe-adk/
│   │   ├── local_mcp/         # Local MCP server
│   │   └── mcp_comparison/    # MCP vs traditional
│   └── langchain/
│       ├── main.py
│       └── multi_agent_demo.py
└── Human in the loop - GIT Link.txt
```

## Tool Development Patterns

### Retry Pattern

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@tool
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def reliable_api_call(endpoint: str) -> str:
    """API call with automatic retry."""
    response = requests.get(endpoint)
    response.raise_for_status()
    return response.text
```

### Caching Pattern

```python
from functools import lru_cache

@tool
def expensive_computation(input: str) -> str:
    """Computation with caching."""
    return _cached_compute(input)

@lru_cache(maxsize=100)
def _cached_compute(input: str) -> str:
    # Expensive operation
    return result
```

### Validation Pattern

```python
from pydantic import BaseModel, validator

class EmailInput(BaseModel):
    to: str
    subject: str
    body: str

    @validator('to')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email address')
        return v

@tool(args_schema=EmailInput)
def send_email(to: str, subject: str, body: str) -> str:
    """Send email with validation."""
    pass
```

## MCP Reference

### Server Setup

```python
from mcp import Server, Resource, Tool

server = Server("my-server", version="1.0.0")

# Tools
@server.tool
def my_tool(param: str) -> str:
    pass

# Resources
@server.resource("data/{id}")
def get_data(id: str) -> dict:
    pass

# Prompts
@server.prompt("greeting")
def greeting_prompt() -> str:
    return "Hello! How can I help you today?"
```

### Client Usage

```python
from mcp import Client

async def main():
    async with Client("http://localhost:8000") as client:
        # List capabilities
        tools = await client.list_tools()
        resources = await client.list_resources()

        # Use tool
        result = await client.call_tool("my_tool", {"param": "value"})

        # Read resource
        data = await client.read_resource("data/123")
```

## Multi-Agent Patterns

### Supervisor Pattern

```python
def supervisor(state):
    """Route to appropriate specialist."""
    task = state["task"]

    if "code" in task.lower():
        return "coder"
    elif "research" in task.lower():
        return "researcher"
    else:
        return "generalist"
```

### Consensus Pattern

```python
def consensus(responses: list[str]) -> str:
    """Combine multiple agent responses."""
    prompt = f"""
    Multiple agents provided these responses:
    {chr(10).join(responses)}

    Synthesize into a single coherent response.
    """
    return llm.invoke(prompt).content
```

## External References

| Resource | Topic |
|----------|-------|
| [Anthropic Tool Use](https://docs.anthropic.com/claude/docs/tool-use) | Claude tool calling |
| [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling) | GPT functions |
| [Human-in-the-Loop](https://github.com/) | Human oversight patterns |

## Practice Projects

### 1. Research Assistant

Build an agent that:
- Searches multiple sources
- Summarizes findings
- Cites sources

### 2. Code Assistant

Create an agent with:
- File reading tool
- Code execution tool
- Documentation lookup

### 3. Customer Service Bot

Implement:
- FAQ lookup
- Ticket creation
- Human escalation

## Next Module

Continue to [Module 5: Production Patterns](/learn/modules/module-5) to learn:
- Deploying agents
- Monitoring and logging
- Testing strategies
- Security best practices
