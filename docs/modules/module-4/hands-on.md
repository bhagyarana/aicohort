---
sidebar_position: 3
title: "Hands-on"
description: Module 4 practical exercises
---

# Module 4 Hands-on Exercises

Build agents with tools, MCP, and multi-agent systems.

## Exercise 1: Creating Custom Tools

Build reusable tools for your agents.

### Basic Tool

```python
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.

    Args:
        expression: A valid Python math expression (e.g., "2 + 2", "10 * 5")

    Returns:
        The result of the calculation
    """
    try:
        # Safe evaluation
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"

        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"
```

### Structured Input Tool

```python
from pydantic import BaseModel, Field
from typing import Optional

class SearchParams(BaseModel):
    query: str = Field(description="Search query")
    limit: int = Field(default=5, description="Max results")
    category: Optional[str] = Field(default=None, description="Filter category")

@tool(args_schema=SearchParams)
def search_products(query: str, limit: int = 5, category: Optional[str] = None) -> str:
    """
    Search for products in the catalog.
    """
    # Mock implementation
    products = [
        {"name": "Widget A", "price": 29.99, "category": "electronics"},
        {"name": "Widget B", "price": 39.99, "category": "electronics"},
        {"name": "Gadget X", "price": 49.99, "category": "gadgets"},
    ]

    results = [p for p in products if query.lower() in p["name"].lower()]

    if category:
        results = [p for p in results if p["category"] == category]

    return str(results[:limit])
```

### Async Tool

```python
import asyncio
import aiohttp

@tool
async def fetch_url(url: str) -> str:
    """Fetch content from a URL asynchronously."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()
```

---

## Exercise 2: LangChain Agent with Tools

Build a complete agent using LangChain.

### Setup Tools

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # Mock response
    weather_data = {
        "Paris": "Sunny, 22°C",
        "London": "Cloudy, 15°C",
        "Tokyo": "Rainy, 18°C",
    }
    return weather_data.get(city, f"Weather data not available for {city}")

@tool
def get_time(timezone: str) -> str:
    """Get current time in a timezone."""
    from datetime import datetime
    import pytz

    try:
        tz = pytz.timezone(timezone)
        return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    except:
        return f"Unknown timezone: {timezone}"

tools = [get_weather, get_time, calculator]
```

### Create Agent

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant with access to tools.
    Use the tools to help answer user questions.
    Always explain your reasoning."""),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# Create the agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

### Run Agent

```python
# Test the agent
result = agent_executor.invoke({
    "input": "What's the weather in Paris and what time is it there?",
    "chat_history": []
})

print(result["output"])
```

---

## Exercise 3: Google ADK Agent

Build an agent using Google's Agent Development Kit.

### Basic ADK Agent

```python
# Note: This requires google-adk package
from google_adk import Agent

# Define the agent
agent = Agent(
    name="assistant",
    model="gemini-1.5-pro",
    instructions="""You are a helpful assistant.
    Be concise and friendly in your responses."""
)

# Simple interaction
response = agent.run("Hello! What can you help me with?")
print(response)
```

### ADK Agent with Tools

```python
from google_adk import Agent, Tool

@Tool
def greet_user(name: str) -> str:
    """Greet a user by their name."""
    return f"Hello, {name}! Welcome to our service."

@Tool
def calculate_bmi(weight_kg: float, height_m: float) -> str:
    """Calculate Body Mass Index."""
    bmi = weight_kg / (height_m ** 2)
    category = (
        "Underweight" if bmi < 18.5 else
        "Normal" if bmi < 25 else
        "Overweight" if bmi < 30 else
        "Obese"
    )
    return f"BMI: {bmi:.1f} ({category})"

# Create agent with tools
agent = Agent(
    name="health_assistant",
    model="gemini-1.5-pro",
    instructions="You are a health assistant. Use tools to help users.",
    tools=[greet_user, calculate_bmi]
)

# Test
response = agent.run("Calculate BMI for someone who is 70kg and 1.75m tall")
```

### ADK Multi-Agent

```python
from google_adk import Agent, Runner

# Define specialist agents
researcher = Agent(
    name="researcher",
    instructions="You research topics and provide factual information."
)

writer = Agent(
    name="writer",
    instructions="You write clear, engaging content based on research."
)

# Create runner for multi-agent
runner = Runner(agents=[researcher, writer])

# Sequential execution
result = runner.run("""
    Research the benefits of exercise, then write a short blog post about it.
""")
```

---

## Exercise 4: MCP Server Implementation

Create an MCP server to expose tools.

### Basic MCP Server

```python
from mcp import Server

# Create server
server = Server("my-tools-server")

@server.tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

@server.tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

@server.tool
def query_database(table: str, filters: dict) -> list:
    """Query the database with filters."""
    # Mock implementation
    return [{"id": 1, "name": "Sample", "table": table}]

# Run the server
if __name__ == "__main__":
    server.run()
```

### MCP Agent Client

```python
from mcp import Client

# Connect to MCP server
client = Client("http://localhost:8000")

# List available tools
tools = client.list_tools()
print(f"Available tools: {[t.name for t in tools]}")

# Call a tool
result = client.call_tool("add_numbers", {"a": 5, "b": 3})
print(f"Result: {result}")
```

### FastMCP Server

```python
from fastmcp import FastMCP

mcp = FastMCP("Data Service")

@mcp.tool()
def get_user(user_id: int) -> dict:
    """Get user information by ID."""
    users = {
        1: {"name": "Alice", "email": "alice@example.com"},
        2: {"name": "Bob", "email": "bob@example.com"},
    }
    return users.get(user_id, {"error": "User not found"})

@mcp.resource("config://settings")
def get_settings() -> dict:
    """Get application settings."""
    return {
        "theme": "dark",
        "language": "en",
        "notifications": True
    }

if __name__ == "__main__":
    mcp.run()
```

---

## Exercise 5: Multi-Agent System

Build a collaborative multi-agent system.

### Define Agents

```python
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, END

class MultiAgentState(TypedDict):
    task: str
    research: str
    draft: str
    review: str
    final: str

llm = ChatOpenAI(model="gpt-4o-mini")

def researcher(state: MultiAgentState) -> dict:
    """Research agent gathers information."""
    prompt = f"Research this topic and provide key facts: {state['task']}"
    response = llm.invoke(prompt)
    return {"research": response.content}

def writer(state: MultiAgentState) -> dict:
    """Writer agent creates content."""
    prompt = f"""Based on this research:
    {state['research']}

    Write a short article about: {state['task']}"""
    response = llm.invoke(prompt)
    return {"draft": response.content}

def reviewer(state: MultiAgentState) -> dict:
    """Reviewer agent improves content."""
    prompt = f"""Review and improve this draft:
    {state['draft']}

    Provide the improved version."""
    response = llm.invoke(prompt)
    return {"final": response.content}
```

### Build Multi-Agent Graph

```python
# Create the graph
graph = StateGraph(MultiAgentState)

# Add agent nodes
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)
graph.add_node("reviewer", reviewer)

# Define flow
graph.set_entry_point("researcher")
graph.add_edge("researcher", "writer")
graph.add_edge("writer", "reviewer")
graph.add_edge("reviewer", END)

# Compile
multi_agent = graph.compile()

# Run
result = multi_agent.invoke({
    "task": "The benefits of AI in healthcare",
    "research": "",
    "draft": "",
    "review": "",
    "final": ""
})

print("Final Article:")
print(result["final"])
```

---

## Checklist

Before moving to Module 5:

- [ ] Create custom tools with proper schemas
- [ ] Build LangChain agents with tools
- [ ] Understand Google ADK patterns
- [ ] Implement MCP servers
- [ ] Design multi-agent workflows

## Next Steps

1. Review [Resources](./resources)
2. Continue to [Module 5: Production Patterns](/learn/modules/module-5)
