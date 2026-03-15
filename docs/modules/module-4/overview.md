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

### 4. Model Context Protocol (MCP) — Deep Dive

MCP is a **standard protocol** that defines how AI applications interact with external tools and data sources. Think of it as the USB standard for AI — any MCP-compatible client can connect to any MCP-compatible server.

> **The three words in the name**:
> - **Model** — the LLM doing the reasoning
> - **Context** — third-party data or situational awareness the model needs
> - **Protocol** — the standard communication rules

```
┌─────────────────────────────────────────────────────────────────┐
│                    MCP Architecture                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────────┐   JSON-RPC 2.0   ┌──────────────────┐   │
│   │   MCP Client     │◀───────────────▶ │   MCP Server     │   │
│   │   (Your Agent)   │                  │ (3rd party tool) │   │
│   └──────────────────┘                  └──────────────────┘   │
│                                                 │                │
│                                    ┌────────────┼────────────┐  │
│                                    ▼            ▼            ▼  │
│                               [Tools]     [Resources]  [Prompts] │
│                             (Actions)    (Data/Files) (Templates)│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

Communication happens via **JSON-RPC 2.0** — a lightweight remote procedure call protocol. The client sends a request like `{"method": "tools/call", "params": {...}}` and the server responds with the result.

### The Three Pillars of an MCP Server

| Pillar | What it is | Example |
|--------|-----------|---------|
| **Tools** (Actions) | Functions the agent can call | `search_flights()`, `send_email()` |
| **Resources** (Data) | Static or dynamic data the agent can read | User preferences, file contents |
| **Prompts** (Templates) | Predefined instructions for the LLM | "When booking flights, always confirm..." |

### Building an MCP Server with FastMCP

`FastMCP` is the easiest way to build MCP servers in Python:

```python
from mcp.server.fastmcp import FastMCP

# Create the server
mcp = FastMCP("flight-booking-server")

# Tool — something the agent can DO
@mcp.tool()
def search_flights(origin: str, destination: str, date: str) -> list[dict]:
    """Search for available flights between two cities."""
    return flight_api.search(origin, destination, date)

@mcp.tool()
def book_flight(flight_id: str, passenger_name: str) -> dict:
    """Book a specific flight for a passenger."""
    return flight_api.book(flight_id, passenger_name)

# Resource — data the agent can READ
@mcp.resource("user://{user_id}/preferences")
def get_user_preferences(user_id: str) -> dict:
    """Get travel preferences for a user."""
    return user_db.get_preferences(user_id)

# Prompt — instruction template
@mcp.prompt()
def booking_instructions() -> str:
    """Standard instructions for flight booking."""
    return "Always confirm the passenger's name spelling before booking. Check baggage policy."

# Run the server
if __name__ == "__main__":
    mcp.run()
```

### Local vs Remote MCP Servers

| | Local (stdio) | Remote (HTTP/SSE) |
|--|--------------|------------------|
| **Transport** | stdin/stdout | HTTP + Server-Sent Events |
| **Use case** | Desktop apps, Claude Desktop | Web apps, multi-user, APIs |
| **Latency** | Lowest | Network dependent |
| **Scaling** | Single user | Multiple clients |
| **Example** | File system access tool | SaaS integration |

### MCP Client Features — What the Client Controls

The MCP Client (your agent) also has capabilities it exposes to servers:

| Client Feature | What it enables |
|----------------|----------------|
| **Context** | Server can send debug/status messages back to the client during execution |
| **Roots** | Client defines "safe" folders the server is allowed to access (security boundary) |
| **Sampling** | Server can ask the client to make an LLM call on its behalf — keeps server decoupled from LLM |
| **Elicitation** | Server can ask the end-user for more information through the client |

```python
# Sampling example — server asks client to do LLM inference
# This means the MCP server doesn't need to know which LLM you're using
# The server just says "please generate X" and the client uses its own LLM

# Roots example — client restricts file access
client = MCPClient(
    roots=["/home/user/documents", "/home/user/projects"]
    # Server cannot access /etc/passwd or any path outside roots
)
```

> **The big picture**: MCP lets developers publish capabilities (servers) that any MCP-compatible AI client can discover and use — without custom integrations. It's an emerging standard that's quickly being adopted across the AI ecosystem.

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

## Human-in-the-Loop (HITL) — Full Playbook

HITL is one of the most powerful and underused features of LangGraph. It lets you pause any graph at any point, wait for human input (seconds or hours), and resume with full state preserved.

> **The core idea**: Automated systems make mistakes. HITL creates "review gates" where humans can catch errors before they have real-world consequences — like sending an email, executing a trade, or publishing content.

### Pattern 1: Approve or Reject

The simplest pattern — pause before a critical node, let a human approve or stop it.

```python
# Compile with interrupt BEFORE the dangerous node
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["send_email"]   # pause before this node runs
)

# Run until interrupt
config = {"configurable": {"thread_id": "task-001"}}
result = app.invoke({"task": "Send performance review to John"}, config)
# Graph pauses. Human sees the draft email.

# Approve: resume (None = continue with saved state)
app.invoke(None, config)

# Reject: update state and stop
app.update_state(config, {"abort": True})
```

### Pattern 2: Review and Edit State

More powerful — a human can *modify* the state before the graph continues. Useful for correcting LLM mistakes.

```python
# Get current state
current_state = app.get_state(config)
print(current_state.values["draft"])  # See what the LLM generated

# Human edits the draft
corrected_draft = "Actually, the quarterly target was $2.4M, not $2.1M"

# Update the state with correction
app.update_state(config, {"draft": corrected_draft})

# Resume with corrected state
app.invoke(None, config)
```

### Pattern 3: Feedback Loop

A real-world example — AI writes a LinkedIn post, human gives feedback, AI revises:

```
User: "Write me a LinkedIn post about our product launch"
                     │
                     ▼
         LLM generates initial draft
                     │
                     ▼
         [INTERRUPT] → Human reviews draft
                     │
            ┌────────┴────────┐
            │ Feedback given  │ No feedback (looks good)
            ▼                 ▼
     LLM summarizes    Continue to publish
     feedback & revises
            │
            └──── [INTERRUPT again] → Human reviews revision
```

```python
def human_feedback_node(state):
    # interrupt() pauses here and returns control to the caller
    feedback = interrupt("Please review the draft and provide feedback:")
    return {"feedback": feedback, "iteration": state["iteration"] + 1}

def revise_node(state):
    if state.get("feedback"):
        revised = revision_chain.invoke({
            "draft": state["draft"],
            "feedback": state["feedback"]
        })
        return {"draft": revised}
    return {}
```

### Pattern 4: Review Tool Calls — Stop Expensive Tools Before They Run

Agents can call expensive, irreversible tools (API calls, database writes, emails). HITL lets humans review the *proposed tool call* before it actually executes.

```python
# Interrupt AFTER agent decides what to do, BEFORE the tool runs
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"]  # pause before any tool executes
)

# Flow:
# 1. User asks: "Who is older — the US President or Narendra Modi?
#               Take the older man's age and find its square root."
# 2. LLM suggests: search("age of US president"), search("Narendra Modi age")
# [INTERRUPT] → Human sees the proposed searches
# 3. Human approves (or modifies the search query)
# 4. Graph resumes: tools execute, LLM gets results, answers question

# Resume from last checkpoint
app.invoke(None, config)

# Can also interrupt AFTER tools run (to review results before LLM sees them)
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_after=["tools"]
)
```

:::tip interrupt_before vs interrupt_after
- `interrupt_before=["tools"]` → human reviews **what the agent wants to do** before it does it
- `interrupt_after=["tools"]` → human reviews **what the tool returned** before the agent uses it
- You can use both simultaneously for maximum control
:::

### The `interrupt()` Function

`interrupt()` is the modern way to pause a node mid-execution:

```python
from langgraph.types import interrupt, Command

def review_node(state):
    # Pauses here, saves state, returns control to caller
    # Works in web apps, APIs, async environments
    # Survives server restarts (state is in checkpointer)
    human_input = interrupt("Waiting for human review...")
    return {"review_result": human_input}
```

### The Command Class — Edgeless Workflows

Instead of hardcoding edges with `add_edge()`, the `Command` class lets each node decide *at runtime* where to go next. This makes complex routing logic readable and flexible.

```python
from langgraph.types import Command

def triage_node(state) -> Command:
    urgency = classify_urgency(state["issue"])

    if urgency == "critical":
        return Command(
            goto="escalate_node",
            update={"priority": "P0", "escalated": True}
        )
    elif urgency == "normal":
        return Command(
            goto="standard_queue_node",
            update={"priority": "P2"}
        )
    else:
        return Command(goto="auto_resolve_node")

# No add_edge() needed! The routing logic lives inside the node.
```

| `Command` field | Purpose |
|-----------------|---------|
| `goto` | Next node to transition to |
| `update` | State changes to apply before moving |

**Why use Command over edges?**
- Routing logic stays with the node that makes the decision
- More readable when logic is complex
- Easier to add conditions without restructuring the graph

## Summary

| Concept | Purpose |
|---------|---------|
| Tools | Agent capabilities |
| ReAct | Reasoning pattern |
| Google ADK | Agent framework |
| MCP | External connectivity |
| Multi-Agent | Collaboration |
| HITL | Human review gates |
| Command | Edgeless routing |

---

## Test Your Understanding

---

> **Q1: You use `interrupt_before=["tools"]`. The agent decides to call `delete_user_account(user_id="123")`. What happens next?**
>
> <details>
> <summary>Show Answer</summary>
>
> The graph **pauses** before the tools node executes. The current state (including the agent's tool call suggestion) is saved to the checkpointer. Control returns to your application. A human can see the proposed tool call — `delete_user_account(user_id="123")` — and decide to approve (resume with `app.invoke(None, config)`) or abort. The tool has NOT executed yet. This is exactly why `interrupt_before` is the right choice for irreversible actions.
> </details>

---

> **Q2: What's the key benefit of using `Command` over `add_conditional_edges()` for routing?**
>
> <details>
> <summary>Show Answer</summary>
>
> **The routing logic stays with the node that makes the decision**. With `add_conditional_edges`, you define routing separately from the nodes — the graph structure becomes disconnected from the logic. With `Command`, the node returns `Command(goto="next_node")`, so reading the node code tells you exactly where it can go. For complex graphs with many routing possibilities, `Command` is much easier to maintain.
> </details>

---

> **Q3: In MCP, what does "Sampling" allow the server to do — and why is this valuable?**
>
> <details>
> <summary>Show Answer</summary>
>
> **Sampling** lets the MCP server ask the client to perform an LLM call on its behalf. Instead of the server needing to know which LLM you're using or managing its own API keys, it says "please generate X for me" and the client uses its own LLM. This keeps the server **decoupled from the LLM** — the same MCP server works with Claude, GPT-4, or any other model the client is using.
> </details>

---

> **Q4: Your agent is in a feedback loop: write draft → human reviews → revise → human reviews → revise → ... How do you prevent this from looping forever?**
>
> <details>
> <summary>Show Answer</summary>
>
> Track an **iteration counter** in state and add a conditional edge that routes to END when `iteration >= max_iterations`. Real systems also give humans the option to explicitly "approve" (skip further iteration) at each review step. Always design HITL feedback loops with an explicit exit condition — otherwise they depend entirely on humans providing the magic "looks good" signal.
> </details>

---

## Next Steps

Proceed to [Hands-on exercises](./hands-on) to build agents with tools.
