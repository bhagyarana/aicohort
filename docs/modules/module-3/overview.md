---
sidebar_position: 2
title: "Overview"
description: Module 3 theory and concepts
---

# Module 3 Overview

Understanding the architecture and concepts behind LangGraph.

## Choosing Your Stack: Framework vs Vanilla SDK

Every team building agents faces this fork in the road: use a framework like LangGraph, or call provider APIs directly?

| | Vanilla Provider SDKs | Agent Frameworks (LangGraph) |
|--|----------------------|------------------------------|
| **Providers** | OpenAI, Anthropic, Google directly | LangChain, LangGraph, Google ADK |
| **Control** | Full — you write everything | Abstracted — framework handles routing, state, persistence |
| **Speed to POC** | Slower | Faster |
| **Production gap** | You solve all production problems yourself | Framework gives you checkpointing, HITL, observability hooks |
| **Debugging** | Easier for simple cases | Requires understanding framework internals |
| **Lock-in** | None | Some — porting between frameworks is work |

> **Most teams underestimate the production gap.** It's easy to build a working demo with raw API calls. It's much harder to add resumable state, HITL, multi-user sessions, and observability from scratch. Frameworks exist because these patterns are solved problems.

### The Decision

- **Fast POC / experiment**: Either works. Vanilla SDK is often faster.
- **Production agent system**: A framework pays back its learning curve.
- **Key insight**: Frameworks are evolving fast. Don't lock in deeply. **Build fast POCs, validate assumptions, then commit.**

:::tip Framework Lock-In is Real
Changing frameworks mid-project is expensive. Before committing, ask: "Can this framework grow with my production requirements?" For LangGraph specifically: persistent checkpointing, HITL support, and the hub-and-spoke multi-agent pattern are built-in — these would take weeks to build from scratch.
:::

---

## Why LangGraph?

Traditional chains are linear: input → process → output. But real agents need:

- **Loops**: Retry until successful
- **Branching**: Different paths based on conditions
- **State**: Remember information across steps
- **Persistence**: Save and resume workflows

LangGraph solves these with a graph-based approach.

## Core Concepts

### 1. State

State is the shared information that flows through your graph.

> **Analogy**: State is the **whiteboard in a meeting room**. Every participant (node) can read what's on it and write to it. When someone adds new info, everyone else immediately sees it. The whiteboard is the single source of truth for the whole meeting.

```python
from typing import TypedDict, Annotated
from operator import add

class AgentState(TypedDict):
    """State shared across all nodes."""
    messages: Annotated[list, add]  # Accumulates messages
    current_step: str
    results: dict
```

**State Reducers:**

| Reducer | Behavior |
|---------|----------|
| `add` | Appends new values to list |
| Custom function | Define your own logic |
| Default | Replaces with new value |

### 2. Nodes

Nodes are functions that process state.

> **Analogy**: Nodes are **assembly line stations**. Each station does exactly one job — stamp, drill, paint. It takes the work from the previous station, does its job, and passes it along. You don't ask the painting station to also do drilling.

```python
from langchain_core.messages import AIMessage

def agent_node(state: AgentState) -> dict:
    """Process messages and decide next action."""
    messages = state["messages"]

    response = llm.invoke(messages)

    return {
        "messages": [response],
        "current_step": "thinking"
    }

def tool_node(state: AgentState) -> dict:
    """Execute tools based on agent decision."""
    # Process tool calls
    result = execute_tool(state["messages"][-1])

    return {
        "messages": [ToolMessage(content=result)],
        "current_step": "executing"
    }
```

### 3. Edges

Edges define how nodes connect.

> **Analogy**: Edges are **train tracks**. Each track connects two stations. Regular edges are straight tracks — the train always goes the same way. Conditional edges are **traffic lights** — the direction the train goes depends on what color the light is when it arrives.

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(AgentState)

# Add nodes
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

# Simple edge (always follows this path)
graph.add_edge("tools", "agent")

# Conditional edge (decides based on state)
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

# Set entry point
graph.set_entry_point("agent")
```

### 4. Graph Execution

```
                 ┌─────────────────────────────────────────┐
                 │                                         │
                 │              Agent Node                 │
                 │         (LLM Decision Making)           │
                 │                                         │
                 └─────────────────┬───────────────────────┘
                                   │
                                   ▼
                         ┌─────────────────┐
                         │  Should Continue │
                         │    (Condition)   │
                         └────────┬────────┘
                                  │
                     ┌────────────┴────────────┐
                     │                         │
                     ▼                         ▼
              ┌─────────────┐           ┌─────────────┐
              │    Tools    │           │     END     │
              │    Node     │           │             │
              └──────┬──────┘           └─────────────┘
                     │
                     │ (loops back)
                     └────────────────────────────────────────┐
                                                              │
                                                              ▼
```

## Building a Simple Agent

### Complete Example

```python
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# Define state
class AgentState(TypedDict):
    messages: Annotated[list, add]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Define nodes
def call_model(state: AgentState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# Define condition
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]

    # If the LLM made a tool call, continue
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"

    # Otherwise, end
    return "end"

# Build graph
graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "agent", "end": END}
)

# Compile
app = graph.compile()

# Run
result = app.invoke({
    "messages": [HumanMessage(content="Hello!")]
})
```

## Advanced Patterns

### Subgraphs

Nest graphs within graphs for modularity:

```python
# Create a subgraph for research
research_graph = StateGraph(ResearchState)
research_graph.add_node("search", search_node)
research_graph.add_node("summarize", summarize_node)
# ... configure research graph

# Use in main graph
main_graph = StateGraph(MainState)
main_graph.add_node("research", research_graph.compile())
main_graph.add_node("write", write_node)
```

### Checkpointing

Save and resume graph execution:

```python
from langgraph.checkpoint.memory import MemorySaver

# Add memory
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# Run with thread ID
config = {"configurable": {"thread_id": "user-123"}}
result = app.invoke({"messages": [...]}, config)

# Resume later with same thread_id
result = app.invoke({"messages": [...]}, config)
```

### Human-in-the-Loop

Pause for human input:

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Add interrupt before sensitive nodes
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["execute_action"]
)

# Run until interrupt
result = app.invoke(input, config)

# Human reviews, then continue
app.invoke(None, config)  # Resume
```

## State Channels

Control how state updates flow:

```python
from langgraph.graph import add_messages

class AgentState(TypedDict):
    # Messages accumulate (never overwritten)
    messages: Annotated[list, add_messages]

    # Current step is replaced each time
    current_step: str

    # Custom reducer for scores
    scores: Annotated[list[float], lambda a, b: a + b]
```

## Error Handling

```python
def safe_tool_node(state: AgentState):
    try:
        result = execute_tool(state)
        return {"messages": [ToolMessage(content=result)]}
    except Exception as e:
        error_msg = f"Tool error: {str(e)}"
        return {
            "messages": [ToolMessage(content=error_msg)],
            "error": True
        }

# Add conditional to handle errors
graph.add_conditional_edges(
    "tools",
    lambda state: "error" if state.get("error") else "success",
    {"error": "error_handler", "success": "agent"}
)
```

## Best Practices

### 1. Keep State Minimal

```python
# Good: Only what's needed
class AgentState(TypedDict):
    messages: Annotated[list, add]
    final_answer: str | None

# Avoid: Too much state
class AgentState(TypedDict):
    messages: list
    intermediate_results: list
    debug_info: dict
    timestamps: list
    # ...
```

### 2. Clear Node Responsibilities

```python
# Good: Single responsibility
def search_node(state): ...
def analyze_node(state): ...
def respond_node(state): ...

# Avoid: Doing too much in one node
def do_everything(state): ...
```

### 3. Test Each Node

```python
def test_agent_node():
    state = {"messages": [HumanMessage("test")]}
    result = agent_node(state)
    assert "messages" in result
```

## Checkpointer Internals — How LangGraph Remembers

The Checkpointer is LangGraph's persistence layer. Understanding its internal structure helps you debug, inspect, and resume conversations correctly.

### How it's organized

```
store = {
    "thread_abc123": [          ← ThreadId is the key
        checkpoint_0,           ← First user message
        checkpoint_1,           ← After agent node (AI response)
        checkpoint_2,           ← After tool node (tool result)
        checkpoint_3,           ← Final response
    ]
}
```

Each **checkpoint** contains:
- **Node output** — the messages appended at that step (Human, AI, or Tool message)
- **Metadata** — tracks which node just ran and which node runs next
- **Timestamp** — when this checkpoint was saved

### Why this matters

1. **Resume after crash** — because state is saved at each step, a server restart doesn't lose progress
2. **Human-in-the-loop** — the graph can pause mid-execution, save state, and wait for human input (hours later if needed)
3. **Time travel** — you can rewind to any checkpoint and re-run from there (useful for debugging and A/B testing)
4. **Multi-user** — each user gets their own ThreadId and separate checkpoint history

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver  # For production

# In-memory (development)
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# SQLite (single server production)
with SqliteSaver.from_conn_string(":memory:") as checkpointer:
    app = graph.compile(checkpointer=checkpointer)

# Every invocation creates a checkpoint
config = {"configurable": {"thread_id": "user-session-42"}}
result = app.invoke({"messages": [HumanMessage("Hello")]}, config)

# Get the full state history
for state in app.get_state_history(config):
    print(state.values)  # See every checkpoint
```

:::tip Time Travel for Debugging
`app.get_state_history(config)` returns every checkpoint for a thread. This is invaluable for debugging why an agent made a bad decision — you can inspect the exact state at every step.
:::

---

## LangSmith — Observability for Your Agents

LangSmith is Anthropic's platform for the **entire lifecycle** of an LLM application: prototyping, testing, and production monitoring.

> **The problem it solves**: When your agent gives a wrong answer, how do you know *which step* went wrong? Was it the prompt? The retrieval? The tool call? LangSmith gives you a complete trace of every step.

### What you get

| Capability | What it shows |
|-----------|---------------|
| **Tracing** | Every LLM call, tool call, and chain step with inputs/outputs |
| **Token usage** | Cost per run, per user, per session |
| **Latency** | Which steps are slow |
| **Error rates** | Where failures happen most |
| **Evaluation** | Automated quality scoring |

### Adding LangSmith to any LangChain app

```python
# In your .env file:
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=your-api-key
# LANGCHAIN_PROJECT=my-agent-project

import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-key"

# That's it. Every chain/agent call is now traced automatically.
# No code changes needed to your actual chain.
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"question": "What is RAG?"})  # This call appears in LangSmith
```

**For more granular control**, use the `@traceable` decorator:

```python
from langsmith import traceable

@traceable(name="fetch-and-analyze")
def fetch_and_analyze(query: str) -> str:
    docs = retriever.invoke(query)
    return analysis_chain.invoke({"docs": docs, "query": query})
```

:::note Works with any LLM framework
LangSmith works with LangChain, LangGraph, and raw API calls. It's not tied to the LangChain framework — you can use it to trace any Python function that calls an LLM.
:::

---

## LangFlow — Visual Agent Building

LangFlow is a **drag-and-drop visual interface** built on top of LangChain. Think of it as a whiteboard where you drag components (LLMs, prompts, retrievers, agents) and connect them visually — no code required.

### When to use LangFlow

| Use Case | LangFlow? |
|----------|-----------|
| Rapid prototyping & MVP | Yes |
| Showing non-technical stakeholders | Yes |
| Experimenting with different chain structures | Yes |
| Production deployment with complex logic | No — use code |

### How it works

1. Install locally: `pip install langflow` then `langflow run`
2. Or deploy on a cloud server (Docker available)
3. Drag components onto the canvas, connect them
4. Test with real inputs directly in the UI
5. Export to API — LangFlow generates an API endpoint for your flow

```python
# LangFlow exposes your flow as an API
import requests

response = requests.post(
    "http://localhost:7860/api/v1/run/your-flow-id",
    json={"input_value": "What is LangChain?"}
)
```

> LangFlow is perfect for demonstrating agent behavior to stakeholders who don't code, and for quickly experimenting with different architectures before committing to code.

---

## Summary

| Concept | Purpose |
|---------|---------|
| State | Shared data across nodes — the whiteboard |
| Nodes | Processing functions — assembly line stations |
| Edges | Transitions between nodes — train tracks |
| Conditions | Branching logic — traffic lights |
| Checkpoints | Persistence and resume — save files |
| LangSmith | Observability and tracing |
| LangFlow | Visual no-code agent builder |

---

## Test Your Understanding

---

> **Q1: Two users both use your LangGraph chatbot. User A asks question 1, User B asks question 2, User A asks question 3. What ensures LangGraph serves the right conversation history to User A on their second message?**
>
> <details>
> <summary>Show Answer</summary>
>
> **ThreadId**. Each user gets a unique `thread_id` in their config. The checkpointer uses `thread_id` as the key in its internal store, ensuring User A's messages never mix with User B's. Always use a stable, unique identifier (user ID, session ID) as your thread_id.
> </details>

---

> **Q2: Your LangGraph agent crashes mid-execution on step 4 of 7. After fixing the server bug and restarting, what happens when the user retries?**
>
> <details>
> <summary>Show Answer</summary>
>
> **It depends on your checkpointer**. With `MemorySaver`, the state is lost (in-memory only) — the user must start over. With a persistent checkpointer like `SqliteSaver` or `PostgresSaver`, the state saved at steps 1-3 is still there. When you re-invoke with the same `thread_id`, the graph resumes from the last checkpoint, not from scratch. This is why production agents should always use persistent checkpointers.
> </details>

---

> **Q3: What's the difference between a regular edge and a conditional edge in LangGraph?**
>
> <details>
> <summary>Show Answer</summary>
>
> A **regular edge** (`add_edge("a", "b")`) always goes from node A to node B — no decision making. A **conditional edge** (`add_conditional_edges("a", routing_fn, {...})`) calls a routing function that inspects the current state and returns a string key, which maps to the next node. Use conditional edges when the next step depends on what the previous step produced (e.g., "did the agent call a tool, or is it done?").
> </details>

---

> **Q4: You add LangSmith tracing to your app. A user reports the chatbot gave a wrong answer. What's your first step in LangSmith?**
>
> <details>
> <summary>Show Answer</summary>
>
> Find the **trace for that specific run**. LangSmith logs every LLM call with inputs, outputs, token counts, and latency. You can see exactly what prompt was sent, what context was retrieved, what the LLM generated at each step, and where the chain diverged from the correct answer. This is infinitely faster than adding print statements and re-running.
> </details>

---

## Next Steps

Proceed to [Hands-on exercises](./hands-on) to build your own LangGraph agents.
