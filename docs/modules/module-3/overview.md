---
sidebar_position: 2
title: "Overview"
description: Module 3 theory and concepts
---

# Module 3 Overview

Understanding the architecture and concepts behind LangGraph.

## Why LangGraph?

Traditional chains are linear: input → process → output. But real agents need:

- **Loops**: Retry until successful
- **Branching**: Different paths based on conditions
- **State**: Remember information across steps
- **Persistence**: Save and resume workflows

LangGraph solves these with a graph-based approach.

## Core Concepts

### 1. State

State is the shared information that flows through your graph:

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

Nodes are functions that process state:

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

Edges define how nodes connect:

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

## Summary

| Concept | Purpose |
|---------|---------|
| State | Shared data across nodes |
| Nodes | Processing functions |
| Edges | Transitions between nodes |
| Conditions | Branching logic |
| Checkpoints | Persistence and resume |

## Next Steps

Proceed to [Hands-on exercises](./hands-on) to build your own LangGraph agents.
