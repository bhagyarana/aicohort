---
sidebar_position: 1
title: "Module 3: LangGraph Essentials"
description: Build stateful agent workflows with LangGraph
---

# Module 3: LangGraph Essentials

Welcome to Module 3! This module introduces LangGraph for building sophisticated, stateful agent workflows.

## Learning Objectives

By the end of this module, you will be able to:

- Understand graph-based agent architectures
- Create stateful workflows with LangGraph
- Implement conditional branching and loops
- Manage state across agent interactions
- Build multi-step reasoning agents

## Module Contents

| Section | Duration | Description |
|---------|----------|-------------|
| [Overview](./overview) | 30 min | Graph concepts and architecture |
| [Hands-on](./hands-on) | 60 min | Building LangGraph agents |
| [Resources](./resources) | - | Additional learning |

## What is LangGraph?

LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain with:

```
┌─────────────────────────────────────────────────────────────────┐
│                       LangGraph Features                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │    State     │    │    Nodes     │    │    Edges     │     │
│   │  Management  │    │  (Actions)   │    │  (Routing)   │     │
│   └──────────────┘    └──────────────┘    └──────────────┘     │
│                                                                  │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │   Cycles &   │    │  Human-in-   │    │  Persistence │     │
│   │    Loops     │    │  the-Loop    │    │   & Memory   │     │
│   └──────────────┘    └──────────────┘    └──────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Topics Covered

### 1. Graph Architecture
Define nodes (actions) and edges (transitions).

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_edge("agent", "tools")
```

### 2. State Management
Track information across graph execution.

```python
from typing import TypedDict, Annotated
from operator import add

class AgentState(TypedDict):
    messages: Annotated[list, add]
    next_action: str
```

### 3. Conditional Routing
Make decisions based on state.

```python
graph.add_conditional_edges(
    "agent",
    decide_next_step,
    {"continue": "tools", "end": END}
)
```

### 4. Cycles and Loops
Enable iterative agent behavior.

```python
# Agent can loop back to itself
graph.add_edge("tools", "agent")  # Creates a cycle
```

## Prerequisites

Before starting this module:

- [ ] Completed [Module 1](/learn/modules/module-1) and [Module 2](/learn/modules/module-2)
- [ ] Understanding of chains and agents
- [ ] Familiarity with Python typing

## Key Takeaways

:::note Why LangGraph?
LangGraph enables building agents that:
- Maintain state across interactions
- Make complex multi-step decisions
- Implement human-in-the-loop workflows
- Handle errors and retry gracefully
:::

## External Resources

This module references the LangGraph course from freeCodeCamp:
- [LangGraph Course Repository](https://github.com/iamvaibhavmehra/LangGraph-Course-freeCodeCamp)

## Next Steps

1. Complete the [Overview](./overview)
2. Work through [Hands-on exercises](./hands-on)
3. Check [Resources](./resources)
4. Move to [Module 4: Agents, Tools & MCP](/learn/modules/module-4)

---

**Estimated Time**: 90 minutes

**Difficulty**: Intermediate to Advanced
