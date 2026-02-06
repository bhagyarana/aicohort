---
sidebar_position: 1
title: "Pattern 3: Multi-Agent System"
description: Build collaborative agent systems
---

# Pattern 3: Multi-Agent System

Learn to build systems where multiple specialized agents collaborate to solve complex problems.

## Overview

### What You'll Build

A multi-agent system with:
- Specialized agents for different tasks
- A supervisor for coordination
- Inter-agent communication
- Collaborative problem-solving

```
┌─────────────────────────────────────────────────────────────────┐
│                    Multi-Agent System                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                    ┌──────────────────┐                         │
│                    │    Supervisor    │                         │
│                    │      Agent       │                         │
│                    └────────┬─────────┘                         │
│                             │                                    │
│          ┌──────────────────┼──────────────────┐                │
│          │                  │                  │                │
│          ▼                  ▼                  ▼                │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│   │  Researcher │    │   Writer    │    │  Reviewer   │        │
│   │    Agent    │    │    Agent    │    │    Agent    │        │
│   └─────────────┘    └─────────────┘    └─────────────┘        │
│          │                  │                  │                │
│          └──────────────────┴──────────────────┘                │
│                     Shared State                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Learning Objectives

- Design multi-agent architectures
- Implement agent coordination
- Handle inter-agent communication
- Build scalable agent systems

## Key Concepts

### Agent Roles

| Role | Responsibility | Skills |
|------|----------------|--------|
| **Supervisor** | Coordinate and delegate | Task routing, decisions |
| **Researcher** | Gather information | Search, analyze |
| **Writer** | Create content | Compose, structure |
| **Reviewer** | Quality assurance | Critique, improve |

### Coordination Patterns

```
Sequential:     A → B → C → Output

Parallel:       A ──┐
                B ──┼──→ Merge → Output
                C ──┘

Hierarchical:   Supervisor
                    ├── Agent A
                    ├── Agent B
                    └── Agent C

Debate:         Agent A ←→ Agent B → Consensus
```

## Implementation Preview

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from operator import add

class MultiAgentState(TypedDict):
    task: str
    research: Annotated[list[str], add]
    draft: str
    feedback: list[str]
    final: str

def supervisor(state):
    """Route task to appropriate agent."""
    if not state.get("research"):
        return {"next": "researcher"}
    elif not state.get("draft"):
        return {"next": "writer"}
    elif not state.get("feedback"):
        return {"next": "reviewer"}
    else:
        return {"next": "end"}

def researcher(state):
    """Research the topic."""
    pass

def writer(state):
    """Write based on research."""
    pass

def reviewer(state):
    """Review and provide feedback."""
    pass
```

## Next Steps

Continue to [Hands-on](./hands-on) to build your multi-agent system.
