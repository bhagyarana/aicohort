---
sidebar_position: 4
title: "Resources"
description: Module 3 additional resources and references
---

# Module 3 Resources

Additional learning materials for LangGraph.

## Official Documentation

| Resource | Description |
|----------|-------------|
| [LangGraph Docs](https://langchain-ai.github.io/langgraph/) | Official documentation |
| [LangGraph Tutorials](https://langchain-ai.github.io/langgraph/tutorials/) | Step-by-step guides |
| [How-To Guides](https://langchain-ai.github.io/langgraph/how-tos/) | Specific implementations |
| [API Reference](https://langchain-ai.github.io/langgraph/reference/) | Detailed API docs |

## Course Repository

This module references the LangGraph course:

**Repository**: [LangGraph-Course-freeCodeCamp](https://github.com/iamvaibhavmehra/LangGraph-Course-freeCodeCamp)

Clone and explore:
```bash
git clone https://github.com/iamvaibhavmehra/LangGraph-Course-freeCodeCamp.git
```

## Key Concepts Reference

### State Definition Patterns

```python
# Basic state
class SimpleState(TypedDict):
    messages: list
    output: str

# With reducers
class AccumulatingState(TypedDict):
    messages: Annotated[list, add]
    scores: Annotated[list[float], lambda a, b: a + b]

# With optional fields
class OptionalState(TypedDict):
    required_field: str
    optional_field: str | None
```

### Graph Patterns

```python
# Linear graph
graph.set_entry_point("a")
graph.add_edge("a", "b")
graph.add_edge("b", "c")
graph.add_edge("c", END)

# Conditional branching
graph.add_conditional_edges("a", condition_fn, {"x": "b", "y": "c"})

# Loop
graph.add_edge("b", "a")  # Back to start

# Parallel (using subgraphs)
# ... separate subgraphs execute independently
```

### Checkpointing Options

```python
# In-memory (development)
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()

# SQLite (local persistence)
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string(":memory:")

# PostgreSQL (production)
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string("postgresql://...")
```

## Common Patterns

### ReAct Agent Pattern

```python
# Reason → Act → Observe → Repeat
while not done:
    thought = llm.think(state)
    action = llm.decide_action(thought)
    observation = execute(action)
    state.update(observation)
```

### Plan-and-Execute

```python
# 1. Create plan
plan = llm.create_plan(task)

# 2. Execute steps
for step in plan:
    result = execute_step(step)
    update_plan_if_needed(result)
```

### Multi-Agent Handoff

```python
def supervisor(state):
    """Decide which agent should handle the task."""
    return {"next_agent": "researcher"}  # or "writer", "reviewer"

graph.add_conditional_edges(
    "supervisor",
    lambda s: s["next_agent"],
    {"researcher": "research_agent", "writer": "write_agent"}
)
```

## Debugging Tips

### Visualize Your Graph

```python
# Print graph structure
print(app.get_graph().draw_mermaid())

# Save to file
with open("graph.md", "w") as f:
    f.write(app.get_graph().draw_mermaid())
```

### Trace Execution

```python
# Enable LangSmith tracing
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-key"

# Run with tracing
result = app.invoke(input)
```

### Debug State

```python
def debug_node(state):
    """Print state for debugging."""
    print(f"Current state: {state}")
    return {}

graph.add_node("debug", debug_node)
```

## Video Resources

| Topic | Resource |
|-------|----------|
| LangGraph Introduction | LangChain YouTube |
| Building Agents | freeCodeCamp Course |
| Advanced Patterns | LangChain Blog |

## Books & Articles

- [Building AI Agents](https://www.deeplearning.ai/) - DeepLearning.AI
- [LangGraph Blog Posts](https://blog.langchain.dev/tag/langgraph/)
- [Agent Architectures](https://lilianweng.github.io/posts/2023-06-23-agent/)

## Practice Projects

### 1. Research Assistant

Build an agent that:
- Searches for information
- Summarizes findings
- Answers follow-up questions

### 2. Code Review Agent

Create a workflow that:
- Analyzes code
- Identifies issues
- Suggests improvements
- Generates report

### 3. Customer Support Bot

Implement an agent that:
- Classifies inquiries
- Routes to specialists
- Escalates when needed
- Maintains conversation history

## Next Module

Continue to [Module 4: Agents, Tools & MCP](/learn/modules/module-4) to learn:
- Creating custom tools
- Google ADK agents
- Model Context Protocol (MCP)
- Multi-agent systems
