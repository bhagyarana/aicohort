---
sidebar_position: 3
title: "Review"
description: Pattern 3 review and discussion
---

# Pattern 3: Review

Review your multi-agent system and discuss advanced architectures.

## Architecture Review

### Agent Responsibilities

Each agent should have:
- **Single responsibility**
- **Clear inputs/outputs**
- **Defined skills**
- **Error handling**

### Coordination Patterns

| Pattern | When to Use |
|---------|-------------|
| Sequential | Dependent tasks |
| Parallel | Independent tasks |
| Hierarchical | Complex workflows |
| Peer-to-peer | Collaborative decisions |

## Discussion Questions

1. **Agent Design**
   - How did you divide responsibilities?
   - What makes a good agent boundary?

2. **State Management**
   - What state do agents share?
   - How do you prevent conflicts?

3. **Scaling**
   - How would you add more agents?
   - How would you handle failures?

## Advanced Patterns

### Dynamic Team Formation

```python
def form_team(task: str) -> List[Agent]:
    """Dynamically select agents for a task."""
    if "research" in task:
        return [researcher, analyst]
    elif "code" in task:
        return [coder, reviewer, tester]
    else:
        return [generalist]
```

### Agent Memory Sharing

```python
class SharedMemory:
    def __init__(self):
        self.findings = []
        self.decisions = []

    def add_finding(self, agent: str, finding: str):
        self.findings.append({"agent": agent, "finding": finding})

    def get_context(self) -> str:
        return "\n".join([f"{f['agent']}: {f['finding']}"
                          for f in self.findings])
```

### Consensus Building

```python
def reach_consensus(opinions: List[str]) -> str:
    """Have agents debate and reach consensus."""
    prompt = f"""
    Multiple perspectives:
    {chr(10).join(opinions)}

    Synthesize these into a consensus view that:
    - Acknowledges all valid points
    - Resolves contradictions
    - Provides a unified conclusion
    """
    return llm.invoke(prompt).content
```

## Performance Considerations

1. **Minimize agent calls** - Each call has latency
2. **Parallelize when possible** - Independent tasks
3. **Cache repeated queries** - Avoid redundant work
4. **Set timeouts** - Prevent infinite loops

## Best Practices Summary

1. **Clear agent roles** - One responsibility each
2. **Explicit routing** - Transparent decisions
3. **Shared context** - Relevant information only
4. **Graceful fallbacks** - Handle agent failures
5. **Observability** - Log all decisions

## Congratulations!

You've completed all three agent patterns! You now know how to:

- Build ReAct reasoning agents
- Create tool-using agents
- Design multi-agent systems

## Next Steps

Ready for the final challenge? Continue to the [Capstone Project](/learn/capstone).
