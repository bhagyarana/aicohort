---
sidebar_position: 2
title: "Hands-on"
description: Build a multi-agent system
---

# Pattern 3: Hands-on

Build a complete multi-agent content creation system.

## Step 1: Define State

```python
from typing import TypedDict, Annotated, List, Optional
from operator import add
from langgraph.graph import StateGraph, END

class ContentState(TypedDict):
    topic: str
    research_notes: Annotated[List[str], add]
    outline: Optional[str]
    draft: Optional[str]
    review_feedback: Optional[str]
    final_content: Optional[str]
    current_agent: str
    iteration: int
```

## Step 2: Create Specialized Agents

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

def researcher_agent(state: ContentState) -> dict:
    """Research agent gathers information on the topic."""
    prompt = f"""You are a research specialist.

Topic: {state['topic']}

Existing research: {state['research_notes'] if state['research_notes'] else 'None yet'}

Provide 3-5 key research points about this topic.
Be factual and informative."""

    response = llm.invoke(prompt)

    return {
        "research_notes": [response.content],
        "current_agent": "researcher"
    }

def planner_agent(state: ContentState) -> dict:
    """Planner creates content outline."""
    prompt = f"""You are a content planner.

Topic: {state['topic']}

Research Notes:
{chr(10).join(state['research_notes'])}

Create a structured outline for an article about this topic.
Include: Introduction, 3-4 main sections, Conclusion."""

    response = llm.invoke(prompt)

    return {
        "outline": response.content,
        "current_agent": "planner"
    }

def writer_agent(state: ContentState) -> dict:
    """Writer creates the content draft."""
    prompt = f"""You are a content writer.

Topic: {state['topic']}

Outline:
{state['outline']}

Research:
{chr(10).join(state['research_notes'])}

Write a complete article following the outline.
Make it engaging and informative."""

    response = llm.invoke(prompt)

    return {
        "draft": response.content,
        "current_agent": "writer"
    }

def reviewer_agent(state: ContentState) -> dict:
    """Reviewer provides feedback on the draft."""
    prompt = f"""You are a content reviewer.

Original topic: {state['topic']}

Draft:
{state['draft']}

Review this draft and provide:
1. What's working well
2. What needs improvement
3. Specific suggestions

Be constructive and helpful."""

    response = llm.invoke(prompt)

    return {
        "review_feedback": response.content,
        "current_agent": "reviewer"
    }

def editor_agent(state: ContentState) -> dict:
    """Editor finalizes the content."""
    prompt = f"""You are a content editor.

Draft:
{state['draft']}

Review Feedback:
{state['review_feedback']}

Create the final version incorporating the feedback.
Polish the writing and ensure quality."""

    response = llm.invoke(prompt)

    return {
        "final_content": response.content,
        "current_agent": "editor"
    }
```

## Step 3: Create Supervisor

```python
def supervisor(state: ContentState) -> dict:
    """Supervisor decides which agent to activate next."""

    if not state.get("research_notes"):
        next_agent = "researcher"
    elif not state.get("outline"):
        next_agent = "planner"
    elif not state.get("draft"):
        next_agent = "writer"
    elif not state.get("review_feedback"):
        next_agent = "reviewer"
    elif not state.get("final_content"):
        next_agent = "editor"
    else:
        next_agent = "complete"

    return {"current_agent": next_agent}

def route_to_agent(state: ContentState) -> str:
    """Route to the appropriate agent."""
    return state.get("current_agent", "supervisor")
```

## Step 4: Build the Graph

```python
# Create the graph
graph = StateGraph(ContentState)

# Add nodes
graph.add_node("supervisor", supervisor)
graph.add_node("researcher", researcher_agent)
graph.add_node("planner", planner_agent)
graph.add_node("writer", writer_agent)
graph.add_node("reviewer", reviewer_agent)
graph.add_node("editor", editor_agent)

# Set entry point
graph.set_entry_point("supervisor")

# Add conditional edges from supervisor
graph.add_conditional_edges(
    "supervisor",
    route_to_agent,
    {
        "researcher": "researcher",
        "planner": "planner",
        "writer": "writer",
        "reviewer": "reviewer",
        "editor": "editor",
        "complete": END
    }
)

# All agents return to supervisor
for agent in ["researcher", "planner", "writer", "reviewer", "editor"]:
    graph.add_edge(agent, "supervisor")

# Compile
multi_agent = graph.compile()
```

## Step 5: Run the System

```python
# Execute the multi-agent workflow
result = multi_agent.invoke({
    "topic": "The Future of AI Agents in Software Development",
    "research_notes": [],
    "outline": None,
    "draft": None,
    "review_feedback": None,
    "final_content": None,
    "current_agent": "supervisor",
    "iteration": 0
})

# Display results
print("=== FINAL CONTENT ===")
print(result["final_content"])
```

## Step 6: Add Iteration Loop

```python
def should_iterate(state: ContentState) -> str:
    """Decide if more iterations are needed."""
    if state.get("iteration", 0) >= 2:  # Max 2 iterations
        return "finalize"

    # Check if review suggests major changes
    feedback = state.get("review_feedback", "")
    if "major revision" in feedback.lower():
        return "revise"

    return "finalize"

# Modified graph with iteration
graph.add_conditional_edges(
    "reviewer",
    should_iterate,
    {
        "revise": "writer",  # Go back to writer
        "finalize": "editor"  # Proceed to editor
    }
)
```

## Step 7: Parallel Execution

```python
from langgraph.graph import StateGraph
from typing import TypedDict, List

class ParallelState(TypedDict):
    query: str
    web_results: List[str]
    db_results: List[str]
    combined: str

def web_researcher(state):
    # Search web
    return {"web_results": ["Web finding 1", "Web finding 2"]}

def db_researcher(state):
    # Search database
    return {"db_results": ["DB finding 1", "DB finding 2"]}

def combiner(state):
    all_results = state["web_results"] + state["db_results"]
    return {"combined": "\n".join(all_results)}

# Parallel execution using fan-out/fan-in
parallel_graph = StateGraph(ParallelState)
parallel_graph.add_node("web", web_researcher)
parallel_graph.add_node("db", db_researcher)
parallel_graph.add_node("combine", combiner)

# Both run in parallel from start
parallel_graph.set_entry_point("web")
parallel_graph.add_edge("web", "combine")
parallel_graph.add_edge("db", "combine")
```

## Exercises

### Exercise 1: Add Fact Checker

Add a fact-checking agent that verifies claims before final publication.

### Exercise 2: Debate Pattern

Create two agents that debate a topic and reach consensus.

### Exercise 3: Hierarchical Teams

Build sub-teams (research team, writing team) under a main supervisor.

## Checklist

- [ ] Created specialized agents
- [ ] Built supervisor logic
- [ ] Implemented agent routing
- [ ] Added iteration capability
- [ ] Tested end-to-end workflow

## Next Steps

Continue to the [Review](./review) session.
