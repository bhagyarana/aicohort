---
sidebar_position: 3
title: "Hands-on"
description: Module 3 practical exercises
---

# Module 3 Hands-on Exercises

Build stateful agent workflows with LangGraph.

## Exercise 1: Basic Graph Structure

Create your first LangGraph workflow.

### Setup

```python
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, END

load_dotenv()
```

### Define State

```python
class SimpleState(TypedDict):
    """State for our simple graph."""
    input: str
    steps: Annotated[list[str], add]
    output: str
```

### Create Nodes

```python
def step_one(state: SimpleState) -> dict:
    """First processing step."""
    print("Executing Step 1")
    return {
        "steps": ["Step 1: Received input"],
        "output": state["input"].upper()
    }

def step_two(state: SimpleState) -> dict:
    """Second processing step."""
    print("Executing Step 2")
    return {
        "steps": ["Step 2: Processed data"],
        "output": f"Processed: {state['output']}"
    }

def step_three(state: SimpleState) -> dict:
    """Final step."""
    print("Executing Step 3")
    return {
        "steps": ["Step 3: Finalized"],
        "output": f"Final: {state['output']}"
    }
```

### Build and Run Graph

```python
# Create graph
graph = StateGraph(SimpleState)

# Add nodes
graph.add_node("step_one", step_one)
graph.add_node("step_two", step_two)
graph.add_node("step_three", step_three)

# Add edges
graph.set_entry_point("step_one")
graph.add_edge("step_one", "step_two")
graph.add_edge("step_two", "step_three")
graph.add_edge("step_three", END)

# Compile
app = graph.compile()

# Run
result = app.invoke({"input": "hello world", "steps": [], "output": ""})
print(f"Steps: {result['steps']}")
print(f"Output: {result['output']}")
```

---

## Exercise 2: Conditional Routing

Add decision-making to your graph.

### State with Decision

```python
class ConditionalState(TypedDict):
    input: str
    category: str
    result: str
```

### Classifier Node

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

def classify_input(state: ConditionalState) -> dict:
    """Classify the input into a category."""
    response = llm.invoke(
        f"Classify this input as 'question', 'command', or 'statement': {state['input']}"
    )

    category = response.content.lower()
    if "question" in category:
        return {"category": "question"}
    elif "command" in category:
        return {"category": "command"}
    else:
        return {"category": "statement"}
```

### Handler Nodes

```python
def handle_question(state: ConditionalState) -> dict:
    response = llm.invoke(f"Answer this question: {state['input']}")
    return {"result": response.content}

def handle_command(state: ConditionalState) -> dict:
    response = llm.invoke(f"Acknowledge this command: {state['input']}")
    return {"result": response.content}

def handle_statement(state: ConditionalState) -> dict:
    response = llm.invoke(f"Respond to this statement: {state['input']}")
    return {"result": response.content}
```

### Routing Function

```python
def route_by_category(state: ConditionalState) -> str:
    """Route to appropriate handler based on category."""
    return state["category"]
```

### Build Conditional Graph

```python
graph = StateGraph(ConditionalState)

# Add nodes
graph.add_node("classify", classify_input)
graph.add_node("question", handle_question)
graph.add_node("command", handle_command)
graph.add_node("statement", handle_statement)

# Entry point
graph.set_entry_point("classify")

# Conditional edges
graph.add_conditional_edges(
    "classify",
    route_by_category,
    {
        "question": "question",
        "command": "command",
        "statement": "statement"
    }
)

# All handlers go to END
graph.add_edge("question", END)
graph.add_edge("command", END)
graph.add_edge("statement", END)

# Compile and run
app = graph.compile()
result = app.invoke({"input": "What is the capital of France?"})
print(f"Result: {result['result']}")
```

---

## Exercise 3: Agent with Tools

Build an agent that can use tools.

### Define Tools

```python
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_weather(city: str) -> str:
    """Get weather for a city (mock)."""
    # Mock response
    return f"Weather in {city}: Sunny, 72F"

tools = [calculator, get_weather]
```

### Agent State

```python
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

class AgentState(TypedDict):
    messages: Annotated[list, add]
```

### Agent Node

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

def agent(state: AgentState) -> dict:
    """Call the LLM with tools."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}
```

### Tool Node

```python
def tool_executor(state: AgentState) -> dict:
    """Execute tool calls."""
    last_message = state["messages"][-1]

    tool_results = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        # Find and execute tool
        tool_fn = {"calculator": calculator, "get_weather": get_weather}[tool_name]
        result = tool_fn.invoke(tool_args)

        tool_results.append(
            ToolMessage(
                content=result,
                tool_call_id=tool_call["id"]
            )
        )

    return {"messages": tool_results}
```

### Routing Logic

```python
def should_continue(state: AgentState) -> str:
    """Decide whether to continue or end."""
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"
```

### Build Agent Graph

```python
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("agent", agent)
graph.add_node("tools", tool_executor)

# Set entry point
graph.set_entry_point("agent")

# Add edges
graph.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "end": END}
)
graph.add_edge("tools", "agent")  # Loop back

# Compile
app = graph.compile()

# Test
result = app.invoke({
    "messages": [HumanMessage(content="What is 25 * 4?")]
})
print(result["messages"][-1].content)
```

---

## Exercise 4: Stateful Conversation

Build an agent with memory.

### Setup Memory

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
```

### Compile with Checkpointer

```python
app = graph.compile(checkpointer=memory)
```

### Run with Thread

```python
# Configuration with thread ID
config = {"configurable": {"thread_id": "conversation-1"}}

# First message
result = app.invoke({
    "messages": [HumanMessage(content="Hi, my name is Alice")]
}, config)
print(result["messages"][-1].content)

# Second message (remembers context)
result = app.invoke({
    "messages": [HumanMessage(content="What's my name?")]
}, config)
print(result["messages"][-1].content)
```

---

## Exercise 5: Multi-Step Reasoning Agent

Build an agent that reasons through problems.

### Research Agent

```python
from pydantic import BaseModel, Field

class ResearchState(TypedDict):
    question: str
    research_notes: Annotated[list[str], add]
    final_answer: str | None

def research_step(state: ResearchState) -> dict:
    """Conduct research on the question."""
    prompt = f"""
    Research question: {state['question']}

    Current notes:
    {chr(10).join(state['research_notes']) if state['research_notes'] else 'None yet'}

    Provide a research note (1-2 sentences) that helps answer the question.
    If you have enough information, say "RESEARCH COMPLETE".
    """

    response = llm.invoke(prompt)
    return {"research_notes": [response.content]}

def synthesize(state: ResearchState) -> dict:
    """Synthesize research into final answer."""
    prompt = f"""
    Question: {state['question']}

    Research notes:
    {chr(10).join(state['research_notes'])}

    Provide a comprehensive answer based on the research.
    """

    response = llm.invoke(prompt)
    return {"final_answer": response.content}

def should_continue_research(state: ResearchState) -> str:
    """Decide if more research is needed."""
    if not state["research_notes"]:
        return "research"

    last_note = state["research_notes"][-1]
    if "RESEARCH COMPLETE" in last_note or len(state["research_notes"]) >= 3:
        return "synthesize"

    return "research"
```

### Build Research Graph

```python
graph = StateGraph(ResearchState)

graph.add_node("research", research_step)
graph.add_node("synthesize", synthesize)

graph.set_entry_point("research")

graph.add_conditional_edges(
    "research",
    should_continue_research,
    {"research": "research", "synthesize": "synthesize"}
)
graph.add_edge("synthesize", END)

app = graph.compile()

# Run
result = app.invoke({
    "question": "What are the benefits of using LangGraph for AI agents?",
    "research_notes": []
})
print(f"Answer: {result['final_answer']}")
```

---

## Checklist

Before moving to Module 4:

- [ ] Create basic graphs with nodes and edges
- [ ] Implement conditional routing
- [ ] Build tool-using agents
- [ ] Use memory for stateful conversations
- [ ] Create multi-step reasoning workflows

## Next Steps

1. Review [Resources](./resources)
2. Continue to [Module 4: Agents, Tools & MCP](/learn/modules/module-4)
