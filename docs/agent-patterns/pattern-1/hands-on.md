---
sidebar_position: 2
title: "Hands-on"
description: Build a ReAct agent step by step
---

# Pattern 1: Hands-on

Build a complete ReAct agent from scratch.

## Step 1: Setup

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

## Step 2: Create Tools

```python
@tool
def search(query: str) -> str:
    """Search for information about a topic."""
    # Mock search results
    results = {
        "python": "Python is a programming language known for readability.",
        "langchain": "LangChain is a framework for LLM applications.",
        "react": "ReAct combines reasoning and acting in AI agents.",
    }

    for key, value in results.items():
        if key in query.lower():
            return value

    return f"No results found for: {query}"

@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_current_date() -> str:
    """Get the current date."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")

tools = [search, calculator, get_current_date]
```

## Step 3: Create ReAct Agent

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

# ReAct prompt
react_prompt = PromptTemplate.from_template("""
Answer the following question as best you can.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
""")

# Create agent
agent = create_react_agent(llm, tools, react_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # See the reasoning process
    max_iterations=5
)
```

## Step 4: Run the Agent

```python
# Test 1: Simple search
result = agent_executor.invoke({
    "input": "What is LangChain?"
})
print(f"\nAnswer: {result['output']}")
```

```python
# Test 2: Calculation
result = agent_executor.invoke({
    "input": "What is 25 * 17 + 100?"
})
print(f"\nAnswer: {result['output']}")
```

```python
# Test 3: Multi-step reasoning
result = agent_executor.invoke({
    "input": "What is today's date and what is 2024 minus the current year?"
})
print(f"\nAnswer: {result['output']}")
```

## Step 5: Observe the Reasoning

With `verbose=True`, you'll see:

```
> Entering new AgentExecutor chain...

Thought: I need to find information about LangChain
Action: search
Action Input: langchain

Observation: LangChain is a framework for LLM applications.

Thought: I now know the final answer
Final Answer: LangChain is a framework for building applications
with Large Language Models (LLMs).

> Finished chain.
```

## Step 6: Add Custom Reasoning

Enhance the agent with better prompting:

```python
enhanced_prompt = PromptTemplate.from_template("""
You are a helpful AI assistant that solves problems step by step.

Available tools:
{tools}

Guidelines:
1. Think carefully before acting
2. Use the most appropriate tool
3. Verify your answer makes sense
4. Be concise but complete

Format:
Question: {input}
Thought: [Your detailed reasoning]
Action: [{tool_names}]
Action Input: [Tool input]
Observation: [Result]
... (repeat as needed)
Thought: I have enough information
Final Answer: [Complete answer]

Question: {input}
Thought: {agent_scratchpad}
""")

enhanced_agent = create_react_agent(llm, tools, enhanced_prompt)
enhanced_executor = AgentExecutor(
    agent=enhanced_agent,
    tools=tools,
    verbose=True
)
```

## Step 7: Handle Errors

```python
from langchain_core.runnables import RunnableConfig

def run_agent_safely(query: str) -> str:
    """Run agent with error handling."""
    try:
        result = agent_executor.invoke(
            {"input": query},
            config=RunnableConfig(max_concurrency=1)
        )
        return result["output"]
    except Exception as e:
        return f"Agent encountered an error: {str(e)}"

# Test error handling
print(run_agent_safely("Divide 10 by 0"))
```

## Exercises

### Exercise 1: Add a Weather Tool

```python
@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    # Your implementation
    pass

# Add to tools and test
```

### Exercise 2: Multi-Turn Reasoning

Test the agent with questions requiring multiple tool calls:

```python
# Try this:
agent_executor.invoke({
    "input": "Search for Python, then calculate how many letters are in its description"
})
```

### Exercise 3: Improve Prompts

Modify the prompt to make the agent:
- More concise
- Better at admitting uncertainty
- Faster to reach conclusions

## Checklist

- [ ] Created basic tools
- [ ] Built ReAct agent
- [ ] Observed reasoning process
- [ ] Handled multi-step problems
- [ ] Added error handling

## Next Steps

Continue to the [Review](./review) session to discuss your implementation.
