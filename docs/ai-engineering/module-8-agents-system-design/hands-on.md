---
sidebar_position: 3
title: "Hands-On"
description: Practical exercises — build a ReAct agent with 3 tools, implement intent routing, add episodic memory, and wire up a multi-agent orchestrator.
---

# Hands-On: Agents & System Design

These exercises build in sequence. Exercise 1 is the foundation (a working ReAct agent). Exercise 2 adds routing. Exercise 3 adds memory. Exercise 4 connects everything into a multi-agent system. All exercises use Python and the OpenAI API — adapt tool names to Anthropic's API if preferred.

**Setup for all exercises:**
```bash
pip install openai anthropic chromadb tiktoken python-dotenv
```

```python
# config.py — shared across all exercises
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

---

## Exercise 1: ReAct Agent with 3 Tools (Beginner)

**Goal:** Build a working ReAct agent from scratch with web search, a calculator, and a code interpreter tool.

**Time:** ~60 min

**Step 1 — Define the tools:**
```python
import openai
import json
import math
import ast

client = openai.OpenAI()

# Tool definitions (what the model sees)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information on any topic. Use when you need facts, news, or data you don't already know.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression. Use for any arithmetic, percentages, or numerical calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A valid Python math expression, e.g. '(142 * 0.08) + 15' or 'math.sqrt(144)'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "code_interpreter",
            "description": "Execute a small Python code snippet and return the output. Use for data transformations, string manipulation, or list operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Must end with a print() statement to return output."
                    }
                },
                "required": ["code"]
            }
        }
    }
]

# Tool implementations (what your code actually runs)
def web_search(query: str) -> str:
    """Simulated web search — replace with real search API (Tavily, SerpAPI, etc.)"""
    mock_results = {
        "python": "Python is a high-level programming language created by Guido van Rossum in 1991. Current version is 3.12.",
        "openai": "OpenAI is an AI research company. GPT-4o is their latest model as of 2024.",
        "default": f"Search results for '{query}': This is a mock result. Replace with a real search API."
    }
    for key, result in mock_results.items():
        if key in query.lower():
            return result
    return mock_results["default"]

def calculator(expression: str) -> str:
    """Safe math evaluation — only allows math operations."""
    try:
        # Restrict to safe operations only
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Calculator error: {str(e)}"

def code_interpreter(code: str) -> str:
    """Execute small Python snippets in a restricted environment."""
    import io
    import sys
    # NOTE: In production, use a sandboxed environment (e.g., Docker, E2B)
    stdout_capture = io.StringIO()
    sys.stdout = stdout_capture
    try:
        exec(code, {"__builtins__": {"print": print, "len": len, "range": range, "sorted": sorted, "list": list, "dict": dict, "str": str, "int": int, "float": float}})
        sys.stdout = sys.__stdout__
        return stdout_capture.getvalue().strip() or "Code executed successfully (no output)"
    except Exception as e:
        sys.stdout = sys.__stdout__
        return f"Code error: {str(e)}"

TOOL_REGISTRY = {
    "web_search": web_search,
    "calculator": calculator,
    "code_interpreter": code_interpreter,
}
```

**Step 2 — Build the agent loop:**
```python
def execute_tool_call(tool_name: str, tool_args: dict) -> str:
    """Route tool call to the correct implementation."""
    if tool_name not in TOOL_REGISTRY:
        return f"Error: Unknown tool '{tool_name}'. Available tools: {list(TOOL_REGISTRY.keys())}"
    return TOOL_REGISTRY[tool_name](**tool_args)

def run_react_agent(user_query: str, max_steps: int = 10, verbose: bool = True) -> str:
    """Run a ReAct agent loop until completion or step limit."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to tools. "
                "Think step by step. Before calling a tool, briefly reason about what you need. "
                "Only call a tool if you genuinely need information you don't have. "
                "If you already have enough information to answer, answer directly without calling more tools."
            )
        },
        {"role": "user", "content": user_query}
    ]

    action_history = []

    for step in range(max_steps):
        if verbose:
            print(f"\n--- Step {step + 1} ---")

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )

        message = response.choices[0].message

        # No tool calls = final answer
        if not message.tool_calls:
            if verbose:
                print(f"Final Answer: {message.content}")
            return message.content

        # Process tool calls
        messages.append(message)

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            if verbose:
                print(f"Tool: {tool_name}({tool_args})")

            # Loop detection
            action_signature = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
            if action_signature in action_history[-3:]:
                if verbose:
                    print("⚠️  Loop detected — same action repeated. Stopping.")
                return "I seem to be stuck in a loop. Here's what I found so far: " + messages[-1].content

            action_history.append(action_signature)
            result = execute_tool_call(tool_name, tool_args)

            if verbose:
                print(f"Result: {result[:200]}{'...' if len(result) > 200 else ''}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

    return "Reached maximum steps without a complete answer."

# Test your agent
if __name__ == "__main__":
    test_queries = [
        "What is 15% of 847, and then add 42 to the result?",
        "Sort this list in descending order: [34, 7, 23, 32, 5, 62]",
        "What programming language was Python named after and when was it created?",
    ]
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        run_react_agent(query)
```

**Checkpoint questions:**
- How many steps does each test query take?
- What happens if you remove the loop detection check and query something that would trigger a loop?
- Try adding a 4th tool: `file_reader(path)` that reads a local file. What do you need to change?

---

## Exercise 2: Intent Router (Intermediate)

**Goal:** Build a router that classifies user queries by intent and dispatches to specialized chains. Measure routing accuracy on a test set.

**Time:** ~45 min

**Step 1 — Define specialized chains:**
```python
def billing_chain(query: str) -> str:
    """Specialized handler for billing questions."""
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a billing specialist. Answer questions about pricing, invoices, refunds, and payment methods concisely and accurately."},
            {"role": "user", "content": query}
        ]
    ).choices[0].message.content

def technical_chain(query: str) -> str:
    """Specialized handler for technical support."""
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a technical support engineer. Help debug issues, explain error messages, and guide API integrations. Ask clarifying questions when needed."},
            {"role": "user", "content": query}
        ]
    ).choices[0].message.content

def general_chain(query: str) -> str:
    """Default handler for general inquiries."""
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful product assistant. Answer general questions about features, use cases, and getting started."},
            {"role": "user", "content": query}
        ]
    ).choices[0].message.content

CHAIN_REGISTRY = {
    "billing": billing_chain,
    "technical": technical_chain,
    "general": general_chain,
}
```

**Step 2 — Build the classifier:**
```python
ROUTE_DEFINITIONS = {
    "billing": "Questions about invoices, payments, refunds, pricing, subscriptions, or charges",
    "technical": "Bug reports, setup issues, API errors, integration problems, error messages, or code questions",
    "general": "Product features, use cases, getting started, comparisons, or anything else"
}

def classify_intent(query: str) -> tuple[str, float]:
    """Classify a query and return (route, confidence)."""
    route_descriptions = "\n".join([f"- {k}: {v}" for k, v in ROUTE_DEFINITIONS.items()])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a query classifier. Output only valid JSON."},
            {"role": "user", "content": f"""Classify this query into exactly one category.

Categories:
{route_descriptions}

Query: {query}

Return JSON: {{"category": "...", "confidence": 0.0-1.0, "reasoning": "one sentence"}}"""}
        ],
        response_format={"type": "json_object"}
    )

    result = json.loads(response.choices[0].message.content)
    category = result.get("category", "general")
    confidence = result.get("confidence", 0.5)

    # Validate category
    if category not in CHAIN_REGISTRY:
        category = "general"

    return category, confidence

def handle_with_routing(query: str, confidence_threshold: float = 0.7) -> dict:
    """Route query and return response with metadata."""
    route, confidence = classify_intent(query)

    # Fall back to general if confidence is low
    if confidence < confidence_threshold:
        route = "general"

    response = CHAIN_REGISTRY[route](query)

    return {
        "query": query,
        "route": route,
        "confidence": confidence,
        "response": response
    }
```

**Step 3 — Evaluate routing accuracy:**
```python
# Test set with ground truth labels
test_cases = [
    ("Why was I charged twice this month?", "billing"),
    ("I'm getting a 401 error when calling your API", "technical"),
    ("What's the difference between your Starter and Pro plans?", "billing"),
    ("How do I get started with your SDK?", "general"),
    ("My webhook isn't firing when I upload a file", "technical"),
    ("Can I get a refund for last month?", "billing"),
    ("What programming languages do you support?", "general"),
    ("The response time is very slow, over 10 seconds per call", "technical"),
]

def evaluate_router(test_cases: list, verbose: bool = True) -> dict:
    correct = 0
    results = []

    for query, expected_route in test_cases:
        actual_route, confidence = classify_intent(query)
        is_correct = actual_route == expected_route

        if is_correct:
            correct += 1

        results.append({
            "query": query[:50] + "..." if len(query) > 50 else query,
            "expected": expected_route,
            "actual": actual_route,
            "confidence": f"{confidence:.2f}",
            "correct": "✓" if is_correct else "✗"
        })

        if verbose:
            status = "✓" if is_correct else f"✗ (expected {expected_route})"
            print(f"{status} [{actual_route} | {confidence:.2f}] {query[:60]}")

    accuracy = correct / len(test_cases)
    print(f"\nAccuracy: {correct}/{len(test_cases)} = {accuracy:.1%}")
    return {"accuracy": accuracy, "results": results}

evaluate_router(test_cases)
```

**What to investigate:** Which misclassified queries could be fixed by improving the category descriptions? Try rewording one definition and re-running — does it improve accuracy without hurting others?

---

## Exercise 3: Episodic Memory (Intermediate)

**Goal:** Add persistent memory to the ReAct agent so it can recall relevant past conversations when starting a new session.

**Time:** ~50 min

```python
import chromadb
import time
import uuid

class EpisodicMemoryStore:
    """Store and retrieve conversation summaries using vector similarity."""

    def __init__(self, collection_name: str = "agent_episodes"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def store_episode(self, conversation_messages: list, user_id: str = "default") -> str:
        """Summarize and store a completed conversation."""
        # Only store if conversation has substance
        non_system = [m for m in conversation_messages if m["role"] != "system"]
        if len(non_system) < 2:
            return None

        # Create a conversation string for summarization
        conv_text = "\n".join([
            f"{m['role'].upper()}: {m['content'] if isinstance(m['content'], str) else '[tool call]'}"
            for m in non_system
            if isinstance(m.get('content'), str)
        ])

        # Generate summary
        summary_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": f"Summarize this conversation in 2-3 sentences. Focus on: what was asked, what tools were used, and what the final answer was.\n\n{conv_text}"}
            ]
        )
        summary = summary_response.choices[0].message.content

        # Store with metadata
        episode_id = str(uuid.uuid4())
        self.collection.add(
            documents=[summary],
            metadatas=[{
                "user_id": user_id,
                "timestamp": time.time(),
                "message_count": len(non_system)
            }],
            ids=[episode_id]
        )

        print(f"Stored episode {episode_id[:8]}...: {summary[:100]}...")
        return episode_id

    def retrieve_relevant(self, current_query: str, user_id: str = "default", n_results: int = 3) -> list[str]:
        """Find past episodes relevant to the current query."""
        try:
            results = self.collection.query(
                query_texts=[current_query],
                n_results=min(n_results, self.collection.count()),
                where={"user_id": user_id} if self.collection.count() > 0 else None
            )
            return results["documents"][0] if results["documents"] else []
        except Exception:
            return []

    def count(self) -> int:
        return self.collection.count()

def run_agent_with_memory(user_query: str, memory: EpisodicMemoryStore, user_id: str = "default", max_steps: int = 10) -> str:
    """ReAct agent with episodic memory injection at session start."""

    # Retrieve relevant past episodes
    past_episodes = memory.retrieve_relevant(user_query, user_id=user_id)

    system_content = "You are a helpful assistant with access to tools. Think step by step before using tools."

    if past_episodes:
        episodes_text = "\n".join([f"- {ep}" for ep in past_episodes])
        system_content += f"\n\nRelevant context from past conversations:\n{episodes_text}\n\nUse this context to give more personalized and consistent responses."

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_query}
    ]

    # Run the agent loop (reuse from Exercise 1)
    for step in range(max_steps):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )
        message = response.choices[0].message

        if not message.tool_calls:
            # Store this conversation before returning
            messages.append({"role": "assistant", "content": message.content})
            memory.store_episode(messages, user_id=user_id)
            return message.content

        messages.append(message)
        for tool_call in message.tool_calls:
            result = execute_tool_call(tool_call.function.name, json.loads(tool_call.function.arguments))
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

    return "Max steps reached."

# Test: run two conversations, check if the second retrieves context from the first
if __name__ == "__main__":
    memory = EpisodicMemoryStore()
    user = "user_001"

    print("=== Session 1 ===")
    r1 = run_agent_with_memory("What is 20% of 850?", memory, user_id=user)
    print(f"Response: {r1}")

    print("\n=== Session 2 (new session, same user) ===")
    r2 = run_agent_with_memory("I need to calculate another percentage like before. What is 15% of 1200?", memory, user_id=user)
    print(f"Response: {r2}")

    print(f"\nTotal episodes stored: {memory.count()}")
```

**Things to observe:**
- Does the system prompt in Session 2 contain the Session 1 summary?
- What happens if you use a different `user_id`? (Hint: it shouldn't retrieve the other user's history)
- Try storing 10+ episodes and checking if irrelevant ones are filtered out correctly

---

## Exercise 4: Multi-Agent System (Advanced)

**Goal:** Build an orchestrator agent that delegates to two specialized sub-agents: a `ResearchAgent` and a `WritingAgent`. The orchestrator decomposes a high-level goal and synthesizes the results.

**Time:** ~75 min

**Step 1 — Build specialized sub-agents:**
```python
def research_agent(task: str) -> str:
    """Agent specialized in information gathering. Has web search tool."""
    research_tools = [TOOLS[0]]  # Only web_search

    messages = [
        {"role": "system", "content": "You are a research specialist. Your job is to gather accurate, relevant information. Use web search when needed. Be thorough but concise. Return your findings as structured bullet points."},
        {"role": "user", "content": task}
    ]

    for _ in range(5):  # Max 5 steps per sub-agent
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=research_tools,
            tool_choice="auto"
        )
        message = response.choices[0].message

        if not message.tool_calls:
            return message.content

        messages.append(message)
        for tool_call in message.tool_calls:
            result = execute_tool_call(tool_call.function.name, json.loads(tool_call.function.arguments))
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

    return messages[-1].content if messages else "Research incomplete."

def writing_agent(task: str, research_context: str = "") -> str:
    """Agent specialized in drafting and formatting content."""
    system_prompt = "You are a writing specialist. Your job is to produce well-structured, clear, and engaging written content. Focus on clarity, flow, and appropriate tone for the audience."

    user_content = task
    if research_context:
        user_content = f"Research Context:\n{research_context}\n\nWriting Task:\n{task}"

    response = client.chat.completions.create(
        model="gpt-4o",  # Use stronger model for writing
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    )
    return response.choices[0].message.content

# Registry for the orchestrator
AGENT_REGISTRY = {
    "research_agent": research_agent,
    "writing_agent": writing_agent,
}
```

**Step 2 — Build the orchestrator:**
```python
def orchestrator(goal: str, verbose: bool = True) -> str:
    """Decompose a high-level goal into sub-tasks, delegate, and synthesize."""

    if verbose:
        print(f"\n{'='*60}")
        print(f"Goal: {goal}")
        print(f"{'='*60}")

    # Step 1: Plan
    plan_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a task orchestrator. Decompose goals into sub-tasks and assign to the right specialist agents. Output only valid JSON."},
            {"role": "user", "content": f"""Decompose this goal into sub-tasks.

Available agents:
- research_agent: gathers information from the web, ideal for fact-finding and research tasks
- writing_agent: drafts, structures, and formats content — needs research context to write accurately

Goal: {goal}

Return a JSON plan:
{{
  "steps": [
    {{"agent": "research_agent", "task": "...", "depends_on": null}},
    {{"agent": "writing_agent", "task": "...", "depends_on": "research_agent"}}
  ]
}}"""}
        ],
        response_format={"type": "json_object"}
    )

    plan = json.loads(plan_response.choices[0].message.content)
    steps = plan.get("steps", [])

    if verbose:
        print(f"\nPlan ({len(steps)} steps):")
        for i, step in enumerate(steps):
            print(f"  {i+1}. [{step['agent']}] {step['task'][:80]}...")

    # Step 2: Execute with dependency resolution
    step_results = {}
    for step in steps:
        agent_name = step["agent"]
        task = step["task"]
        depends_on = step.get("depends_on")

        if verbose:
            print(f"\n→ Running {agent_name}...")

        # Inject dependency results into writing tasks
        if depends_on and depends_on in step_results:
            context = step_results[depends_on]
            result = AGENT_REGISTRY[agent_name](task, research_context=context) if agent_name == "writing_agent" else AGENT_REGISTRY[agent_name](task)
        else:
            result = AGENT_REGISTRY[agent_name](task)

        step_results[agent_name] = result

        if verbose:
            print(f"  Result preview: {result[:150]}...")

    # Step 3: Synthesize final output
    all_results = "\n\n".join([f"[{agent}]:\n{result}" for agent, result in step_results.items()])
    synthesis = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a synthesis specialist. Combine sub-task outputs into a coherent final response that directly addresses the original goal."},
            {"role": "user", "content": f"Original goal: {goal}\n\nSub-task outputs:\n{all_results}\n\nProvide the final synthesized response:"}
        ]
    )

    return synthesis.choices[0].message.content

# Test the orchestrator
if __name__ == "__main__":
    goals = [
        "Write a 3-paragraph introduction to Python for complete beginners",
        "Create a comparison of OpenAI and Anthropic for a technical audience",
    ]

    for goal in goals:
        result = orchestrator(goal, verbose=True)
        print(f"\n{'='*60}")
        print("FINAL OUTPUT:")
        print(result)
```

**Self-Assessment Checklist**

- [ ] Exercise 1: ReAct agent runs end-to-end with all 3 tools working
- [ ] Exercise 1: Loop detection fires correctly on a repeated action
- [ ] Exercise 1: I can add a 4th tool without changing the core loop
- [ ] Exercise 2: Router achieves ≥ 80% accuracy on the test set
- [ ] Exercise 2: I understand which queries are hardest to classify and why
- [ ] Exercise 3: Session 2 context prompt contains Session 1 summary
- [ ] Exercise 3: Different user IDs produce isolated memory stores
- [ ] Exercise 4: Orchestrator decomposes a new goal without code changes
- [ ] Exercise 4: I can trace which sub-agent produced which part of the final output
- [ ] I can explain why tool results are strings, not objects

**Mini-Project:** Combine all four exercises into a single assistant that:
1. Routes incoming queries (billing / technical / research)
2. Uses the ReAct agent for research queries
3. Stores completed sessions in episodic memory
4. Uses the orchestrator for complex multi-step goals it detects via routing

Add a simple CLI loop that keeps running until the user types "exit".
