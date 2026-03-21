---
sidebar_position: 2
title: "Overview"
description: Theory and concepts — the agent loop, ReAct pattern, tool calling, prompt chaining, routing, memory systems, multi-agent coordination, and failure mode prevention.
---

# Agents & System Design — Deep Dive

The difference between a chatbot and an agent is control flow. A chatbot responds. An agent decides what to do next, does it, observes the result, and decides again. That loop — repeated until a goal is reached or a stopping condition fires — is what makes an agent an agent.

## The Agent Loop: Perceive → Plan → Act → Observe

Every agent, regardless of framework, implements some version of this loop:

```
1. PERCEIVE   — receive input (user message, tool result, environment state)
2. PLAN       — reason about what action to take next
3. ACT        — execute: call a tool, generate text, update state, or stop
4. OBSERVE    — receive the result of the action
5. → back to PERCEIVE with updated context
```

The loop terminates when the agent:
- Produces a final answer
- Hits a max-steps limit
- Encounters an error it cannot recover from
- Decides to escalate to a human

This is the conceptual foundation. Every architectural pattern in this module is a variation or refinement of this loop.

---

## Reactive vs Planning Agents

| Type | How it works | Strengths | Weaknesses |
|------|-------------|-----------|------------|
| **Reactive** | Responds directly to current input, no lookahead | Fast, low cost, predictable | Can't handle multi-step tasks that require coordination |
| **Planning** | Reasons about a full sequence of steps before acting | Handles complex goals | Slow, expensive, can loop, harder to debug |
| **Hybrid** | Plans high-level, reacts at the step level | Balanced | More complex to implement |

:::tip When to use each
For tasks where the number of steps is known and small (1–3), use prompt chaining (reactive). For tasks where the agent needs to figure out *how many* steps to take and in what order, use a planning agent. Most production agents are hybrid: a planner decomposes the goal, then each step executes reactively.
:::

---

## The ReAct Pattern

ReAct (Reason + Act) is the most widely used general-purpose agent pattern. The model alternates between reasoning about what to do and calling an action, with each observation feeding back into the next reasoning step.

```
Thought: I need to find the current price of AAPL stock.
Action: search("AAPL stock price today")
Observation: AAPL is trading at $189.23 as of 11:45 AM ET.

Thought: Now I have the price. The user also asked about the 52-week high.
Action: search("AAPL 52-week high")
Observation: AAPL 52-week high is $220.20.

Thought: I have both pieces of information. I can answer the user's question now.
Final Answer: AAPL is currently trading at $189.23. Its 52-week high is $220.20.
```

**Why ReAct is robust:**
- Reasoning before acting catches errors before they happen
- Observations are grounded in real tool results, not hallucinated
- The chain of thought is inspectable — you can see *why* the agent did each thing
- Naturally handles multi-step tasks with unknown step counts

**Implementing ReAct with tool calling:**
```python
import openai
import json

tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            }
        }
    }
]

def run_react_agent(user_query: str, max_steps: int = 10) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Think step by step before calling tools. Always reason about what you know and what you still need to find out."},
        {"role": "user", "content": user_query}
    ]

    for step in range(max_steps):
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        message = response.choices[0].message

        # If no tool call, the agent has produced a final answer
        if not message.tool_calls:
            return message.content

        messages.append(message)

        # Execute each tool call and add results to context
        for tool_call in message.tool_calls:
            result = execute_tool(tool_call.function.name, json.loads(tool_call.function.arguments))
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })

    return "Max steps reached without completing the task."
```

---

## Tool Calling Architecture

Tool calling is the mechanism through which agents interact with the world. The flow is always:

```
LLM decides to use tool
    ↓
Emits structured JSON: {"tool": "search", "args": {"query": "..."}}
    ↓
Your code parses the JSON and executes the actual function
    ↓
Result string is injected back into the conversation
    ↓
LLM continues reasoning with the new information
```

**Critical insight:** The LLM never actually calls any function. It emits a description of what it wants to call. Your code does the execution. This separation is what makes agents safe to build — you control what tools exist and what they can do.

**Tool design principles:**

| Principle | Why it matters |
|-----------|----------------|
| One responsibility per tool | Easier for model to select the right tool |
| Rich descriptions with examples | Model uses description to decide when to call |
| Return strings, not complex objects | Model reads the result as text |
| Fail gracefully with error messages | Model can reason about failures and retry |
| Log every call with args and results | Observability is essential for debugging |

**Designing a good tool schema:**
```python
# Bad: too generic, model doesn't know when to use it
{"name": "get_data", "description": "Gets data"}

# Good: specific, with clear trigger conditions
{
    "name": "get_stock_price",
    "description": "Retrieve the current stock price for a publicly traded company. Use when the user asks about stock prices, market cap, or trading values. Returns current price and timestamp.",
    "parameters": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol, e.g. 'AAPL', 'MSFT', 'GOOGL'"
            }
        },
        "required": ["ticker"]
    }
}
```

---

## Prompt Chaining and Routing

Not everything needs a full agent loop. For predictable, fixed-step workflows, prompt chaining is simpler, cheaper, and more reliable.

### Prompt Chaining

Output of one LLM call becomes input to the next. Orchestration logic lives in code, not in the model's reasoning.

```python
def process_document(document: str) -> dict:
    # Step 1: Extract key entities
    entities = llm_call(
        prompt=f"Extract all named entities (people, orgs, locations) from this text. Return JSON list.\n\n{document}",
        parse_json=True
    )

    # Step 2: Classify document type
    doc_type = llm_call(
        prompt=f"Classify this document type: legal, financial, medical, general.\n\n{document}",
    )

    # Step 3: Generate summary tailored to document type
    summary = llm_call(
        prompt=f"Summarize this {doc_type} document focusing on the key {doc_type} details.\nEntities involved: {entities}\n\n{document}"
    )

    return {"entities": entities, "type": doc_type, "summary": summary}
```

### Routing

Classify intent first, then dispatch to the right specialized chain. Avoids building one massive prompt that handles everything poorly.

```
Multi-step agent architecture:

User Query
    ↓
[Router] → classify intent
    ↓
[Planner] → decompose into steps
    ↓
[Executor] → for each step:
    ├── Call LLM with current context
    ├── Parse tool calls
    ├── Execute tools
    └── Update context with results
    ↓
[Synthesizer] → compile final answer
    ↓
Response
```

```python
ROUTE_DEFINITIONS = {
    "billing": "Questions about invoices, payments, refunds, or pricing",
    "technical": "Bug reports, setup issues, API errors, integration problems",
    "general": "Product questions, feature requests, general information"
}

def route_query(user_query: str) -> str:
    route_descriptions = "\n".join([f"- {k}: {v}" for k, v in ROUTE_DEFINITIONS.items()])
    response = llm_call(
        prompt=f"Classify the following query into exactly one category:\n{route_descriptions}\n\nQuery: {user_query}\n\nReturn only the category name, nothing else."
    )
    route = response.strip().lower()
    return route if route in ROUTE_DEFINITIONS else "general"

def handle_query(user_query: str) -> str:
    route = route_query(user_query)
    handlers = {
        "billing": billing_chain,
        "technical": technical_chain,
        "general": general_chain,
    }
    return handlers[route](user_query)
```

---

## Parallel Execution

When multiple independent sub-tasks need to complete, fan them out simultaneously rather than running sequentially.

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def research_topic(topic: str) -> str:
    """Single research call — can run in parallel with others."""
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Research this topic briefly: {topic}"}]
    )
    return response.choices[0].message.content

async def parallel_research(topics: list[str]) -> list[str]:
    """Fan out all research tasks simultaneously."""
    tasks = [research_topic(topic) for topic in topics]
    results = await asyncio.gather(*tasks)
    return list(results)

async def synthesize_report(query: str) -> str:
    # Step 1: Generate sub-topics to research (sequential — depends on query)
    sub_topics_response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"List 4 key sub-topics to research for: {query}\nReturn as JSON array of strings."}]
    )
    sub_topics = json.loads(sub_topics_response.choices[0].message.content)

    # Step 2: Research all sub-topics in parallel
    research_results = await parallel_research(sub_topics)

    # Step 3: Synthesize (sequential — depends on all research results)
    combined = "\n\n".join([f"## {t}\n{r}" for t, r in zip(sub_topics, research_results)])
    synthesis = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"Synthesize this research into a coherent report:\n{combined}"}]
    )
    return synthesis.choices[0].message.content
```

**When to parallelize:** Any time you have N independent calls that don't depend on each other's results. Common patterns: multi-source research, batch classification, simultaneous tool calls.

---

## Memory Systems

Agents need different types of memory for different purposes. Confusing them leads to bloated context, missed information, or expensive retrieval.

| Memory Type | What it stores | Where | Lifetime |
|-------------|---------------|-------|----------|
| **Working memory** | Current conversation context | Token window | Current session |
| **Episodic memory** | Past conversation summaries | Database | Persistent |
| **Semantic memory** | Facts, documents, knowledge | Vector DB | Persistent |
| **Procedural memory** | Skills, behaviors, patterns | Model weights | Permanent (fine-tuning) |

### Working Memory Management

The context window is finite. As conversations grow, you must decide what to keep and what to compress:

```python
def manage_context(messages: list, max_tokens: int = 8000) -> list:
    """Keep system prompt + last N messages, summarize the middle if needed."""
    system_messages = [m for m in messages if m["role"] == "system"]
    conversation = [m for m in messages if m["role"] != "system"]

    current_tokens = estimate_tokens(messages)

    if current_tokens <= max_tokens:
        return messages  # Fits — no pruning needed

    # Summarize older messages to compress context
    if len(conversation) > 6:
        to_summarize = conversation[:-4]  # Keep last 4 messages intact
        summary = llm_call(
            prompt=f"Summarize this conversation history concisely:\n{format_messages(to_summarize)}"
        )
        compressed = [{"role": "system", "content": f"Earlier conversation summary: {summary}"}]
        return system_messages + compressed + conversation[-4:]

    return messages
```

### Episodic Memory

Store conversation summaries between sessions. Retrieve relevant past interactions when starting a new session:

```python
import chromadb

class EpisodicMemory:
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection("episodes")

    def store_episode(self, session_id: str, conversation: list) -> None:
        summary = llm_call(
            prompt=f"Summarize this conversation in 2-3 sentences, capturing the key topics and any decisions made:\n{format_messages(conversation)}"
        )
        self.collection.add(
            documents=[summary],
            metadatas=[{"session_id": session_id, "timestamp": time.time()}],
            ids=[session_id]
        )

    def retrieve_relevant(self, current_query: str, n_results: int = 3) -> list[str]:
        results = self.collection.query(
            query_texts=[current_query],
            n_results=n_results
        )
        return results["documents"][0] if results["documents"] else []
```

---

## Multi-Agent Architecture

Single agents struggle when tasks require different specialized capabilities. Multi-agent systems solve this by having an orchestrator delegate to specialized sub-agents.

```
Orchestrator Agent
    ├── receives high-level goal
    ├── decomposes into sub-tasks
    ├── assigns sub-tasks to appropriate specialist agents
    └── synthesizes results into final response

Specialist Agents
    ├── Research Agent (web search, summarization tools)
    ├── Data Agent (SQL queries, CSV analysis, charting tools)
    └── Writing Agent (drafting, editing, formatting tools)
```

**Orchestrator pattern:**
```python
def orchestrator_agent(goal: str) -> str:
    # Step 1: Plan
    plan = llm_call(
        prompt=f"""You are an orchestrator. Break this goal into sub-tasks and assign each to a specialist.
Available specialists: research_agent, data_agent, writing_agent.

Goal: {goal}

Return a JSON list of: [{{"agent": "...", "task": "..."}}]"""
    )
    tasks = json.loads(plan)

    # Step 2: Execute sub-tasks (can parallelize if independent)
    results = {}
    for item in tasks:
        agent_fn = agent_registry[item["agent"]]
        results[item["task"]] = agent_fn(item["task"])

    # Step 3: Synthesize
    return llm_call(
        prompt=f"Synthesize these sub-task results into a final response for the goal: {goal}\n\nResults: {json.dumps(results)}"
    )
```

**Common mistakes in multi-agent design:**

| Mistake | Problem | Fix |
|---------|---------|-----|
| Agents share mutable state | Race conditions, inconsistent results | Pass results explicitly as messages |
| Orchestrator re-implements sub-agent logic | Duplication, maintenance burden | Each agent does one thing; orchestrator only plans and routes |
| No timeout or cost limit per sub-agent | Runaway costs | Set max_steps and max_tokens per agent call |
| Sub-agents call each other recursively | Infinite loops | Only orchestrator initiates agent calls |

---

## Agent Failure Modes & Prevention

Agents fail in predictable ways. Build defenses before you need them.

| Failure Mode | Cause | Prevention |
|-------------|-------|-----------|
| **Infinite loop** | Agent keeps calling tools without making progress | Step counter with hard limit; detect repeated identical actions |
| **Context overflow** | Long agent runs fill the context window | Compress intermediate results; summarize completed steps |
| **Over-tool-calling** | Agent calls tools when it already has the answer | Instruct model to answer directly when it has enough information |
| **Conflicting tool outputs** | Two tools return inconsistent information | Prompt model to explicitly reconcile conflicts before deciding |
| **Hallucinated tool calls** | Model invents tool names that don't exist | Validate tool name against whitelist before execution |
| **Cost runaway** | Complex task triggers hundreds of LLM calls | Set cost budget per session; alert before cutting off |

**Loop detection:**
```python
def detect_loop(action_history: list, window: int = 3) -> bool:
    """Detect if the agent is repeating the same sequence of actions."""
    if len(action_history) < window * 2:
        return False
    recent = action_history[-window:]
    prior = action_history[-window*2:-window]
    return recent == prior
```

**Safe execution wrapper:**
```python
class SafeAgent:
    def __init__(self, max_steps: int = 15, max_cost_usd: float = 1.0):
        self.max_steps = max_steps
        self.max_cost_usd = max_cost_usd
        self.steps = 0
        self.total_cost = 0.0
        self.action_history = []

    def check_safety(self, action: str) -> None:
        self.steps += 1
        self.action_history.append(action)

        if self.steps >= self.max_steps:
            raise StopIteration(f"Max steps ({self.max_steps}) reached.")
        if self.total_cost >= self.max_cost_usd:
            raise StopIteration(f"Cost limit ${self.max_cost_usd} reached.")
        if detect_loop(self.action_history):
            raise StopIteration("Agent is stuck in a loop. Stopping.")
```

---

## Quiz

<details>
<summary>**Q1:** What is the key difference between prompt chaining and a ReAct agent?</summary>

**Answer:** Prompt chaining uses a fixed sequence of LLM calls defined in code — the developer knows in advance how many steps there are and in what order. A ReAct agent determines its own sequence of steps at runtime, reasoning about what to do next based on tool observations. Prompt chaining is predictable and cheap; ReAct agents are flexible but harder to control.
</details>

<details>
<summary>**Q2:** Why does an LLM never "actually" call a tool?</summary>

**Answer:** The LLM only emits a structured description of the tool call it wants to make (typically JSON). Your application code reads that description, executes the real function (search API, database query, etc.), and returns the result as a string. The LLM has no ability to directly execute code, make HTTP requests, or access external systems — it's always the surrounding code that does the execution.
</details>

<details>
<summary>**Q3:** A ReAct agent is running a research task and you notice it has called the same `search("machine learning basics")` query three times in a row. What failure mode is this and how do you fix it?</summary>

**Answer:** This is an infinite loop / stuck agent. Fixes: (1) implement loop detection — compare the last N actions, stop if they're identical; (2) inject the previous search result into the system prompt so the model knows it already has that information; (3) add a max-steps limit that triggers a graceful stop message; (4) include in the system prompt: "If a tool returns information you already have, do not call it again — use the existing information to proceed."
</details>

---

## Common Mistakes

**Putting all orchestration logic inside a single prompt.** When your system prompt is 3,000 tokens of "first do X, then if Y do Z, unless W do Q...", you've built a brittle prompt-driven FSM. Put control flow in code. Use LLMs for reasoning, not branching.

**Not logging tool calls.** Agents are hard to debug. If you don't log every tool call with its arguments, result, latency, and token cost, you will have no idea why an agent produced the wrong answer.

**Forgetting that tool errors are information.** If a tool fails, don't raise an exception — return the error message as a string. The model can reason about failure ("the search failed with a rate limit; I'll try a different approach") far better than it can recover from an uncaught exception crashing your loop.

**Building multi-agent systems prematurely.** A well-designed single agent with 4–5 good tools handles 80% of tasks better than a hastily designed 5-agent system. Only introduce multiple agents when a task genuinely requires specialized capabilities that conflict (e.g., one agent needs write access to a database; another should never touch it).

---

:::info See Also
Want to implement these patterns in code? See **[Module 3: LangGraph](/learn/modules/module-3)** (stateful agent graphs) and **[Module 4: Agents, Tools & MCP](/learn/modules/module-4)** (custom tools, Google ADK, Model Context Protocol) in the internal training track.
:::

## Next Steps

→ [Hands-On: Agents & System Design](./hands-on) — Build a ReAct agent, add episodic memory, wire up a multi-agent system
