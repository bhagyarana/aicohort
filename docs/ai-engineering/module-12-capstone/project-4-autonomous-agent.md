---
sidebar_position: 5
title: "Project 4: Autonomous Workflow Agent"
description: Build a goal-driven agent that plans and executes multi-step research and reporting workflows using web search, data extraction, summarization, and formatting tools.
---

# Project 4: Autonomous Workflow Agent

**Difficulty:** Advanced
**Modules:** 4, 7, 8, 9, 11
**Time:** 14–20 hours

---

## Problem Statement

Build an autonomous agent that:
1. Accepts a high-level goal (e.g., "research our top 3 competitors and produce a competitive analysis report")
2. Decomposes the goal into a plan with concrete steps
3. Executes each step using available tools (web search, URL fetcher, summarizer, data extractor)
4. Synthesizes findings into a structured final report
5. Operates within a token budget and step limit to prevent runaway costs

**What makes this hard**: Preventing infinite loops and runaway costs. An autonomous agent that runs unchecked can call 200 tools, spend $30, and produce garbage. The engineering challenge is building a system with hard boundaries, sensible defaults, and graceful degradation when it hits limits.

---

## Architecture

```
High-level goal (string)
         ↓
[Planner]
  Decompose goal into 3–7 concrete steps
  Each step: tool, inputs, expected output
         ↓
[Executor] — step loop (max N steps)
  ├── [Budget Guard] — abort if token/cost limit exceeded
  ├── LLM decides: which tool to call next
  ├── [Tool Dispatcher] — execute the chosen tool
  ├── [Result Summarizer] — compress tool output to fit context
  └── Update context with results
         ↓
[Synthesizer]
  Compile all step results into final report
         ↓
Structured report + execution metadata
```

---

## Implementation Guide

### Phase 1: Tool Library (2–3 hours)

Define the tools the agent can call. Each tool has: name, description, parameters, execute function.

```python
import os
import json
import time
import requests
from typing import Callable
from dataclasses import dataclass
import openai

client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema
    execute: Callable


def web_search(query: str, num_results: int = 5) -> str:
    """Search the web and return result summaries."""
    # Using DuckDuckGo Instant Answer API (no key required)
    try:
        response = requests.get(
            "https://api.duckduckgo.com/",
            params={
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1"
            },
            timeout=10
        )
        data = response.json()

        results = []
        if data.get("AbstractText"):
            results.append(f"Summary: {data['AbstractText']}")

        for topic in data.get("RelatedTopics", [])[:num_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(topic["Text"])

        if not results:
            return f"No direct results found for: {query}. Try a more specific query."

        return "\n\n".join(results[:num_results])

    except Exception as e:
        return f"Search failed: {e}"


def fetch_url(url: str, max_chars: int = 3000) -> str:
    """Fetch and extract text content from a URL."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        # Simple HTML stripping
        import re
        text = response.text
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text[:max_chars] + ("..." if len(text) > max_chars else "")

    except Exception as e:
        return f"Failed to fetch {url}: {e}"


def summarize_text(text: str, focus: str = "") -> str:
    """Summarize text, optionally focused on a specific aspect."""
    focus_instruction = f"\nFocus on: {focus}" if focus else ""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"Summarize the following text concisely (3-5 key points).{focus_instruction}\n\n{text[:4000]}"
        }],
        max_tokens=300
    )
    return response.choices[0].message.content


def extract_structured_data(text: str, schema: str) -> str:
    """Extract structured data matching a given schema from text."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Extract information from the text according to this schema:
{schema}

Text:
{text[:3000]}

Return valid JSON matching the schema. Use null for missing fields."""
        }],
        response_format={"type": "json_object"},
        max_tokens=500
    )
    return response.choices[0].message.content


def write_report_section(title: str, content: str, format_type: str = "markdown") -> str:
    """Format content as a professional report section."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Format the following research findings as a professional report section.

Section Title: {title}
Format: {format_type}
Content to format:
{content}

Write in a clear, professional style suitable for a business report."""
        }],
        max_tokens=600
    )
    return response.choices[0].message.content


# Tool registry
TOOLS = [
    Tool(
        name="web_search",
        description="Search the web for information. Use for finding current facts, company information, news.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "num_results": {"type": "integer", "description": "Number of results", "default": 5}
            },
            "required": ["query"]
        },
        execute=lambda args: web_search(args["query"], args.get("num_results", 5))
    ),
    Tool(
        name="fetch_url",
        description="Fetch and read the content of a specific URL. Use when you have a specific URL to read.",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch"},
                "max_chars": {"type": "integer", "description": "Max characters to return", "default": 3000}
            },
            "required": ["url"]
        },
        execute=lambda args: fetch_url(args["url"], args.get("max_chars", 3000))
    ),
    Tool(
        name="summarize_text",
        description="Summarize a long piece of text into key points.",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to summarize"},
                "focus": {"type": "string", "description": "Optional focus area for the summary"}
            },
            "required": ["text"]
        },
        execute=lambda args: summarize_text(args["text"], args.get("focus", ""))
    ),
    Tool(
        name="extract_structured_data",
        description="Extract specific fields from text according to a JSON schema.",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Source text"},
                "schema": {"type": "string", "description": "JSON schema describing what to extract"}
            },
            "required": ["text", "schema"]
        },
        execute=lambda args: extract_structured_data(args["text"], args["schema"])
    ),
    Tool(
        name="write_report_section",
        description="Format research findings into a professional report section.",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Section title"},
                "content": {"type": "string", "description": "Raw content to format"},
                "format_type": {"type": "string", "description": "Output format: markdown or plain", "default": "markdown"}
            },
            "required": ["title", "content"]
        },
        execute=lambda args: write_report_section(args["title"], args["content"], args.get("format_type", "markdown"))
    ),
]

TOOL_MAP = {t.name: t for t in TOOLS}
```

### Phase 2: Planner (1–2 hours)

```python
PLANNER_PROMPT = """You are a research planning agent. Break down the goal into concrete steps.

Goal: {goal}

Available tools:
{tool_descriptions}

Create a plan with 3–6 steps. Each step should:
- Have a clear, specific objective
- Specify which tool to use
- Be achievable with a single tool call

Return JSON:
{{
  "goal_summary": "one sentence summary",
  "steps": [
    {{
      "step_number": 1,
      "objective": "what this step achieves",
      "tool": "tool_name",
      "rationale": "why this step is needed"
    }}
  ]
}}"""


def create_plan(goal: str) -> dict:
    """Create an execution plan for a goal."""
    tool_descriptions = "\n".join(
        f"- {t.name}: {t.description}" for t in TOOLS
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": PLANNER_PROMPT.format(
                goal=goal,
                tool_descriptions=tool_descriptions
            )
        }],
        response_format={"type": "json_object"},
        max_tokens=800
    )
    return json.loads(response.choices[0].message.content)
```

### Phase 3: Executor with Budget Guard (2–3 hours)

```python
@dataclass
class AgentBudget:
    max_steps: int = 15
    max_tokens: int = 50_000
    max_cost_usd: float = 0.50
    max_time_seconds: float = 120.0

    steps_used: int = 0
    tokens_used: int = 0
    cost_used: float = 0.0
    start_time: float = None

    def start(self):
        self.start_time = time.time()

    def record_call(self, input_tokens: int, output_tokens: int, model: str):
        self.steps_used += 1
        self.tokens_used += input_tokens + output_tokens
        # Approximate cost
        prices = {"gpt-4o": (2.50, 10.0), "gpt-4o-mini": (0.15, 0.60)}
        ip, op = prices.get(model, (2.50, 10.0))
        self.cost_used += input_tokens / 1_000_000 * ip + output_tokens / 1_000_000 * op

    def check_limits(self) -> tuple[bool, str]:
        """Returns (ok, reason_if_not_ok)."""
        if self.steps_used >= self.max_steps:
            return False, f"Step limit reached ({self.max_steps} steps)"
        if self.tokens_used >= self.max_tokens:
            return False, f"Token limit reached ({self.tokens_used:,} tokens)"
        if self.cost_used >= self.max_cost_usd:
            return False, f"Cost limit reached (${self.cost_used:.4f})"
        if self.start_time and (time.time() - self.start_time) >= self.max_time_seconds:
            return False, f"Time limit reached ({self.max_time_seconds}s)"
        return True, ""

    def summary(self) -> dict:
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            "steps": self.steps_used,
            "tokens": self.tokens_used,
            "cost_usd": round(self.cost_used, 4),
            "time_seconds": round(elapsed, 1)
        }


EXECUTOR_SYSTEM_PROMPT = """You are an autonomous research agent executing a plan step by step.

Goal: {goal}

Plan:
{plan}

Execution history so far:
{history}

Current step {step_num}: {step_objective}

Choose the right tool and parameters for this step.
Be specific and concrete in your tool call parameters."""


def execute_step(goal: str, plan: dict, step: dict, history: list[dict], budget: AgentBudget) -> dict:
    """Execute a single plan step using the appropriate tool."""
    ok, reason = budget.check_limits()
    if not ok:
        return {"status": "ABORTED", "reason": reason}

    history_text = "\n".join(
        f"Step {h['step']}: {h['objective']} → {h['result'][:200]}..."
        for h in history[-5:]  # Last 5 steps only, to save tokens
    ) if history else "No steps completed yet."

    tool_schemas = [
        {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
        for t in TOOLS
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": EXECUTOR_SYSTEM_PROMPT.format(
                goal=goal,
                plan=json.dumps(plan["steps"], indent=2),
                history=history_text,
                step_num=step["step_number"],
                step_objective=step["objective"]
            )
        }],
        tools=tool_schemas,
        tool_choice={"type": "function", "function": {"name": step["tool"]}},
        max_tokens=600
    )

    budget.record_call(
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
        "gpt-4o"
    )

    # Execute the tool call
    tool_call = response.choices[0].message.tool_calls[0]
    tool_name = tool_call.function.name
    tool_args = json.loads(tool_call.function.arguments)

    tool = TOOL_MAP.get(tool_name)
    if not tool:
        return {"status": "ERROR", "reason": f"Unknown tool: {tool_name}"}

    tool_result = tool.execute(tool_args)

    # Summarize long results to preserve context budget
    if len(tool_result) > 1000:
        summarized = summarize_text(tool_result, focus=step["objective"])
        result_for_context = f"[Summarized] {summarized}"
    else:
        result_for_context = tool_result

    return {
        "status": "SUCCESS",
        "step": step["step_number"],
        "objective": step["objective"],
        "tool": tool_name,
        "tool_args": tool_args,
        "result": result_for_context,
        "raw_result": tool_result
    }
```

### Phase 4: Synthesis and Report Generation (1–2 hours)

```python
SYNTHESIS_PROMPT = """You are creating a final report based on completed research steps.

Goal: {goal}

Research findings (each step's results):
{findings}

Create a well-structured, professional report that:
1. Addresses the original goal directly
2. Synthesizes key findings across all research steps
3. Highlights the most important insights
4. Notes any gaps or areas where information was unavailable

Format as a markdown report with clear sections and headers."""


def synthesize_report(goal: str, execution_history: list[dict]) -> str:
    """Synthesize all execution results into a final report."""
    successful_steps = [h for h in execution_history if h.get("status") == "SUCCESS"]

    findings = "\n\n".join(
        f"**Step {h['step']}: {h['objective']}**\n{h['result']}"
        for h in successful_steps
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": SYNTHESIS_PROMPT.format(goal=goal, findings=findings)
        }],
        max_tokens=1500
    )
    return response.choices[0].message.content
```

### Phase 5: Full Agent Loop (1 hour)

```python
def run_agent(goal: str, budget: AgentBudget | None = None) -> dict:
    """Run the autonomous agent on a goal."""
    if budget is None:
        budget = AgentBudget()

    budget.start()
    print(f"\nGoal: {goal}")
    print("=" * 60)

    # Plan
    plan = create_plan(goal)
    print(f"Plan: {len(plan['steps'])} steps")
    for step in plan["steps"]:
        print(f"  {step['step_number']}. [{step['tool']}] {step['objective']}")

    # Execute
    execution_history = []
    aborted = False

    for step in plan["steps"]:
        ok, reason = budget.check_limits()
        if not ok:
            print(f"\nAborting: {reason}")
            aborted = True
            break

        print(f"\nStep {step['step_number']}: {step['objective']}")
        result = execute_step(goal, plan, step, execution_history, budget)

        if result["status"] == "ABORTED":
            print(f"  Aborted: {result['reason']}")
            aborted = True
            break

        execution_history.append(result)
        print(f"  ✓ Done ({budget.steps_used} steps, ${budget.cost_used:.4f})")

    # Synthesize
    if execution_history:
        print("\nSynthesizing report...")
        report = synthesize_report(goal, execution_history)
    else:
        report = "Agent aborted before any steps completed."

    return {
        "goal": goal,
        "plan": plan,
        "execution_history": execution_history,
        "report": report,
        "budget_summary": budget.summary(),
        "aborted": aborted
    }


# Test it
result = run_agent(
    goal="Research the current state of open-source LLM models. What are the top 3 options, their strengths, and when should developers choose one over another?",
    budget=AgentBudget(max_steps=8, max_cost_usd=0.30)
)

print("\n" + "=" * 60)
print("FINAL REPORT")
print("=" * 60)
print(result["report"])
print("\nExecution summary:", result["budget_summary"])
```

---

## Evaluation Criteria

| Criterion | Target | How to Measure |
|-----------|--------|----------------|
| Goal completion rate | > 80% of goals fully addressed | Human evaluation of 10 reports |
| Budget adherence | Never exceeds defined limits | Check `aborted` is False; verify costs |
| Report quality | Coherent, structured, factual | Human evaluation rubric |
| Step efficiency | Completes goals in < 8 steps | Track `budget_summary.steps` |
| Hallucination rate | All claims grounded in tool results | Cross-check 5 factual claims per report |

---

## Stretch Goals

1. **Reflection step**: After synthesizing the report, have the agent evaluate whether the goal was achieved and what's missing — then run additional steps if needed
2. **Parallel execution**: Run independent steps in parallel using `asyncio` to reduce total time
3. **Human-in-the-loop**: Pause before expensive or irreversible steps, ask for user confirmation
4. **Memory across runs**: Store past research in a vector DB so repeated goals can reuse prior findings
5. **Fine-tuning data collection**: Log all successful runs as (goal, plan, execution) triples for future fine-tuning

---

## Common Failure Modes

- **Infinite loop**: Agent keeps searching without progressing → enforce step limit, add a "step effectiveness" checker
- **Budget explosion**: One tool returns 20K tokens, filling context → always summarize tool outputs > 1000 chars
- **Plan hallucination**: Planner invents steps that aren't achievable with available tools → validate plan tool names against tool registry
- **Synthesis fabrication**: Final report adds claims not in any step result → add grounding check: every claim must cite a step number
- **Sequential bottleneck**: 6 independent search steps run one-by-one → identify parallelizable steps and run them with `asyncio.gather`
