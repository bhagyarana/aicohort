---
sidebar_position: 2
title: "Overview"
description: Module 5 - Production patterns and observability
---

# Module 5 Overview

You've built agents that work. Now let's make them work **reliably in production** — where users depend on them, mistakes are visible, and you need to know exactly what's happening inside your system.

## The Hard Truth About Production Agents

Shipping an agent is harder than building one. Here's what changes when you go from notebook to production:

| Challenge | Development | Production |
|-----------|-------------|------------|
| **Visibility** | You watch it run live | It runs 10,000 times/day — you need logs |
| **Errors** | You see them immediately | They happen silently at 3am |
| **Quality** | You judge each response manually | You need automated quality scoring |
| **Iteration** | Edit code, re-run | Need to update prompts without deployments |
| **Debugging** | Print statements | Distributed tracing across all components |

The four biggest unsolved production challenges:
1. No centralized place to track data, prompts, and experiments
2. Hard to include non-technical stakeholders (SMEs) in quality review
3. Slow to iterate — changing a prompt requires a code deployment
4. No visibility into *why* the agent gave a wrong answer

**Arize** solves all four.

---

## From Demo to Production: A Real Story

Picture a marketing content creation workflow. **5 agents, linear pipeline:**

```
Research → Strategy → Content Write → Review → Publish
```

Single user, happy path, 3-5 minutes end to end. The audience applauds the demo. Now let's add reality.

### Step 1: Add a Real User (Human-in-the-Loop)

Same workflow, but now the human gives feedback and the agent revises — multiple rounds:

- Turn 1: "Target mid-market, not enterprise"
- Turn 3: "Tone too formal"
- Turn 5: "Add a cost section"

The workflow stretches from **2 minutes to 30-60 minutes** with human review gaps. By turn 5, the agent needs to remember what the user said in turn 1 — **but the context is overflowing**. The agent "forgets" the brand voice feedback. Not because it's dumb. Because old messages got pushed out of the context window.

### Step 2: Add Scale and Parallel Work

Now run 3 content pieces in parallel for enterprise customers — 3 orgs on shared infrastructure, each with their own data. Problems multiply:

- **Data isolation**: Org A's data must never appear in Org B's context
- **Orchestration**: Who decides what runs next?
- **Observability**: 9 parallel workflows — something fails at 3am. Which one? What step?
- **Context Engineering**: Each agent must have the right context at the right time

> This is why "it worked in the demo" is the most dangerous phrase in AI engineering.

---

## The 4 Production Problems

Every production agent system will hit these. Plan for them upfront.

| # | Problem | What it looks like in practice |
|---|---------|-------------------------------|
| **1** | **Context Explosion & Memory Loss** | Token limit hit at step 8, original brief truncated, HITL feedback forgotten |
| **2** | **Observability & Crash Recovery** | 45-minute failure with no trace — what went wrong? Can you resume? |
| **3** | **Orchestration** | HITL blocks everything. Peer-to-peer agent calls cause deadlocks. No central coordination |
| **4** | **Enterprise: Whose Data?** | Multi-tenant isolation, auth expiry during long runs, sandboxed execution, audit trails |

:::warning The Critical Mental Model
**Context window ≠ memory.** The context window is a *sliding window of recent information*. Anything outside it is gone. This is not a bug — it's how LLMs work. Your job is to engineer around it.
:::

---

---

## Arize — Observability for Production AI Agents

Arize is the industry standard platform for monitoring and evaluating LLM applications. Think of it as the combination of **Datadog + A/B testing + quality review** — but specifically built for AI.

### What Arize gives you

```
┌─────────────────────────────────────────────────────────────────┐
│                        Arize Platform                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Tracing    │  │  Evaluations │  │   Metrics Dashboard  │  │
│  │  (every step)│  │  (auto score)│  │ (cost, latency, etc) │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Prompt Hub  │  │   Datasets   │  │   Labeling Queues    │  │
│  │  (versioned) │  │ (hallucin.)  │  │   (SME review)       │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Integrating Arize — 3 Lines of Code

```python
# pip install arize-otel opentelemetry-sdk
from arize.otel import register

# Register your project — this instruments your app automatically
tracer_provider = register(
    space_id="your-space-id",
    api_key="your-api-key",
    project_name="my-financial-agent",
)

# That's it. Every LLM call, tool call, and LangGraph node
# is now traced automatically in Arize.
```

For LangGraph specifically:

```python
from openinference.instrumentation.langchain import LangChainInstrumentor

LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

# Now run your LangGraph agent — every step appears in Arize
result = app.invoke(
    {"messages": [HumanMessage("Tell me about Wells Fargo stock")]},
    config={"configurable": {"thread_id": "session-001"}}
)
```

---

## Tracing — See Every Step of Every Run

Arize traces show you the complete execution tree for every agent run:

```
User Query: "Tell me about Wells Fargo stock"
│
├── [LangGraph: agent node]
│   ├── Input: "Tell me about Wells Fargo stock"
│   ├── LLM Call: gpt-4o-mini → "I need to search for current WFC data"
│   └── Tool Decision: search_stock_data(ticker="WFC")
│
├── [LangGraph: tools node]
│   ├── Tool: search_stock_data(ticker="WFC")
│   └── Result: {price: 58.32, change: +1.2%, ...}
│
└── [LangGraph: agent node]
    ├── Input: search result + original question
    ├── LLM Call: gpt-4o-mini → "WFC is trading at $58.32..."
    └── Final Answer: "Wells Fargo (WFC) is currently trading at $58.32..."
```

Every trace includes:
- **Token counts** per step (and cost)
- **Latency** for each LLM call and tool
- **Exact inputs and outputs** at every node
- **Error details** if anything failed

---

## Evaluations — Automated Quality Scoring

Manually reviewing 10,000 agent responses isn't possible. Arize uses **LLM-as-a-judge** to automatically score your agent's outputs:

| Evaluation | What it checks | Score |
|-----------|---------------|-------|
| **Hallucination** | Did the agent make up information? | 0-1 |
| **Correctness** | Is the answer factually accurate? | 0-1 |
| **Relevance** | Does the answer address the question? | 0-1 |
| **Toxicity** | Does the response contain harmful content? | 0-1 |
| **Bias** | Are there discriminatory patterns? | 0-1 |

```python
from arize.experimental.datasets.experiments.evaluators.llm_evaluators import (
    HallucinationEvaluator,
    QACorrectnessEvaluator
)

# Arize runs these automatically on your traces
evaluators = [
    HallucinationEvaluator(model="gpt-4o"),
    QACorrectnessEvaluator(model="gpt-4o"),
]

# Results appear in the dashboard — filter by low-scoring runs
```

> **Real example**: Your financial agent is asked "What is the P/E ratio of Apple?" and responds with "14.2" (the correct answer is 28.6). The Hallucination evaluator flags this as a hallucination. You can then filter for all hallucinated responses, inspect the traces, and find the root cause — maybe your retrieval is fetching outdated documents.

---

## Metrics Dashboard — Know Your System's Health

Arize gives you a real-time dashboard with the metrics that matter:

| Metric | Why it matters |
|--------|---------------|
| **Token usage** | Direct cost indicator — unexpected spikes = bug |
| **Cost per run** | Budget management |
| **Latency (p50, p95, p99)** | User experience — p99 latency is what unhappy users experience |
| **Error rate** | System reliability |
| **Hallucination rate** | Quality over time — alert if it increases |
| **Toxicity incidents** | Safety and compliance |

```
Example Dashboard:
┌─────────────────────────────────────────────────────────────┐
│  Today's Summary                                             │
│  Total runs: 4,821    Avg cost: $0.003/run                  │
│  Avg latency: 1.2s    P99 latency: 4.8s                    │
│  Error rate: 0.3%     Hallucination rate: 2.1%             │
│                                                              │
│  ⚠️  Hallucination rate increased 40% in last 2 hours       │
│     → Likely caused by the prompt change at 14:30           │
└─────────────────────────────────────────────────────────────┘
```

---

## Prompt Hub — Iterate Without Code Deployments

One of the biggest production pain points: you want to fix a bad prompt but doing so requires a code change → PR review → deployment pipeline. Arize's **Prompt Hub** solves this.

```
Traditional prompt iteration:
Edit code → PR → Review → CI/CD → Deploy → Test → Repeat
(Takes hours to days)

With Prompt Hub:
Edit prompt in UI → Save version → A/B test → Deploy to prod
(Takes minutes)
```

```python
# In your code — pull the prompt from Arize Prompt Hub
from arize.experimental.datasets.prompt_hub import PromptHub

hub = PromptHub(api_key="your-key")
prompt_template = hub.get_prompt("financial-analysis-v3")  # versioned

chain = ChatPromptTemplate.from_template(prompt_template) | llm | parser
```

Non-technical team members (compliance, product, domain experts) can update prompts directly in the Arize UI without touching your code.

---

## Datasets and Labeling Queues — SME Review Workflow

When your auto-evaluations flag suspicious responses, you need domain experts (SMEs) to review them. Arize provides a structured workflow:

```
1. Arize flags traces as potentially hallucinated
           │
           ▼
2. Filtered traces → saved to a "Dataset" (collection of bad examples)
           │
           ▼
3. Dataset → pushed to "Labeling Queue" in Arize UI
           │
           ▼
4. SME (compliance officer, domain expert) logs into Arize
   Reviews each trace → marks: Correct / Hallucinated / Needs improvement
           │
           ▼
5. Labeled data → used to fine-tune evaluators or update prompts
```

```python
# Save suspicious traces to a dataset for SME review
from arize import Client

arize_client = Client(space_id="your-space", api_key="your-key")

# Filter all traces with hallucination_score < 0.5
# and export to a labeled dataset
arize_client.create_dataset(
    name="hallucinated-responses-march-2026",
    filters={"hallucination_score": {"lt": 0.5}},
    project_name="financial-agent"
)
```

---

## Development vs Production with Arize

| | Development | Production |
|--|-------------|------------|
| **Goal** | Build and iterate quickly | Maintain quality at scale |
| **Tracing** | Full trace on every run | Sampling (e.g., 10% of runs) |
| **Evaluations** | Run on test sets manually | Run automatically on live traffic |
| **Alerts** | Not needed | Set up for key metric thresholds |
| **Prompt changes** | Edit code directly | Prompt Hub with A/B testing |
| **Datasets** | Curated test cases | Growing from production traces |

```python
import os

# Development: trace everything
if os.getenv("ENV") == "development":
    tracer_provider = register(project_name="my-agent-dev", sample_rate=1.0)

# Production: sample 10% + keep all errors
else:
    tracer_provider = register(
        project_name="my-agent-prod",
        sample_rate=0.1,
        always_sample_on_error=True  # Always capture failures
    )
```

---

## Production Architecture — Putting It All Together

```
User Request
     │
     ▼
FastAPI Backend ──── Authentication / Rate limiting
     │
     ▼
LangGraph Agent ──── Arize tracing (automatic)
     │    │
     │    ├── Tool calls ──── MCP Server
     │    └── LLM calls  ──── Prompt Hub templates
     │
     ▼
Response ──── Arize auto-evaluation (async)
                    │
                    └── Dashboard: metrics, hallucination monitor
                    └── Alerts: Slack/email on threshold breach
                    └── Labeling Queue: SME review for flagged traces
```

---

## Context Engineering: Your Agent WILL Forget

Context engineering is the #1 production challenge — more important than prompt engineering. A perfect prompt with forgotten context produces bad output.

### The Catastrophic Forgetting Pattern

This plays out in every long-running agent:

```
Step 1:  User provides detailed brief (target audience, tone, key messages)
Step 5:  User gives feedback — "professional but approachable"
Step 8:  Token limit hit — framework compacts older messages (lossy compression)
Step 10: Agent writes final draft ignoring brief and feedback
          → They were compacted away
```

The agent didn't get dumber. The context window filled up, and old messages were pushed out. **This is predictable and preventable** — if you engineer for it from day one.

### The 4-Solution Memory Architecture

Not every agent needs all four. Pick the right layer for your use case:

```
┌────────────────────────────────────────────────────────────────┐
│                    Memory Architecture Ladder                    │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Layer 4: Knowledge Graph  ← relationships between entities     │
│  Layer 3: Vector RAG       ← semantic search over large docs     │
│  Layer 2: Long-Term Memory ← org knowledge across sessions       │
│  Layer 1: Session Memory   ← smart compaction within one chat    │
│                                                                  │
│  Start at Layer 1. Only go higher if you actually need it.      │
└────────────────────────────────────────────────────────────────┘
```

#### Solution 1: Session Memory — Smart Compaction

**Problem it solves**: Context window overflow *within* a single session.

The naive approach: let the framework auto-compress messages (lossy, uncontrolled). The smart approach: treat compaction as a retrievable event.

```
Principle: All data that flows into the agent must be stored and queryable.
After compaction → agent fetches missing context on demand (not lost forever)
```

**Stop here if**: your agents are short-lived, single-session, no cross-session needs.

#### Solution 2: Long-Term Memory — Persistent Organizational Knowledge

**Problem it solves**: The agent forgets brand voice, user preferences, and HITL feedback the moment the session ends.

Think of it like building a repo of customer knowledge — similar to how Claude Code maintains memory files about your project:

- Capture HITL feedback as organizational knowledge (not ephemeral chat)
- Store with logical namespaces: `brand/voice`, `audiences/enterprise`, `content_style/blog`
- Future sessions **start pre-loaded** with org knowledge
- Document RAG sometimes works better than vector RAG here — no retrieval tuning, full context preserved

```python
# Namespaced long-term memory example
memory_store = {
    "brand/voice": "Professional yet approachable. Never use jargon.",
    "audiences/enterprise": "CTO personas, focus on ROI and security",
    "content_style/blog": "2000 words, 3 examples, end with a call to action"
}

# Agent loads this at session start — not relying on context window
def load_org_context(namespace: str) -> str:
    return memory_store.get(namespace, "")
```

#### Solution 3: Vector RAG

**Problem it solves**: Agent needs to search a large document corpus efficiently.

Use when:
- Document set is too large to load in full context
- Agent needs to find specific facts across many sources
- Retrieval quality is measurable and tunable

**Tradeoff**: Full docs with agentic search vs vector chunks. Vector RAG is more efficient but doesn't solve *relationships* in data.

#### Solution 4: Knowledge Graph

**Problem it solves**: When your data has complex relationships that vector similarity search can't capture.

Example: A legal agent that needs to understand "which contracts reference which clauses which are affected by which regulations." Vector search returns similar text — a knowledge graph returns connected entities.

```
Vector RAG: "Find documents similar to 'GDPR data retention'"
Knowledge Graph: "Find all contracts → that reference clause X →
                  affected by GDPR → with expiry before 2026"
```

> **When to use Knowledge Graph**: When "relationships between things" matter more than "similarity to a query." Most teams don't need this. If you're not sure, you probably don't need it yet.

---

### Decision Tree: Which Memory Layer Do You Need?

```
Is this a short session with no memory requirements?
  └─ Yes → Use raw context window. No memory layer needed.
  └─ No ↓

Does the agent forget things within a single long session?
  └─ Yes → Add Solution 1: Session Memory (Smart Compaction)
  └─ No ↓

Does the agent need knowledge from previous sessions?
  └─ Yes → Add Solution 2: Long-Term Memory
  └─ No ↓

Does the agent need to search a large document corpus?
  └─ Yes → Add Solution 3: Vector RAG
  └─ No ↓

Does the agent need to traverse relationships between entities?
  └─ Yes → Add Solution 4: Knowledge Graph
  └─ No → You've over-engineered it. Go back up.
```

---

## Enterprise-Grade Agents: What Demos Never Show

If you're shipping agents to enterprise customers, there are four non-negotiable requirements that almost never appear in tutorials:

### 1. Multi-Tenant Data Isolation

Every artifact, document, and event must be **scoped by org ID**. Org A's data must never appear in Org B's context — not just by accident, but by architecture.

```python
# Every DB query, every vector search — scoped by org
def get_agent_context(org_id: str, query: str):
    return vector_store.search(
        query=query,
        filter={"org_id": org_id}  # Hard filter, not optional
    )
```

> This is not a nice-to-have. A single cross-tenant data leak is a SOC 2 violation and potentially a legal issue.

### 2. JWT Auto-Refresh for Long-Running Agents

Standard JWTs expire in 30 minutes. Your agents run for 10-60+ minutes. They will expire mid-workflow without auto-renewal.

```
Without auto-refresh:
  Agent starts at 10:00am → JWT expires at 10:30am
  Agent calls external API at 10:45am → 401 Unauthorized
  45 minutes of work lost.

With auto-refresh:
  Agent detects token expiry before each API call
  Refreshes credential silently
  Continues without interruption
```

### 3. Sandboxed Code Execution

If your agent executes code (common in data analysis, automation), every execution must be sandboxed:

| Constraint | Why |
|-----------|-----|
| Non-root user | Prevents privilege escalation |
| Read-only filesystem | Agent can't modify the host |
| Network isolated | Agent can't exfiltrate data |
| CPU/RAM/time limits | Prevents resource exhaustion (denial of service) |
| Dropped capabilities | Removes Linux kernel capabilities |

Docker with appropriate flags is the standard solution. Never let an AI-generated script run directly on the host.

### 4. Audit Trails

Every tool call, artifact creation, and HITL interaction must be recorded — queryable per org, compliance-ready:

```python
def log_agent_action(org_id: str, action_type: str, details: dict):
    audit_log.append({
        "timestamp": datetime.utcnow().isoformat(),
        "org_id": org_id,
        "action_type": action_type,  # "tool_call", "artifact_created", "human_reviewed"
        "details": details,
        "agent_session_id": current_session_id
    })
```

> Audit trails aren't just for compliance — they're your primary debugging tool when something goes wrong in a multi-hour agent run.

---

## Multi-Agent Architecture: When (and When Not) to Use It

The hub-and-spoke model is the right pattern when you do need multi-agent:

```
                    ┌─────────────────┐
                    │   Orchestrator  │
                    │     Agent       │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
       ┌──────────┐  ┌──────────┐  ┌──────────┐
       │ Research │  │ Writer   │  │ Review   │
       │  Agent   │  │  Agent   │  │  Agent   │
       └──────────┘  └──────────┘  └──────────┘
```

**The golden rule**: All agents communicate **only through the orchestrator**. Never direct agent-to-agent calls.

Why? Peer-to-peer agent communication creates:
- Deadlocks (Agent A waits for B, B waits for A)
- No single observation point for debugging
- Cascading failures that are impossible to trace

:::caution 90% of You Don't Need Multi-Agent
Multi-agent systems are hard to debug, hard to evaluate, and hard to improve. Before building one, ask: can a **single well-prompted agent with good tools** handle this? Usually the answer is yes. Multi-agent adds complexity without proportional benefit unless tasks are genuinely parallel or require deep specialization.
:::

---

## 5 Hard-Won Production Lessons

These come from real production failures — not theory:

### 1. Agents Are Non-Deterministic ML Systems — Eval Everything

Your agent is not software in the traditional sense. The same input can produce different outputs. Every component — compaction, retrieval, reasoning — must be evaluated, not just tested. "The tests pass" is not the same as "the agent is correct."

### 2. Context Engineering > Prompt Engineering

A perfect prompt with forgotten context produces bad output. A mediocre prompt with complete context often produces good output. **Invest in context management first, prompt refinement second.**

### 3. Build Observability Before Features

The instinct is to build features first and add logging later. Don't. Instrument your agent from day one. When the first production failure happens (and it will), you want traces, not print statements.

> "The first incident without observability costs you more time than building observability would have taken."

### 4. Design for Failure from Day One

Plan for: server crashes mid-run, token limits, API timeouts, tool failures. Specifically:
- Use **deterministic IDs** for all operations (retry-safe)
- Make operations **idempotent** (running twice = same result as running once)
- Use **auto-expiring locks** (never deadlock on a crashed agent)

### 5. Choose Boring — Reduce Degrees of Freedom

Every degree of freedom (tool, API call, sub-agent) is another surface for failure. Start with the minimum capable agent. Add complexity only when you've proven you need it.

```
More tools = More powerful = More ways to fail = Harder to debug
```

Build guardrails. Make behavior predictable. **Predictable, debuggable, testable > clever**.

---

## Your Production Decision Framework

Use this when architecting a new agent system:

| Question | If Yes | If No |
|---------|--------|-------|
| Does it need to call external tools? | Design with HITL for irreversible ones | Keep it as a chain |
| Will sessions last > 10 min? | Add session memory | Raw context is fine |
| Does it need cross-session memory? | Add long-term memory layer | Session memory is enough |
| Will multiple users share infrastructure? | Add multi-tenant isolation from day 1 | Single-tenant is simpler |
| Does it need to run code? | Add sandboxed execution | Direct execution OK |
| Will it run for > 30 min? | Add JWT auto-refresh | Standard auth OK |
| Do you need audit compliance? | Log every action from day 1 | Standard logging OK |
| Does it need specialized parallel tasks? | Consider multi-agent (hub-and-spoke) | Single agent is usually enough |

> **Default to simpler.** Complexity that isn't needed is technical debt that will hurt you in production.

---

## Test Your Understanding

---

> **Q1: Your agent's hallucination rate is normally 2%. It suddenly jumps to 15% at 2pm on a Tuesday. What's the most likely cause and how would you find it?**
>
> <details>
> <summary>Show Answer</summary>
>
> Most likely cause: a **prompt change or model update** at around 2pm. In Arize, filter traces by time (before/after 2pm) and look at the inputs. Check if a new prompt version was deployed to Prompt Hub. Compare the prompts side by side. You can also check if a model version changed (e.g., the LLM provider pushed a model update). The sudden spike with a clear timestamp is your diagnostic clue — gradual increases suggest data drift, sudden jumps suggest deployment events.
> </details>

---

> **Q2: Why would you use `sample_rate=0.1` in production instead of tracing every request?**
>
> <details>
> <summary>Show Answer</summary>
>
> **Cost and performance**. Sending every trace to Arize for a high-volume system (10,000+ requests/day) adds latency and cost. Sampling 10% still gives you statistically significant data for monitoring. Critically, you should combine sampling with `always_sample_on_error=True` — so you *never* miss a failure even when sampling. If your volume is low (< 1,000/day), trace everything.
> </details>

---

> **Q3: An SME reviews a flagged response and marks it as "Hallucinated." How does this labeling improve your system going forward?**
>
> <details>
> <summary>Show Answer</summary>
>
> In multiple ways: (1) The labeled example joins your **dataset** of known bad outputs, which you can use to evaluate future prompt changes against ("did this prompt fix the hallucination?"). (2) You can use labeled data to **fine-tune your evaluator** — making the automated hallucination detector more accurate for your specific domain. (3) Patterns in the labeled data reveal systematic issues (e.g., "the agent always hallucinates stock prices for small-cap companies") that guide prompt or retrieval improvements.
> </details>

---

> **Q4: What's the biggest advantage of using Prompt Hub over storing prompts in your code?**
>
> <details>
> <summary>Show Answer</summary>
>
> **Speed of iteration and stakeholder access**. When a prompt is in code, changing it requires a developer, a PR, review, and a deployment — potentially hours. With Prompt Hub, a non-technical domain expert (compliance officer, product manager) can refine the prompt directly in the UI, version it, and deploy it in minutes. This is transformative for teams where the people who understand the business logic aren't the same people who write code.
> </details>

---

> **Q5: Your marketing content agent runs for 45 minutes. By step 10, it ignores the brand voice feedback from step 3. No error was thrown. What is the most likely cause — and how do you fix it?**
>
> <details>
> <summary>Show Answer</summary>
>
> **Context window overflow / prompt compaction**. The framework compressed older messages (including the step-3 feedback) to stay within the token limit. This is the catastrophic forgetting pattern — the agent didn't get dumb, the *context* got full. Fix: implement session memory with retrievable compaction, so the agent can fetch the forgotten feedback on demand instead of losing it permanently. The key insight: all data that flows into the agent must be stored *and* queryable, so post-compaction retrieval is possible.
> </details>

---

> **Q6: You're building an agent for two enterprise clients — Acme Corp and Beta Inc — on shared infrastructure. A junior dev says "we'll add data isolation later once we validate the product." What's the risk?**
>
> <details>
> <summary>Show Answer</summary>
>
> **It's the hardest thing to retrofit**. Multi-tenant isolation must be enforced at the *data layer* — every DB query, every vector search, every artifact scoped by `org_id`. Adding it after the fact means auditing every data access path in your system, which can take weeks. More critically: a single cross-tenant data leak is a SOC 2 violation and can be a legal issue. The "validate first, isolate later" approach sounds pragmatic but is almost always more expensive than building it right from day one.
> </details>

---

## Next Steps

Proceed to [Hands-on exercises](./hands-on) to set up Arize monitoring and build a production-grade agent.
