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

## Next Steps

Proceed to [Hands-on exercises](./hands-on) to set up Arize monitoring and build a production-grade agent.
