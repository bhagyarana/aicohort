---
sidebar_position: 4
title: "Resources"
description: Papers, frameworks, and further reading on agent architectures, tool calling, multi-agent systems, and production agent safeguards.
---

# Resources: Agents & System Design

## Papers & Research

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Yao et al. (2022). The paper that introduced the ReAct pattern. Shows that interleaving reasoning traces with actions significantly outperforms acting alone or reasoning alone. The benchmark comparisons in Section 4 are worth reading carefully.

- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) — Schick et al. (2023). Shows that models can learn when to call APIs (calculator, search, calendar) by self-supervised fine-tuning. Important conceptual foundation for understanding why tool calling works.

- [HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace](https://arxiv.org/abs/2303.17580) — Shen et al. (2023). Early example of an orchestrator-style multi-agent system using ChatGPT to plan and dispatch to specialized models for vision, audio, and NLP tasks.

- [AgentBench: Evaluating LLMs as Agents](https://arxiv.org/abs/2308.03688) — Liu et al. (2023). A benchmark for evaluating agents across 8 distinct environments. Reveals wide performance gaps between models on real-world agentic tasks vs standard benchmarks.

- [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) — Park et al. (2023). Shows how agents with memory (reflection, retrieval) can exhibit believable long-horizon behavior. The memory architecture section is directly applicable to production agent design.

- [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651) — Madaan et al. (2023). Demonstrates that models can improve their own outputs through iterative critique and revision — foundational for self-reviewing agent steps.

- [Cognitive Architectures for Language Agents](https://arxiv.org/abs/2309.02427) — Sumers et al. (2023). A survey that provides a unified framework for thinking about agent memory, action spaces, and decision-making. Good reference for understanding how the pieces fit together.

## Framework Documentation

- [LangGraph](https://langchain-ai.github.io/langgraph/) — The graph-based agent orchestration framework built on LangChain. Define agent workflows as stateful directed graphs with cycles. The best tool for production agents that need explicit control flow and state management.

- [LangChain Agents](https://python.langchain.com/docs/modules/agents/) — Higher-level agent abstraction with pre-built ReAct loops, tool wrappers, and memory integrations. Good for rapid prototyping before moving to LangGraph for production.

- [AutoGen](https://microsoft.github.io/autogen/) — Microsoft's multi-agent framework. Focuses on agent-to-agent conversation patterns, with built-in support for human-in-the-loop and code execution agents. Strong for research and complex multi-agent workflows.

- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling) — The official guide to structured tool calling with the OpenAI API. Covers parallel tool calls (multiple tools in one response), required vs optional tools, and the `tool_choice` parameter.

- [Anthropic Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — Anthropic's tool use documentation for Claude. Similar pattern to OpenAI but with some differences in the message format. Includes examples for computer use agents.

- [Tavily Search API](https://tavily.com/) — Purpose-built search API for AI agents. Returns clean, structured results optimized for LLM consumption. Better than scraping for web search tool implementations.

- [E2B Code Interpreter](https://e2b.dev/) — Sandboxed code execution environment designed for AI agents. Runs Python, JavaScript, and other languages in isolated containers. The safe alternative to `exec()` for code interpreter tools.

## Videos & Courses

- [Building Agents with LangGraph — LangChain YouTube](https://www.youtube.com/watch?v=9BPCV5TYPmg) — Step-by-step LangGraph tutorial covering state machines, conditional edges, and tool nodes. Good starting point before reading the LangGraph docs.

- [Multi-Agent Systems — Andrew Ng / DeepLearning.AI](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/) — Short course on building agents and multi-agent systems with LangGraph. Well-structured, covers ReAct, memory, and multi-agent patterns with working code.

- [Agents — Chip Huyen (AI Engineering Summit talk)](https://www.youtube.com/watch?v=sal78ACtGTc) — Practical perspective on what works and what doesn't in production agents. Covers failure modes, cost management, and when not to use agents.

- [Building Production-Ready Agents — Harrison Chase (LangChain)](https://www.youtube.com/watch?v=f_qgZtsjM88) — LangChain's founder on the lessons learned building agent infrastructure. Covers controllability, observability, and the hard parts of production deployment.

## Tools to Know

| Tool | What it does | When to use |
|------|-------------|-------------|
| LangGraph | Graph-based stateful agent orchestration | Production agents needing explicit control flow |
| LangChain Agents | ReAct + tool wrappers, rapid prototyping | First pass, proof-of-concept |
| AutoGen | Multi-agent conversation framework | Agent-to-agent coordination, research workflows |
| Tavily | Search API optimized for LLM agents | Web search tool implementation |
| E2B | Sandboxed code execution | Code interpreter tool in production |
| LangSmith | Full trace observability for LLM calls | Debugging agent behavior, evaluating steps |
| Helicone | LLM proxy with logging and caching | Production monitoring, cost tracking |
| Weights & Biases Weave | Agent trace visualization | Understanding multi-step agent runs |

## What to Read Next

- [Module 9: Evaluation & Safety](/learn/ai-engineering/module-9-evaluation-safety) — How to know if your agent is actually working correctly. Covers LLM-as-judge evaluation, hallucination detection, prompt injection attacks (especially important for tool-calling agents), and safety guardrails.
- [Module 11: Production AI Systems](/learn/ai-engineering/module-11-production-systems) — Deploying agents at scale: observability stacks, cost management, rate limiting, fallback chains, and caching strategies for multi-step agent workflows.

:::info See Also
Want to build agents in code with LangGraph? See **[Module 3: LangGraph](/learn/modules/module-3)** and **[Module 4: LangGraph Agents](/learn/modules/module-4)** in the internal training track.
:::
