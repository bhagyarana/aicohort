---
sidebar_position: 1
title: "Module 8: Agents & System Design"
description: Build systems where LLMs take actions, not just generate text. Covers the ReAct pattern, tool calling, multi-agent architectures, memory systems, and failure mode prevention.
---

# Module 8: Agents & System Design

An LLM that only generates text is a powerful autocomplete engine. An agent that perceives state, reasons about goals, calls tools, and adapts based on results is a system that can do work.

The gap between a chatbot and an agent isn't a single feature — it's an architecture. This module is about that architecture: how to design systems where language models make decisions, call tools, manage context across multiple steps, and either complete tasks or fail gracefully.

## What You'll Learn

- What makes a system an agent: the Perceive → Plan → Act → Observe loop
- Task decomposition: breaking complex goals into executable sub-tasks
- Reactive vs planning agents — when each pattern is appropriate
- The ReAct pattern (Reason + Act): the most robust general-purpose agent architecture
- Tool calling: how LLMs emit structured tool calls, how code executes them, and how results flow back
- Prompt chaining and routing: orchestrating multi-step flows in code, not in prompts
- Parallel execution: fan-out multiple calls simultaneously and merge results
- Memory systems: working memory, episodic memory, semantic memory, procedural memory
- Multi-agent architectures: orchestrators, sub-agents, and how they coordinate
- Agent failure modes: infinite loops, context overflow, over-tool-calling, conflicting tool outputs
- Production safeguards: cost controls, loop detection, human-in-the-loop escalation

## Prerequisites

- [Module 3: Prompting & Reasoning](/learn/ai-engineering/module-3-prompting-reasoning) — tool calling and output schemas are covered there; this module builds on them
- [Module 4: RAG Systems](/learn/ai-engineering/module-4-rag-systems) — semantic memory in agents uses the same retrieval stack
- Comfortable writing Python async code (for parallel tool execution exercises)
- An OpenAI or Anthropic API key with function calling / tool use support

## Time Estimate

~5 hours

## Module Structure

| Page | What's covered |
|------|----------------|
| [Overview](./overview) | The full agent architecture — loops, tools, memory types, routing, parallelism, multi-agent coordination, and failure modes with prevention patterns |
| [Hands-On](./hands-on) | Build a ReAct agent with 3 tools, implement an intent router, add episodic memory, and wire up a multi-agent orchestrator with two specialized sub-agents |
| [Resources](./resources) | Papers (ReAct, Toolformer, AgentBench), framework docs (LangGraph, AutoGen), and curated reading on agent failure modes and production safeguards |

:::info See Also
Want to build agents with LangGraph in code? See **[Module 4: LangGraph Agents](/learn/modules/module-4)** in the internal training track for hands-on LangGraph implementation.
:::

---

**Ready to start? →** [Overview: Agents & System Design](./overview)
