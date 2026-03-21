---
sidebar_position: 1
title: "Module 12: Capstone Projects"
description: Demonstrate full-stack AI engineering competence by building four complete systems — from a research assistant to an autonomous workflow agent.
---

# Module 12: Capstone Projects

This is where you stop studying AI engineering and start doing it.

Each capstone project combines multiple modules into a complete, shippable system. They are intentionally underspecified in places — just like real projects. You will encounter tradeoffs with no single correct answer, and you will have to make engineering decisions and defend them.

The projects are ordered by complexity. Pick the one that matches your current level, or attempt all four.

## Projects

| Project | Modules Combined | Difficulty | What You Build |
|---------|-----------------|------------|----------------|
| [AI Research Assistant](./project-1-research-assistant) | 1, 2, 3, 4, 5 | Intermediate | A system that answers research questions with citations from retrieved papers |
| [Customer Support Automation](./project-2-support-automation) | 3, 4, 8, 9, 11 | Intermediate | A multi-agent support pipeline that routes, retrieves, responds, and escalates |
| [Code Generation Assistant](./project-3-code-generation) | 2, 3, 6, 8, 11 | Advanced | A coding assistant that generates, self-reviews, and iterates based on test results |
| [Autonomous Workflow Agent](./project-4-autonomous-agent) | 4, 7, 8, 9, 11 | Advanced | A goal-driven agent that plans and executes multi-step research and reporting workflows |

## How to Approach a Capstone

Each project page includes:

1. **Problem Statement** — what the system needs to do, written as a product brief
2. **Architecture Diagram** — the recommended component structure
3. **Implementation Guide** — phased breakdown of what to build in what order
4. **Starter Code** — skeleton files to scaffold the key components
5. **Evaluation Criteria** — how to know when it's actually working
6. **Stretch Goals** — extensions for when the core system is complete

### Build in phases, not all at once

The most common mistake is trying to build everything before testing anything. Each project breaks into at least 3 phases. Finish Phase 1 (the minimal working version), evaluate it honestly, then extend.

### What "done" looks like

A capstone is complete when:
- The system handles the happy path end-to-end
- You have evaluated it on at least 10 real examples
- You have identified and documented the top 3 failure modes
- The system degrades gracefully when a component fails

---

## What Comes Next

After completing a capstone project, you have a portfolio piece that demonstrates system design, evaluation thinking, and production awareness — the three skills that differentiate senior AI engineers.

Consider sharing your project in the cohort channel, documenting the key design decisions you made, and revisiting your implementation after completing the production checklist from [Module 11](../module-11-production-systems).

:::info See Also
For the practical LangGraph and LangChain implementations used in these projects, see the **[Internal Training Track](/learn/modules)**.
:::
