---
sidebar_position: 4
title: "Resources"
description: Papers, tools, deployment guides, and further reading on production AI systems
---

# Resources: Production AI Systems

## Papers & Research

- **[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)** (Kwon et al., 2023) — The paper behind vLLM. Explains how KV cache management dramatically improves LLM serving throughput. Highly relevant if you're self-hosting models.

- **[ORCA: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)** (Yu et al., OSDI 2022) — Introduces continuous batching for LLM inference servers. Explains why static batching is inefficient and how dynamic batching works.

- **[Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314)** (Snell et al., 2024) — Research on when to spend more compute at inference time vs training time. Relevant for designing cost-efficient systems.

---

## Observability & Monitoring Tools

| Tool | What It Does | Best For |
|------|-------------|----------|
| **[LangSmith](https://smith.langchain.com)** | Distributed tracing for LLM chains | LangChain users, prompt iteration |
| **[Langfuse](https://langfuse.com)** | Open-source LLM observability | Self-hosted tracing, team collaboration |
| **[Helicone](https://helicone.ai)** | LLM proxy with built-in logging | Drop-in cost/latency tracking |
| **[Braintrust](https://braintrust.dev)** | LLM evaluation + observability | Evaluation-focused teams |
| **[Datadog LLM Observability](https://www.datadoghq.com/product/llm-observability/)** | Enterprise LLM monitoring | Teams already on Datadog |
| **[Prometheus + Grafana](https://prometheus.io)** | General metrics and dashboards | Custom metrics, self-hosted |

---

## Deployment Infrastructure

- **[vLLM](https://github.com/vllm-project/vllm)** — High-throughput LLM serving with PagedAttention and continuous batching. The standard for self-hosting open-source models.

- **[Ray Serve](https://docs.ray.io/en/latest/serve/index.html)** — Scalable model serving framework. Good for custom serving logic and multi-model systems.

- **[AWS SageMaker Real-Time Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html)** — Managed endpoint deployment on AWS. Simplest path for teams already on AWS.

- **[Google Cloud Run](https://cloud.google.com/run)** — Serverless container deployment. Good for variable-traffic, stateless LLM endpoints.

- **[Modal](https://modal.com)** — Serverless GPU compute platform. Good for running open-source model inference without managing infrastructure.

---

## Cost Calculators & Pricing References

- **[OpenAI Pricing](https://openai.com/api/pricing/)** — Current token pricing for all OpenAI models. Bookmark and check before capacity planning.

- **[Anthropic Pricing](https://www.anthropic.com/pricing)** — Current Claude pricing. Includes prompt caching discounts.

- **[LLM Pricing Calculator (llmprices.com)](https://llmprices.com)** — Side-by-side comparison of pricing across all major providers and models. Updated frequently.

- **[Token Counter (platform.openai.com/tokenizer)](https://platform.openai.com/tokenizer)** — Count tokens in a prompt before sending. Useful for capacity planning and cost estimation.

---

## Caching Libraries

- **[Redis](https://redis.io/docs/getting-started/)** — In-memory data store used for both exact-match and semantic caching. Standard choice.

- **[GPTCache](https://github.com/zilliztech/GPTCache)** — Purpose-built semantic cache for LLMs. Integrates directly with OpenAI and LangChain.

- **[Semantic Cache (Upstash)](https://upstash.com/blog/semantic-caching-llm)** — Managed semantic cache service. Useful if you don't want to manage Redis.

---

## Security & Guardrails

- **[Guardrails AI](https://github.com/guardrails-ai/guardrails)** — Python library for adding input/output validation to LLM calls. Validates structured output schemas, PII detection, and policy enforcement.

- **[Nemo Guardrails (NVIDIA)](https://github.com/NVIDIA/NeMo-Guardrails)** — Configurable rails for LLM applications. Supports topical rails, fact-checking, and jailbreak prevention.

- **[LLM Guard](https://llm-guard.com)** — Security toolkit for LLM applications: prompt injection detection, PII anonymization, toxicity filtering.

---

## Books & Long-Form Reading

- **[Designing Machine Learning Systems — Chip Huyen](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)** — The best book on ML systems design. Chapters on feature stores, model deployment, monitoring, and data management are directly applicable to LLM systems.

- **[Building LLM Applications for Production — Chip Huyen](https://huyenchip.com/2023/04/11/llm-engineering.html)** — Free blog post. One of the most practical overviews of LLM engineering concerns in production.

- **[The Pragmatic Engineer Newsletter — AI Engineering Edition](https://newsletter.pragmaticengineer.com)** — Regular in-depth coverage of how companies are building and deploying AI systems.

---

## What to Read Next

- [Module 12: Capstone Projects](../module-12-capstone) — Apply production patterns to complete end-to-end systems. The Customer Support Automation and Autonomous Workflow Agent capstones both require production-quality architecture.

:::info See Also
For the practical LangGraph implementation of agent orchestration, see **[Module 4: LangGraph Agents](/learn/modules/module-4)** in the internal training track.
:::
