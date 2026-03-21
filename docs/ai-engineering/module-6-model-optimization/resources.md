---
sidebar_position: 4
title: "Resources"
description: Papers, tools, and further reading on model optimization, quantization, Flash Attention, and efficient inference.
---

# Resources: Model Optimization & Efficiency

## Papers & Research

- [Flash Attention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Dao et al. (2022). The original paper showing how tiling attention computation in SRAM eliminates the O(N²) memory bottleneck. Essential reading if you work with long context.

- [Flash Attention 2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Dao (2023). 2× faster than FA1, better multi-head parallelism. This is what most modern training runs use.

- [Speculative Decoding](https://arxiv.org/abs/2211.17192) — Leviathan et al. (2022). The paper that introduced the draft-verify approach for lossless 2–3× speedup. Short and clear.

- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) — Dettmers et al. (2022). The paper behind bitsandbytes INT8 quantization. Explains why naive INT8 fails and how mixed-precision decomposition fixes it.

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) — Frantar et al. (2022). One-shot weight quantization using the Hessian inverse. Fastest INT4 inference available.

- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) — Lin et al. (2023). Better quality than GPTQ at same bit depth for most models by identifying and protecting salient weights.

- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Jiang et al. (2024). The Mixtral 8×7B paper. Explains MoE routing, load balancing, and how MoE achieves large-model quality at small-model inference cost.

- [Knowledge Distillation: A Survey](https://arxiv.org/abs/2006.05525) — Gou et al. (2021). Comprehensive survey of distillation techniques from the pre-LLM era — the fundamental methods still apply.

## Official Documentation

- [vLLM](https://docs.vllm.ai) — The leading open-source LLM serving framework. Implements PagedAttention for efficient KV Cache management, dynamic batching, and speculative decoding. Start here for self-hosted serving.

- [bitsandbytes](https://huggingface.co/docs/bitsandbytes) — HuggingFace integration for INT8 and INT4 quantization. The `load_in_8bit=True` / `load_in_4bit=True` flags in `from_pretrained`.

- [HuggingFace PEFT (for QLoRA)](https://huggingface.co/docs/peft) — Also relevant for Module 7; QLoRA combines INT4 quantization with LoRA fine-tuning.

- [LMDeploy](https://lmdeploy.readthedocs.io) — Production inference framework from Shanghai AI Lab. Good alternative to vLLM with strong GPTQ support.

- [Ollama](https://ollama.com) — Simplest way to run quantized local models (Llama, Mistral, Phi, Gemma). Uses GGUF format with llama.cpp under the hood.

- [llama.cpp](https://github.com/ggerganov/llama.cpp) — C++ inference engine for quantized models. Runs on CPU with optional GPU offloading. GGUF format = the standard for local quantized inference.

- [tiktoken](https://github.com/openai/tiktoken) — OpenAI's fast BPE tokenizer. Use this to count tokens accurately for cost estimation and context budget tracking.

## Videos & Courses

- [Efficient LLM Inference — Stanford MLSys Seminar](https://www.youtube.com/watch?v=mykcd9Q5iKQ) — Technical deep dive on KV Cache, continuous batching, and PagedAttention from the vLLM team. Worth watching in full if you're building serving infrastructure.

- [Flash Attention Explained — Aleksa Gordić (AI Epiphany)](https://www.youtube.com/watch?v=IoMSGuiwV3g) — Visual walkthrough of the tiling algorithm and IO-awareness. Much easier to follow than the paper.

- [Quantization Explained — Andrej Karpathy](https://www.youtube.com/watch?v=__Gf5Dv0fXk) — Covers the math of INT8/INT4 quantization clearly, including why precision loss is non-uniform across layers.

- [LLM Inference Optimization — Tim Dettmers](https://www.youtube.com/watch?v=KAhRNuhFqLE) — The author of bitsandbytes explains the theory behind LLM.int8() and what "quantization error" actually means in practice.

## Tools to Know

| Tool | What it does | When to use |
|------|-------------|-------------|
| vLLM | Production LLM serving with PagedAttention + dynamic batching | Self-hosted serving at scale |
| bitsandbytes | Runtime INT8/INT4 quantization via HuggingFace | Experimentation and fine-tuning |
| GPTQ (AutoGPTQ) | Post-training quantization, offline | Fastest INT4 inference |
| AWQ (llm-awq) | Activation-aware quantization | Higher quality INT4 |
| Ollama | Run local quantized models with one command | Local dev, privacy-sensitive apps |
| llama.cpp | CPU-friendly quantized inference | No-GPU inference, edge deployment |
| LiteLLM | Unified interface for 100+ LLM providers with fallbacks | Multi-provider routing |
| Langfuse | Open-source LLM observability: traces, latency, cost | Production monitoring |
| Helicone | Proxy-based LLM observability for OpenAI/Anthropic | Quick setup observability |
| TGI (Text Generation Inference) | HuggingFace's serving framework, good Flash Attention support | HuggingFace ecosystem |

## What to Read Next

- [Module 7: Fine-Tuning & Adaptation](/learn/ai-engineering/module-7-fine-tuning) — Once you can serve models efficiently, the next question is whether to adapt them. Fine-tuning and quantization interact closely (QLoRA combines both).
- [Module 11: Production AI Systems](/learn/ai-engineering/module-11-production-systems) — Brings optimization together with caching, fallback chains, rate limiting, and observability into a complete production architecture.
