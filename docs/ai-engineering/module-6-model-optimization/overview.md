---
sidebar_position: 2
title: "Overview"
description: Theory and concepts — KV Cache, batching, streaming, quantization, distillation, SLMs, MoE, Flash Attention, and speculative decoding.
---

# Model Optimization & Efficiency — Deep Dive

Every millisecond and every token costs money. When you serve 10,000 requests a day, the difference between a naive deployment and an optimized one can be 3× in cost and 5× in latency — with identical model quality. This overview gives you the mental models to make those gains systematically.

## KV Cache: The Single Biggest Free Win

When an LLM generates a response, it runs attention over every token in the context at every generation step. Without caching, a 2,000-token context at step 500 means recomputing attention over 2,500 tokens — redundantly. The KV Cache eliminates this by storing the computed Key and Value matrices from prior steps.

```
Step 1: Compute K, V for tokens [1..N]. Store them.
Step 2: Only compute K, V for the NEW token. Reuse cached [1..N].
Step 3: Same. Append to cache.

Without KV Cache: O(T²) compute per sequence
With KV Cache:    O(T) compute per sequence (amortized)
```

**What gets cached:** The K and V matrices for every layer and every prior token in the current sequence.

**When the cache is invalidated:**
- New request starts (cache is per-sequence, not global by default)
- The prompt changes (even one token difference = cache miss on most systems)
- Batch size changes (some systems invalidate on rebatch)
- Context length exceeds configured cache budget

**Memory cost:** A 70B model with FP16 KV Cache at 4K context uses roughly:
```
2 (K + V) × num_layers × num_heads × head_dim × seq_len × 2 bytes
= 2 × 80 × 64 × 128 × 4096 × 2 ≈ 10.7 GB per sequence
```
This is why serving many concurrent long-context requests is memory-bound, not compute-bound.

:::tip
Prefix caching (available in vLLM, SGLang, and some API providers) extends KV Cache across requests that share the same system prompt. If you have a fixed system prompt, this can cut your first-token latency by 60%+ for repeat users.
:::

---

## Batching: Throughput vs Latency

A single forward pass through a transformer costs roughly the same whether you process 1 sequence or 32. GPUs are massively parallel — they want large batch sizes to amortize fixed overhead.

**Static batching:** Wait until you have a full batch of N requests, then process together.
- Pro: maximum GPU utilization, lowest cost per token
- Con: the first request in a batch waits for N−1 others to arrive before processing starts

**Dynamic (continuous) batching:** Process requests as they arrive, inserting new sequences mid-generation. Used by vLLM, TGI, and other production inference servers.
- Pro: eliminates the wait — latency is close to single-request latency
- Con: slightly lower peak throughput vs pure static batching

| Batching Strategy | Latency | Throughput | GPU Utilization | Complexity |
|-------------------|---------|------------|-----------------|------------|
| No batching | Lowest | Worst | ~10–20% | None |
| Static batching | Highest | Best | ~80–95% | Low |
| Dynamic batching | Medium | Good | ~60–80% | High |

:::note
For user-facing features, dynamic batching is almost always the right choice. Static batching makes sense for offline batch inference jobs (generating embeddings for a corpus, overnight report generation).
:::

---

## Streaming: Perceived vs Actual Latency

Streaming does not make your model faster. It makes it *feel* faster because the user sees the first token in ~200ms rather than waiting 8 seconds for the complete response.

```
Without streaming:
[user waits 8 seconds] → receives full 400-token response at once

With streaming:
[user sees token 1 at 200ms] → [token 2 at 250ms] → ... → [token 400 at 8s]
```

The critical metric split:
- **Time to First Token (TTFT):** latency before anything appears — minimize this
- **Time Per Output Token (TPOT):** how fast tokens arrive after the first — affects reading comfort
- **Total generation time:** sum of both — the old single metric, less meaningful for UX

Streaming also enables early termination: if a user sees the answer forming incorrectly, they can stop and retry rather than waiting for the full response.

---

## Quantization: Shrinking Models Without Destroying Them

Weights in a neural network are floating-point numbers. The default is FP32 (32 bits per weight). Quantization reduces this precision to save memory and speed up matrix multiplications.

```
Precision ladder:
FP32 (32-bit): Full precision. Baseline.
FP16 (16-bit): Half precision. 2× smaller. Negligible quality loss for inference.
BF16 (16-bit): Brain float. Same size as FP16, better range. Preferred for training.
INT8  (8-bit): 4× smaller than FP32. Small quality loss. Widely supported.
INT4  (4-bit): 8× smaller than FP32. Noticeable loss on small models. OK on 7B+.
```

**Real size example — a 70B parameter model:**

| Precision | Size | Can run on |
|-----------|------|------------|
| FP32 | ~280 GB | 4× A100 80GB |
| FP16 | ~140 GB | 2× A100 80GB |
| INT8 | ~70 GB | 1× A100 80GB |
| INT4 | ~35 GB | 2× RTX 4090 (consumer) |

**How quantization works (simplified):**
```
Original weights: [-1.23, 0.45, 2.89, -0.11]  (FP32, 4 bytes each)

INT8 quantization:
  1. Find min = -1.23, max = 2.89
  2. Scale = (max - min) / 255 = 0.0165
  3. Zero point = round(-min / scale) = 75
  4. Quantized = round(weight / scale) + zero_point
     → [0, 102, 255, 68]  (1 byte each)

At inference: dequantize → (quantized - zero_point) × scale
```

**When quantization hurts:**
- Very small models (< 3B params): precision matters more — fewer redundant weights to absorb rounding error
- Reasoning-heavy tasks: arithmetic and multi-step logic are sensitive to weight precision
- First layers and last layers: more critical to overall quality

### GPTQ, AWQ, and bitsandbytes

Three popular quantization approaches for production use:
- **bitsandbytes:** runtime quantization, easy to apply with HuggingFace, best for experimentation
- **GPTQ:** quantize-once offline using calibration data, faster inference than bitsandbytes
- **AWQ (Activation-aware Weight Quantization):** identifies which weights matter most using activations, higher quality at same bit depth

---

## Knowledge Distillation: Teaching Small Models Big Tricks

Fine-tuning a small model on task labels gives you a model that matches task outputs. Distillation gives you a model that matches the full reasoning process — including how confident the large model is across all possible outputs.

```
Standard fine-tuning:
  Input → Small model → Output
  Loss = CrossEntropy(output, ground_truth_label)

Distillation:
  Input → Teacher (large model) → Soft probability distribution over all tokens
  Input → Student (small model) → Its own distribution
  Loss = KL divergence(student_distribution, teacher_distribution)
```

The soft distribution from the teacher contains more signal than a hard label. If the teacher gives "cat" 60% probability and "dog" 35%, the student learns that these two are semantically close — a one-hot label "cat" doesn't teach that.

**When to distill:**
- You have a high-quality large model already solving the task
- You need lower latency or lower cost in production
- You can afford the compute to generate teacher outputs

---

## Small Language Models (SLMs): When Smaller Wins

The assumption that bigger = better is often wrong for production deployments.

| Model | Params | Context | Strengths |
|-------|--------|---------|-----------|
| Phi-3 Mini | 3.8B | 128K | Reasoning, code — remarkable for its size |
| Gemma 2 2B | 2B | 8K | General instruction following, fast inference |
| Gemma 2 9B | 9B | 8K | Strong across benchmarks, open weights |
| Mistral 7B | 7B | 32K | Instruction following, best-in-class at release |
| Llama 3.1 8B | 8B | 128K | General purpose, strong open-source baseline |

**SLMs beat large models when:**
- The task is well-defined and narrow (classification, extraction, formatting)
- You fine-tune the SLM on your specific distribution
- Latency requirements are tight (< 500ms)
- Privacy requirements prevent cloud API calls (run locally)

**Cost comparison (approximate):**
```
GPT-4o:      $5.00 / 1M input tokens + $15.00 / 1M output tokens
GPT-4o-mini: $0.15 / 1M input tokens + $0.60 / 1M output tokens
Self-hosted Mistral 7B INT4 on RTX 4090: ~$0.002 / 1M tokens (amortized GPU cost)
```

---

## Mixture of Experts (MoE): Conditional Computation

A standard transformer activates all parameters for every token. MoE changes this: only a small subset of "expert" sub-networks activate per token, chosen by a learned router.

```
Dense model (e.g., Llama 7B):
  Token → All 7B parameters activate → Output

MoE model (e.g., Mixtral 8×7B):
  Token → Router selects 2 of 8 experts (each ~7B params)
         → Only ~2×7B = 14B parameters activate
         → But model has 8×7B = 56B total params
```

The result: Mixtral 8×7B has the quality of a ~56B parameter model at the inference cost of a ~14B model.

**Tradeoff:** All 56B parameters must fit in memory even though only 14B activate at once. MoE is memory-heavy but compute-efficient.

---

## Flash Attention: Fixing the Memory Bottleneck

Standard attention computes the full N×N attention matrix and materializes it in GPU HBM (high-bandwidth memory):

```
Standard attention memory: O(N²) — for N=8K context, that's 64M floats
```

Flash Attention avoids materializing the full matrix by tiling the computation: it processes the attention calculation in blocks that fit in the GPU's fast SRAM (on-chip cache), accumulating the result without writing the full matrix to slower HBM.

```
Flash Attention memory: O(N) — linear in sequence length
Speed improvement: 2–4× for long contexts
```

**Practical impact:**
- Flash Attention 2 (2023): 2× faster than FA1, used in most modern training frameworks
- Flash Attention 3 (2024): optimized for H100 GPUs, up to 75% of H100 FP8 theoretical peak
- Without FA: 32K context window is barely feasible; with FA: 128K+ becomes practical

---

## Speculative Decoding: Parallel Verification

LLM generation is sequential — you can't generate token 5 until you have tokens 1–4. This sequential bottleneck limits throughput. Speculative decoding breaks it with a two-model approach:

```
Step 1: Draft model (small, fast) generates K candidate tokens in parallel
Step 2: Large model verifies all K candidates in one forward pass
Step 3: Accept tokens up to the first rejection, discard the rest
Step 4: Repeat

If the draft model agrees with the large model 80% of the time:
→ Average accepted tokens per large-model pass ≈ 3–4
→ Effective speedup: 2–3× vs unaccelerated generation
```

The key insight: verifying K tokens takes the same compute as generating 1 token (the large model processes the whole sequence in parallel). You only pay extra for the draft model, which is tiny.

**Common pairings:**
- GPT-4 family: internal speculative decoding with undisclosed draft model
- Gemma 27B + Gemma 2B as draft
- Llama 3.1 70B + Llama 3.1 8B as draft

---

## Cost Optimization Decision Tree

```
Is quality good enough with a smaller model?
  YES → Use SLM (Phi-3, Gemma 2B, Mistral 7B)
          Fine-tune it on your distribution for best results
  NO  → Are you hitting latency limits?
          YES → Enable KV caching + streaming
                Add speculative decoding if self-hosting
                Use dynamic batching (vLLM/TGI)
          NO  → Is cost the primary concern?
                  YES → Quantize: INT8 first, INT4 if acceptable quality
                        Route simple sub-tasks to cheaper models
                  NO  → Use full model, optimize prompts instead
                          (Shorter prompts = cheaper, not just prompt engineering)
```

---

## Common Mistakes

| Mistake | Why it happens | Fix |
|---------|---------------|-----|
| Using GPT-4 for every task | Easier to set up, "just works" | Route simple tasks to cheaper/smaller models |
| Not streaming user-facing features | Default API behavior is non-streaming | Always enable streaming for interactive UX |
| Ignoring quantization until too late | Works fine in dev, breaks on budget | Plan quantization strategy at architecture stage |
| Invalidating KV Cache unnecessarily | Prepending dynamic timestamps to system prompt | Put dynamic content at the end of the prompt, not the beginning |
| Batching with static sizes | Simpler code | Use dynamic batching — the latency penalty of static batching is severe |

---

## Quiz

> **Q: You have a 70B model serving 1,000 concurrent users. Your p95 latency is 12 seconds and your GPU memory is nearly full. Which optimization should you try first, and why?**
>
> <details><summary>Show Answer</summary>
>
> Start with **INT8 quantization**. It halves your memory footprint (140GB → 70GB for a 70B FP16 model), allowing more concurrent sequences in memory and more effective batching. This directly addresses both the memory pressure and the latency — more batching → better GPU utilization → higher throughput. Streaming and KV caching are already likely in place; speculative decoding requires a second model and adds complexity. Quantization gives the largest return with the least architectural change.
>
> If INT8 alone doesn't solve it, add **dynamic batching** (vLLM or TGI) to maximize GPU utilization across those concurrent users.
>
> </details>

---

## Summary Table

| Technique | What it does | Primary benefit | When to apply |
|-----------|-------------|-----------------|---------------|
| KV Cache | Cache prior token K/V matrices | Reduces per-step compute from O(T²) to O(T) | Always — it's automatic in most frameworks |
| Prefix caching | Share KV cache across requests with same prefix | Cuts TTFT for repeat system prompts | Fixed system prompts at scale |
| Dynamic batching | Process requests continuously, not in fixed batches | Low latency + high throughput | Production serving (vLLM, TGI) |
| Streaming | Send tokens as generated | Reduces perceived latency | All user-facing interfaces |
| Quantization | Reduce weight precision | Smaller memory footprint, faster matmul | When memory or cost is constrained |
| Distillation | Train small model to mimic large model | Production-grade small model | When you have labeled teacher outputs |
| SLMs | Use purpose-built small model | Lower cost + latency | Narrow, well-defined tasks |
| Flash Attention | Tile attention computation | Memory-efficient long context | Long context (>8K) + self-hosted models |
| Speculative decoding | Draft + verify in parallel | 2–3× generation speedup | High-throughput self-hosted serving |

## Next Steps

→ [Hands-On: Model Optimization](./hands-on)
