---
sidebar_position: 2
title: "Overview"
description: Theory and concepts — when to fine-tune vs prompt vs RAG, SFT, LoRA, QLoRA, DPO, catastrophic forgetting, and evaluation.
---

# Fine-Tuning & Adaptation — Deep Dive

Fine-tuning is one of the most misused tools in AI engineering. Teams fine-tune when they should be prompting. Teams prompt when they need to fine-tune. Understanding the decision boundary is more valuable than knowing the training API.

## The Decision Framework: Prompting vs RAG vs Fine-Tuning

Start here every time someone proposes fine-tuning.

```
Question 1: Does the model need knowledge it doesn't have?
  YES → Use RAG. Fine-tuning doesn't reliably inject factual knowledge
         and doesn't scale to dynamic or large knowledge bases.

Question 2: Does the model already have the knowledge but needs a new
            behavior, style, or format?
  YES → Consider fine-tuning.

Question 3: Is the base model already capable of the task with good prompting?
  YES → Use better prompting first. Fine-tuning has real costs:
        training compute, inference deployment, maintenance overhead.
```

### When Fine-Tuning Clearly Wins

| Scenario | Why fine-tuning wins |
|----------|---------------------|
| Domain jargon base model doesn't know | Medical codes, legal terms, company-internal terminology — RAG retrieves context but the model still can't *parse* it correctly |
| Consistent output format at scale | "Always return valid JSON with these exact fields" — baking this into weights eliminates format failures completely |
| Shrinking prompt length at high volume | A 2,000-token system prompt at 1M daily requests = $300/day on GPT-4o. Fine-tune a cheap model instead. |
| Latency-critical applications | Fine-tuned Mistral 7B + fast inference >> GPT-4o for latency-sensitive tasks where the 7B has been properly adapted |
| Privacy and data residency | Can't send proprietary documents to cloud APIs → fine-tune a local model |

### When Fine-Tuning Is the Wrong Tool

| Scenario | Better approach |
|----------|----------------|
| Model doesn't know recent facts | RAG — fine-tuning doesn't reliably memorize facts |
| Want model to cite sources | RAG with grounding instructions |
| Task works with GPT-4 but not GPT-4o-mini | Try better prompting with the smaller model first |
| Only have 10–20 examples | Few-shot prompting — not enough data for fine-tuning |
| Want to prevent specific outputs | Output filtering / guardrails — fine-tuning doesn't guarantee suppression |

:::tip
If you're considering fine-tuning, first run the task with a well-crafted prompt for 100 examples and measure quality. If quality is above your threshold, you don't need to fine-tune. If quality is consistently below and prompting improvements have plateaued, then fine-tuning is the next step.
:::

---

## Supervised Fine-Tuning (SFT): Curating the Dataset

SFT means training a pretrained model on (input, ideal_output) pairs using the standard cross-entropy loss. The model learns to reproduce the ideal outputs given those inputs.

```python
# The fundamental data format for SFT (ChatML / Alpaca style)
training_example = {
    "messages": [
        {"role": "system", "content": "You are a JSON extraction assistant. Output only valid JSON."},
        {"role": "user", "content": "Extract the entity names: 'Apple and Microsoft announced a partnership.'"},
        {"role": "assistant", "content": '{"entities": ["Apple", "Microsoft"]}'},
    ]
}
```

### Dataset Quality Is Everything

The single most important variable in SFT is dataset quality, not quantity.

```
500 high-quality, diverse, well-formatted examples
  > 50,000 noisy, inconsistent, or mislabeled examples

Why:
- Model learns the distribution of your data
- Noise teaches the model to output noise
- Inconsistency teaches the model to be inconsistent
- Quality teaches the model to be high quality
```

**What "high quality" means in practice:**
- Every output is exactly what you want the model to produce
- Consistent format across all examples (if JSON, always valid JSON)
- Coverage across your input distribution (don't just sample easy cases)
- Edge cases included: short inputs, long inputs, ambiguous inputs, failure cases

**Dataset size guidelines:**
| Task type | Minimum examples | Target examples |
|-----------|-----------------|-----------------|
| Style / tone change | 100–200 | 500–1K |
| Output format (JSON, code) | 200–500 | 1K–5K |
| Domain adaptation (medical, legal) | 500–2K | 5K–50K |
| Full behavior change | 2K+ | 10K–100K |

---

## Parameter-Efficient Fine-Tuning (PEFT)

Full fine-tuning updates all model weights. For a 7B parameter model, that means storing gradients and optimizer states for 7B parameters — roughly 84GB for AdamW in FP32. This is prohibitive for most teams.

PEFT methods fine-tune a tiny fraction of parameters while keeping the base model frozen.

### LoRA: Low-Rank Adaptation

LoRA is the most widely used PEFT method. It works by decomposing the weight update into two small matrices:

```
Full fine-tuning:
  W' = W + ΔW     where ΔW is [d × d], same size as W

LoRA decomposition:
  ΔW ≈ A · B      where A is [d × r] and B is [r × d], r << d

Example with r=8 and d=1000:
  Full ΔW: 1000 × 1000 = 1,000,000 parameters
  LoRA A:  1000 × 8    =     8,000 parameters
  LoRA B:     8 × 1000 =     8,000 parameters
  Total LoRA: 16,000 parameters (1.6% of full ΔW)

At inference:
  W_effective = W + α/r · A · B
  Where α is a scaling hyperparameter (often set to 2r or r)
```

**Which layers to apply LoRA to?**
LoRA is typically applied to the attention projection matrices (Q, K, V, and output projection). The query and value matrices are most impactful:

```python
# In HuggingFace PEFT
peft_config = LoraConfig(
    r=16,                        # Rank — higher = more capacity, more params
    lora_alpha=32,               # Scaling: effective_alpha = lora_alpha / r
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.05,           # Regularization
    bias="none",
    task_type="CAUSAL_LM",
)
```

**Rank selection guide:**
| Rank (r) | Trainable params (7B model) | Use when |
|----------|---------------------------|----------|
| 4 | ~1M | Minimal adaptation, style only |
| 8 | ~2M | Default starting point |
| 16 | ~4M | More complex behavior changes |
| 64 | ~16M | Domain-heavy adaptation |
| 128+ | ~32M+ | Rarely needed; consider full fine-tune |

:::note
Higher rank is not always better. Start with r=8 or r=16, evaluate, then increase only if task metrics don't converge.
:::

### QLoRA: Fine-Tuning 7B Models on a Consumer GPU

QLoRA combines two ideas:
1. Load the base model in 4-bit NF4 (Normal Float 4-bit) quantization
2. Apply LoRA adapters in full precision on top of the frozen quantized base

```
QLoRA memory usage for a 7B model:
  Base model (4-bit NF4): ~3.5 GB
  LoRA adapters (FP16):   ~0.1 GB
  Gradient checkpointing: ~2 GB
  Total:                  ~6–7 GB

→ Fits on a single RTX 3090/4090 (24GB) or Google Colab T4 (16GB)

Without QLoRA:
  Base model (FP16):      ~14 GB
  Gradients + optimizer:  ~42 GB
  Total:                  ~56 GB → requires A100 80GB
```

**The NF4 data type:** Unlike INT4 which uniformly quantizes a range, NF4 is designed specifically for normally-distributed neural network weights. It places more quantization levels near zero (where most weights cluster) and fewer at the extremes, reducing quantization error.

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  # Additional quantization on the quantization constants
    bnb_4bit_quant_type="nf4",       # Normal Float 4-bit
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in BF16 despite storage in 4-bit
)
```

---

## Reward Models, RLHF, and DPO

After SFT, LLMs can follow instructions but don't necessarily give the *best* answer — they give a *plausible* answer. RLHF and DPO are methods to push the model toward outputs humans prefer.

### RLHF (Reinforcement Learning from Human Feedback)

```
Step 1: SFT the base model on high-quality examples
Step 2: Collect human preference data — show two outputs, human picks better one
Step 3: Train a Reward Model (RM) to predict which output humans prefer
Step 4: Use PPO (a RL algorithm) to fine-tune the SFT model to maximize RM score
         while staying close to the SFT model (KL penalty)
```

**Problems with RLHF:**
- PPO is unstable and sensitive to hyperparameters
- Reward model can be "hacked" — model finds outputs that score high but aren't actually good
- Requires a separate reward model with its own training pipeline
- Complex to implement correctly

### DPO (Direct Preference Optimization)

DPO achieves the same goal as RLHF without training a separate reward model. It directly optimizes the policy on preference pairs using a clever mathematical reformulation.

```python
# DPO training data format
dpo_example = {
    "prompt": "Explain how vaccines work:",
    "chosen": "Vaccines introduce weakened or inactivated pathogens...",  # preferred
    "rejected": "Vaccines are injections that prevent diseases...",         # less preferred
}
```

DPO is:
- Simpler to implement than RLHF (no RL loop, no separate reward model)
- More stable in training
- Used by models like Zephyr, Tulu 2, and many open-source aligned models

For most teams doing fine-tuning in 2024+, DPO is the practical alternative to RLHF.

---

## Catastrophic Forgetting

When you fine-tune a model on a narrow task, it may "forget" general capabilities it had before. This is called catastrophic forgetting — the new gradient updates overwrite representations used for other tasks.

```
Before fine-tuning — model can:
  - Write Python code        ✓
  - Answer general questions ✓
  - Translate text           ✓
  - Your specialized task    ✗ (poor)

After naive fine-tuning on specialized task:
  - Write Python code        ? (might be degraded)
  - Answer general questions ? (might be worse)
  - Translate text           ? (might be broken)
  - Your specialized task    ✓ (good)
```

**Mitigations:**
1. **LoRA (the best mitigation):** Because LoRA keeps the base weights frozen and only trains the adapters, the base model's general capabilities are preserved. This is one of the strongest arguments for PEFT over full fine-tuning.

2. **Include general data in your training mix:** Mix 5–10% general instruction-following data into your task-specific data.

3. **Evaluate before and after on general benchmarks:** Run a standard benchmark (MMLU, HellaSwag, or a simple held-out general QA set) before and after fine-tuning to detect degradation.

4. **Use a smaller learning rate:** Large learning rates cause more forgetting. Start at 1e-4 for LoRA, 1e-5 for full fine-tuning.

---

## Evaluation Strategy

Fine-tuning without rigorous evaluation is dangerous — you might ship a model that performs well on your training distribution but breaks silently on real inputs.

### The Held-Out Test Set (Non-Negotiable)

Never evaluate on your training data. Create a test split *before* you start training:

```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="my_data.jsonl")["train"]

# 80% train, 10% validation (tune hyperparameters), 10% test (final evaluation only)
split = dataset.train_test_split(test_size=0.2, seed=42)
train_val = split["train"].train_test_split(test_size=0.125, seed=42)

train_set = train_val["train"]
val_set   = train_val["test"]
test_set  = split["test"]  # HOLD THIS OUT. Do not touch until final eval.
```

### Task-Specific Metrics

| Task | Primary metric | Secondary metric |
|------|---------------|-----------------|
| Classification | F1 score | Precision, Recall |
| JSON extraction | Exact match or field-level F1 | Schema validity rate |
| Code generation | Pass@k (% that pass unit tests) | Compilation rate |
| Summarization | LLM-as-judge score | ROUGE (weak signal) |
| Instruction following | Instruction compliance rate (judge) | Preference score (DPO eval) |

### Capability Degradation Check

```python
# Before fine-tuning — run these and record scores
general_benchmarks = [
    "What is the capital of Japan?",  # factual
    "Write a Python quicksort.",       # code
    "Translate 'hello' to Spanish.",   # translation
    # Add 20–50 diverse prompts covering capabilities you care about
]

# After fine-tuning — run the same prompts
# Manually or with LLM-as-judge, score degradation
```

---

## Mental Model: What Fine-Tuning Actually Changes

Fine-tuning modifies the probability distribution the model assigns to tokens given a context. It does not:
- Add new factual knowledge reliably (use RAG for that)
- Make the model "smarter" (it can't reason better than it already could)
- Guarantee specific outputs (only shifts probabilities)

It does:
- Shift output style toward your training distribution
- Reduce format errors if your training data is consistently formatted
- Improve task accuracy when the base model was already close but inconsistent
- Reduce prompt length needed at inference time

---

## Common Mistakes

| Mistake | Why it happens | Fix |
|---------|---------------|-----|
| Fine-tuning to inject knowledge | Intuitive but wrong — weights don't reliably store facts | Use RAG for knowledge, fine-tune for behavior |
| Using noisy, crowd-sourced data | Cheap to collect | Curate carefully — 500 clean beats 10K dirty |
| Not checking for catastrophic forgetting | Easy to skip | Always run general capability eval before and after |
| Overfitting on a small dataset | Training too many epochs | Validate on held-out set, stop when val loss stops improving |
| Skipping the eval-before-training step | "I know this model works" | Always establish a baseline to measure improvement against |
| Setting rank too high initially | "More capacity = better" | Start with r=8, increase only if needed |

---

## Quiz

> **Q: Your team wants to build a customer support bot for a fintech company. The bot needs to: (1) know your company's specific product features, (2) always respond in a formal tone, and (3) output responses in a structured JSON format for downstream processing. Which combination of techniques would you use, and why?**
>
> <details><summary>Show Answer</summary>
>
> **Knowledge (product features):** RAG. Product information changes frequently and is too large to bake into weights reliably. Retrieve from a vector database of your docs.
>
> **Tone (formal style):** Fine-tuning (SFT or LoRA). Consistent tone is a behavioral property that prompting can achieve but fine-tuning makes robust and reduces the system prompt length needed. Curate ~500 examples in your desired tone.
>
> **Format (structured JSON):** Fine-tuning + output validation. Fine-tuning on JSON-formatted training examples dramatically reduces format errors. Add a JSON schema validator at the output layer as a safety net.
>
> **Implementation order:** Start with RAG + prompting (fastest to deploy). Measure format failure rate and tone consistency. If format failures exceed 2% or prompt length is getting expensive, add fine-tuning.
>
> </details>

---

## Summary Table

| Concept | What it is | When to use |
|---------|-----------|------------|
| SFT | Training on (input, output) pairs | Baseline adaptation for any task |
| LoRA | Low-rank adapters, base model frozen | Standard PEFT — use by default |
| QLoRA | LoRA + 4-bit base model | Fine-tuning 7B+ models on consumer GPU |
| DPO | Preference optimization without reward model | Aligning behavior, replacing RLHF |
| RLHF | RL from human preferences via reward model | Full alignment pipelines (complex) |
| Catastrophic forgetting | General capability loss after fine-tuning | Risk to manage — LoRA mitigates it |

## Next Steps

→ [Hands-On: Fine-Tuning & Adaptation](./hands-on)
