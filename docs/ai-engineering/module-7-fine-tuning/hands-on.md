---
sidebar_position: 3
title: "Hands-On"
description: Practical exercises — fine-tune GPT-2, apply LoRA, attempt QLoRA on a 7B model, and run before/after evaluation.
---

# Hands-On: Fine-Tuning & Adaptation

These exercises progress from a simple full fine-tune of a tiny model (Exercise 1) to parameter-efficient methods (Exercises 2–3) to rigorous evaluation (Exercise 4). Exercises 1 and 2 run on CPU or a small GPU. Exercise 3 needs 16GB+ VRAM (Google Colab T4 works). Exercise 4 runs anywhere.

**Setup for all exercises:**
```bash
pip install transformers datasets peft trl accelerate bitsandbytes evaluate torch
```

---

## Exercise 1: Full Fine-Tune GPT-2 on a Custom Dataset (Beginner)

**Goal:** Understand the full fine-tuning loop — dataset loading, tokenization, training, and inference — using a small model.

**Time:** ~40 min

**Step 1 — Prepare a custom instruction dataset:**
```python
import json

# We'll fine-tune GPT-2 to respond in a specific Q&A format about AI concepts.
# In production, you'd have hundreds of these — we're using 50 to see the mechanics.
training_data = [
    {"prompt": "What is a token in LLMs?", "completion": "A token is the basic unit of text processed by a language model. Common words are single tokens; uncommon words may be split into multiple tokens. GPT-4 uses roughly 0.75 tokens per word."},
    {"prompt": "What does temperature control in LLMs?", "completion": "Temperature controls randomness in token selection. Temperature 0 selects the highest-probability token deterministically. Temperature 1 samples proportionally to the model's distribution. Higher values increase diversity; lower values increase consistency."},
    {"prompt": "What is RAG?", "completion": "Retrieval-Augmented Generation (RAG) combines a retrieval system with a language model. The retrieval system finds relevant documents from a knowledge base; these documents are injected into the prompt so the model can generate factually grounded responses."},
    # ... add more examples covering diverse AI concepts
]

# Format for GPT-2 (no chat template — just concatenate prompt + completion)
formatted = []
for item in training_data:
    text = f"### Question: {item['prompt']}\n### Answer: {item['completion']}\n\n"
    formatted.append({"text": text})

# Save as JSONL
with open("ai_qa_dataset.jsonl", "w") as f:
    for item in formatted:
        f.write(json.dumps(item) + "\n")
```

**Step 2 — Load and tokenize:**
```python
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token by default

dataset = load_dataset("json", data_files="ai_qa_dataset.jsonl")["train"]

def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
tokenized = tokenized.train_test_split(test_size=0.1, seed=42)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
```

**Step 3 — Load model and configure training:**
```python
from transformers import TrainingArguments, Trainer
import torch

model = GPT2LMHeadModel.from_pretrained("gpt2")

# Count parameters before training
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Trainable %:          {100 * trainable_params / total_params:.1f}%")

training_args = TrainingArguments(
    output_dir="./gpt2-ai-qa",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=5e-5,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    data_collator=data_collator,
)
```

**Step 4 — Train and compare before/after:**
```python
# Test BEFORE fine-tuning
def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 100) -> str:
    formatted_prompt = f"### Question: {prompt}\n### Answer:"
    inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_output[len(formatted_prompt):]

test_prompt = "What is a transformer in machine learning?"
print("BEFORE training:")
print(generate_response(model, tokenizer, test_prompt))

# Train
print("\nTraining...")
trainer.train()

print("\nAFTER training:")
print(generate_response(model, tokenizer, test_prompt))
```

**What to observe:** GPT-2 is too small to genuinely understand these concepts, but you'll see it learn the Q&A *format* reliably. This illustrates the key SFT insight: the model learns the distribution of your data.

---

## Exercise 2: Apply LoRA — Compare Trainable Parameter Counts (Intermediate)

**Goal:** Apply LoRA to a model, verify the parameter reduction, and train on the same dataset.

**Time:** ~35 min

**Step 1 — Load a larger base model:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

model_id = "facebook/opt-1.3b"  # 1.3B params — small enough for CPU/small GPU

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

# Baseline: all parameters
total = sum(p.numel() for p in model.parameters())
print(f"Base model parameters: {total:,}")
```

**Step 2 — Apply LoRA configuration:**
```python
lora_config = LoraConfig(
    r=8,                          # Rank
    lora_alpha=16,                # Scaling: alpha/r = 2.0
    target_modules=["q_proj", "v_proj"],  # Apply to attention Q and V
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

peft_model = get_peft_model(model, lora_config)

# Count trainable parameters
trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
total_peft = sum(p.numel() for p in peft_model.parameters())

print(f"\nWith LoRA (r=8, target=['q_proj', 'v_proj']):")
print(f"  Total parameters:     {total_peft:,}")
print(f"  Trainable parameters: {trainable:,}")
print(f"  Trainable %:          {100 * trainable / total_peft:.2f}%")
peft_model.print_trainable_parameters()
```

**Step 3 — Compare different rank values:**
```python
from peft import LoraConfig, get_peft_model

def count_trainable_params(model_id: str, rank: int, target_modules: list) -> dict:
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    config = LoraConfig(r=rank, lora_alpha=rank*2, target_modules=target_modules, bias="none", task_type=TaskType.CAUSAL_LM)
    peft_model = get_peft_model(model, config)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    del peft_model, model
    return {"rank": rank, "trainable": trainable, "pct": round(100 * trainable / total, 3)}

targets_narrow = ["q_proj", "v_proj"]
targets_wide = ["q_proj", "k_proj", "v_proj", "out_proj"]

print(f"{'Rank':<8} {'Targets':<35} {'Trainable params':>20} {'%':>8}")
print("-" * 75)
for r in [4, 8, 16, 64]:
    for targets in [targets_narrow, targets_wide]:
        result = count_trainable_params(model_id, r, targets)
        target_str = "+".join(t.split("_")[0] for t in targets)
        print(f"r={r:<6} {target_str:<35} {result['trainable']:>20,} {result['pct']:>7.3f}%")
```

**Step 4 — Train the LoRA model:**
```python
from trl import SFTTrainer

# Reuse the dataset from Exercise 1 (or create new examples)
training_args = TrainingArguments(
    output_dir="./opt-lora-ai-qa",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # Effective batch size = 8
    warmup_steps=10,
    learning_rate=3e-4,             # LoRA uses higher LR than full fine-tuning
    fp16=torch.cuda.is_available(),
    logging_steps=10,
    evaluation_strategy="epoch",
)

trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    dataset_text_field="text",
    max_seq_length=512,
)

trainer.train()

# Save only the LoRA adapters (tiny — typically 5–50 MB vs the full 2.6 GB model)
peft_model.save_pretrained("./opt-lora-adapters")
print("\nAdapter files saved:")
import os
for f in os.listdir("./opt-lora-adapters"):
    size_mb = os.path.getsize(f"./opt-lora-adapters/{f}") / 1024**2
    print(f"  {f}: {size_mb:.2f} MB")
```

---

## Exercise 3: QLoRA on a 7B Model (GPU Required) (Intermediate)

**Goal:** Fine-tune Mistral-7B or Llama-3.1-8B using QLoRA on a T4 GPU.

**Time:** ~60 min (including setup)

**Recommended environment:** Google Colab (T4 GPU, 16GB VRAM)

**Step 1 — Setup and model loading:**
```python
# In Colab: !pip install transformers peft trl accelerate bitsandbytes datasets

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
# Alternative if you hit rate limits: "meta-llama/Llama-3.1-8B-Instruct"

# 4-bit quantization config (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading model in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

print(f"GPU memory after loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
```

**Step 2 — Prepare model for QLoRA training:**
```python
# This cast ensures trainable layers (LoRA) are in float32 while the base stays in 4-bit
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],  # Include MLP for stronger adaptation
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected output: ~0.5–1% of total parameters
```

**Step 3 — Prepare an instruction dataset:**
```python
from datasets import load_dataset

# Use a small public instruction dataset for demonstration
# Replace with your own domain data in production
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:2000]")

def format_chat(example):
    """Format to Mistral instruction format."""
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

formatted_dataset = dataset.map(format_chat)
split = formatted_dataset.train_test_split(test_size=0.05, seed=42)
```

**Step 4 — Train with SFTTrainer:**
```python
from trl import SFTTrainer
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./mistral-qlora",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,       # Save memory: recompute activations during backprop
    optim="paged_adamw_32bit",        # Memory-efficient optimizer for QLoRA
    save_steps=50,
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=True,                         # BF16 for computation on modern GPUs
    max_grad_norm=0.3,                 # Gradient clipping
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="none",
    evaluation_strategy="steps",
    eval_steps=50,
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=split["train"],
    eval_dataset=split["test"],
    dataset_text_field="text",
    max_seq_length=1024,
    tokenizer=tokenizer,
)

print("Starting QLoRA training...")
trainer.train()

# Save adapters
model.save_pretrained("./mistral-qlora-adapters")
tokenizer.save_pretrained("./mistral-qlora-adapters")
print("Training complete. Adapters saved.")
```

**Step 5 — Inference with the fine-tuned model:**
```python
from peft import PeftModel

# Load base model + adapters
base = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
ft_model = PeftModel.from_pretrained(base, "./mistral-qlora-adapters")

def chat(model, tokenizer, user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    inputs = inputs.to(model.device)
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=256, temperature=0.1, do_sample=True)
    return tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

test_message = "Explain the difference between supervised and unsupervised learning."
print(chat(ft_model, tokenizer, test_message))
```

---

## Exercise 4: Before/After Evaluation — Measure Task Accuracy and Capability Degradation (Intermediate)

**Goal:** Run systematic evaluation before and after fine-tuning to measure gain on task and check for general capability loss.

**Time:** ~30 min

**Step 1 — Define your evaluation sets:**
```python
# Task-specific eval (what you're fine-tuning for)
task_eval = [
    {
        "input": "Extract company name: 'Google announced Gemini 2.0 today.'",
        "expected": "Google",
    },
    {
        "input": "Extract company name: 'Tesla reported record deliveries in Q4.'",
        "expected": "Tesla",
    },
    # ... 20+ examples
]

# General capability eval (catastrophic forgetting check)
general_eval = [
    {
        "input": "What is the capital of France?",
        "expected_contains": "Paris",
    },
    {
        "input": "Write a Python function that returns the factorial of n.",
        "expected_contains": "def",
    },
    {
        "input": "What is 17 × 13?",
        "expected_contains": "221",
    },
    # ... 20+ diverse examples
]
```

**Step 2 — Automated evaluation function:**
```python
from openai import OpenAI
import json

client = OpenAI()

def evaluate_with_judge(model_output: str, expected: str, context: str) -> dict:
    """Use GPT-4o-mini as an automated judge."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Evaluate if the model output correctly answers the task.

Task input: {context}
Expected: {expected}
Model output: {model_output}

Respond with JSON only: {{"correct": true/false, "reason": "brief explanation"}}"""
        }],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)

def run_eval_suite(model_fn, eval_set: list, suite_name: str) -> dict:
    """Run a model function against an eval suite and return accuracy."""
    correct = 0
    results = []
    for item in eval_set:
        output = model_fn(item["input"])
        expected = item.get("expected") or item.get("expected_contains")
        judgment = evaluate_with_judge(output, expected, item["input"])
        if judgment["correct"]:
            correct += 1
        results.append({
            "input": item["input"],
            "output": output,
            "correct": judgment["correct"],
            "reason": judgment["reason"],
        })

    accuracy = correct / len(eval_set)
    print(f"\n{suite_name}: {correct}/{len(eval_set)} correct ({accuracy:.1%})")
    return {"accuracy": accuracy, "results": results}
```

**Step 3 — Compare before and after:**
```python
# Define model wrappers for before/after comparison
def before_model(prompt: str) -> str:
    """Call base model (or record outputs before training)."""
    # Replace with your actual before-training model call
    return generate_response(base_model, tokenizer, prompt)

def after_model(prompt: str) -> str:
    """Call fine-tuned model."""
    return generate_response(finetuned_model, tokenizer, prompt)

# Run evaluations
before_task = run_eval_suite(before_model, task_eval, "BEFORE - Task accuracy")
after_task = run_eval_suite(after_model, task_eval, "AFTER - Task accuracy")

before_general = run_eval_suite(before_model, general_eval, "BEFORE - General capability")
after_general = run_eval_suite(after_model, general_eval, "AFTER - General capability")

# Print summary
print("\n=== Fine-Tuning Impact Summary ===")
print(f"Task accuracy:        {before_task['accuracy']:.1%} → {after_task['accuracy']:.1%} "
      f"({'↑' if after_task['accuracy'] > before_task['accuracy'] else '↓'}"
      f"{abs(after_task['accuracy'] - before_task['accuracy']):.1%})")
print(f"General capability:   {before_general['accuracy']:.1%} → {after_general['accuracy']:.1%} "
      f"({'↑' if after_general['accuracy'] > before_general['accuracy'] else '↓'}"
      f"{abs(after_general['accuracy'] - before_general['accuracy']):.1%})")

if after_general['accuracy'] < before_general['accuracy'] - 0.05:
    print("\n⚠ WARNING: General capability degraded by >5%. Consider:")
    print("  - Reducing number of training epochs")
    print("  - Using LoRA instead of full fine-tuning")
    print("  - Adding general instruction-following data to your training mix")
```

---

## Mini-Project: Domain-Specific JSON Extractor

**Goal:** Fine-tune a model with LoRA to reliably extract structured data from a specific domain (e.g., job listings) and output valid JSON.

**Specification:**
```
Input:  "Software Engineer at Stripe. Remote. $150K-$200K. 5+ years Python required."
Output: {"title": "Software Engineer", "company": "Stripe", "location": "Remote",
          "salary_min": 150000, "salary_max": 200000, "required_years": 5,
          "required_skills": ["Python"]}
```

**What to build:**
1. Generate or collect 200+ (input, JSON output) pairs for job listings
2. Apply LoRA (r=16) to Phi-2 or OPT-1.3B — small enough to train locally
3. Measure JSON validity rate before and after (% of outputs that parse without error)
4. Measure field accuracy: what % of extracted fields are correct?

**Starter function for JSON validity rate:**
```python
import json

def json_validity_rate(model_fn, test_inputs: list[str]) -> float:
    valid = 0
    for inp in test_inputs:
        output = model_fn(inp)
        try:
            # Try to parse as JSON
            json.loads(output)
            valid += 1
        except json.JSONDecodeError:
            pass
    return valid / len(test_inputs)

# Expected: before fine-tuning ~20-40% validity
#           after fine-tuning  ~85-95% validity (if dataset is clean)
```

---

## Checklist

- [ ] Completed Exercise 1: full fine-tuned GPT-2 and saw format learning
- [ ] Completed Exercise 2: applied LoRA, confirmed >98% parameter reduction, compared ranks
- [ ] Completed Exercise 3: ran QLoRA on a 7B model (or reviewed the code + GPU memory math)
- [ ] Completed Exercise 4: ran before/after evaluation and checked for capability degradation
- [ ] Built the mini-project: domain-specific JSON extractor with LoRA
