---
sidebar_position: 2
title: "Overview"
description: Theory and concepts — VLM architecture, diffusion models, multimodal agents, audio pipelines, and Document AI
---

# Multimodal Systems — Deep Dive

Text has dominated AI for good reason: it is cheap to collect, easy to standardize, and sufficient for a huge range of tasks. But the world communicates in more than words. Contracts are PDFs. Dashboards are screenshots. Customer complaints include photos. Medical records contain scans.

Multimodal AI bridges that gap — and in 2024–2025, it reached production-quality for the first time.

---

## Vision-Language Models (VLMs)

A vision-language model (VLM) is a model that accepts both image and text as input and produces text as output. The architectural challenge: images and text live in completely different spaces. Merging them requires a bridging mechanism.

### How Images Become Tokens

The standard approach is **patch embedding**:

```
Input image (e.g., 448×448 pixels)
          ↓
Divide into patches (e.g., 16×16 pixels each)
          ↓
784 patches (448/16 × 448/16 = 28 × 28)
          ↓
Each patch → linear projection → 768-dim vector
          ↓
784 image tokens added to the token sequence
          ↓
Standard transformer processes combined text + image tokens
```

The image is treated as a sequence of visual "tokens" — just like word tokens. The model attends across text and image tokens in the same attention layers.

### Cross-Attention vs Unified Attention

Two architectural approaches:

| Architecture | How It Works | Examples | Tradeoff |
|---|---|---|---|
| Unified attention | Image patches + text tokens share the same transformer layers | GPT-4V, LLaVA | Tight coupling, higher capacity |
| Cross-attention | Separate image encoder, cross-attention bridges to text decoder | Flamingo | Modular, easier to swap encoders |
| Adapter layers | Frozen LLM + small adapter that projects image features into text space | LLaVA-v1.5 | Efficient, less image-text integration |

### CLIP: The Foundation

Most VLMs are built on top of CLIP (Contrastive Language–Image Pre-training):

```
Training setup:
  Image: "a photo of a dog"
  Positive pair: [dog photo] ↔ "a photo of a dog"
  Negative pairs: [dog photo] ↔ "a photo of a car" (pushed apart)

  Loss: maximize similarity between matching pairs,
        minimize similarity between non-matching pairs

Result: image embeddings and text embeddings live in the same space
        → semantically similar images and texts have close vectors
```

This shared embedding space is why you can do zero-shot image classification: embed the image, embed candidate class labels, find the nearest label.

:::tip
CLIP embeddings are also useful outside of full VLMs — you can use them for image search, image clustering, and multimodal RAG without spinning up a full vision-language model.
:::

### Key VLMs in 2024–2025

| Model | Provider | Strengths | Weaknesses |
|-------|----------|-----------|------------|
| GPT-4o (vision) | OpenAI | Strong reasoning, chart reading, broad knowledge | Expensive, no self-hosting |
| Claude 3.5 Sonnet | Anthropic | Excellent document understanding, safety | Cost, no self-hosting |
| Gemini 1.5 Pro | Google | Very long context (1M tokens), video support | API availability |
| LLaVA-1.6 | Open source | Self-hostable, free, good general VQA | Lower quality than frontier models |
| Idefics2 | HuggingFace | Open, lightweight | Limited capabilities |

---

## What VLMs Can and Cannot Do

### Reliable capabilities

- **Image captioning** — describe what's in an image at multiple levels of detail
- **Visual question answering** — answer specific questions about image content
- **Chart and graph reading** — extract values from bar charts, line graphs, pie charts
- **Document understanding** — read and interpret text in images (menus, signs, slides)
- **Object detection (descriptive)** — name and describe objects in scenes
- **Style and mood analysis** — describe aesthetic qualities, emotions, atmosphere

### Unreliable capabilities (use with caution)

| Task | Why VLMs Struggle | Workaround |
|------|------------------|------------|
| Counting objects | Attention mechanism loses track at >10 objects | Use specialized detection model |
| Precise spatial reasoning | Left/right/above/below are often wrong | Ask for grid-based description instead |
| Reading very small text | Resolution limits, patch size too coarse | Pre-process with OCR, pass text separately |
| Exact color matching | Perception varies by model | Provide reference color codes in prompt |
| Temporal reasoning in video | Most models see static frames only | Extract key frames, label them manually |
| Pixel-level editing instructions | Not a generation task | Use diffusion models instead |

:::note
The gap between "technically can do" and "reliably does" is wide for multimodal models. Always evaluate on your specific document/image type before committing to a VLM-based pipeline.
:::

---

## Diffusion Models

Diffusion models power text-to-image generation. The core idea is elegant: learn to reverse a noise-adding process.

### Forward and Reverse Processes

```
Forward process (training):
  Clean image → add Gaussian noise step by step → pure noise

  x₀ (clean) → x₁ (slightly noisy) → x₂ → ... → xT (pure noise)

  The model learns to PREDICT the noise added at each step.

Reverse process (inference):
  Pure noise → predict and remove noise step by step → clean image

  xT (noise) → xT-1 → ... → x₁ → x₀ (generated image)

  The text condition guides WHICH direction to remove noise.
```

### Text Conditioning via CLIP

```
Text prompt: "a golden retriever running on a beach at sunset"
         ↓
CLIP text encoder → text embedding
         ↓
Cross-attention in U-Net: embedding guides denoising direction
         ↓
After ~50 denoising steps → final image
```

The text embedding influences every denoising step, steering the generation toward images that match the text description.

### Model Families

| Model | Creator | Hosting | Notes |
|-------|---------|---------|-------|
| DALL-E 3 | OpenAI | API only | Best at following complex text descriptions |
| Stable Diffusion 3 | Stability AI | Open weights | Self-hostable, large ecosystem of fine-tunes |
| Midjourney | Midjourney | Discord/web | Highest aesthetic quality, not API-friendly |
| Imagen 3 | Google | API (limited) | Strong photorealism |
| Flux | Black Forest Labs | Open weights | Excellent quality, SOTA open-source |

---

## Text-to-Image Prompt Engineering

Generating good images requires different intuitions from text prompting.

### Anatomy of an Effective Image Prompt

```
[Subject] + [Style] + [Composition] + [Technical] + [Quality modifiers]

Example:
"Portrait of an elderly botanist examining a rare orchid,
 oil painting in the style of Rembrandt,
 warm candlelight, shallow depth of field,
 8K resolution, highly detailed"
```

### Prompt Modifiers That Consistently Work

| Category | Examples |
|----------|---------|
| Style | "oil painting", "watercolor", "photorealistic", "anime style", "line art" |
| Lighting | "golden hour", "studio lighting", "dramatic shadows", "neon lights" |
| Camera | "portrait lens", "wide angle", "macro photography", "aerial view" |
| Quality | "highly detailed", "sharp focus", "8K", "professional photography" |
| Negative (SD only) | "blurry, low quality, watermark, text, bad anatomy" |

:::tip
DALL-E 3 understands natural language descriptions well — write it as a detailed scene description. Stable Diffusion responds better to comma-separated keyword lists with explicit style modifiers.
:::

---

## Multimodal Agents

The most powerful multimodal pattern combines vision, text, and tool use in a loop:

```
User: "Book the flight shown in this screenshot"
         ↓
[VLM] analyzes screenshot → extracts: airline, date, price, booking URL
         ↓
[Agent] plans: 1) verify details, 2) navigate to URL, 3) complete booking
         ↓
[Browser tool] navigates to booking page → takes new screenshot
         ↓
[VLM] analyzes new screenshot → confirms booking form
         ↓
[Agent] fills form fields, submits
         ↓
[VLM] verifies confirmation → responds to user
```

### Screenshot-to-Action Pattern

This is the foundation of computer-use agents (Anthropic's Claude Computer Use, OpenAI's Operator):

```python
# Simplified screenshot-to-action loop
def screenshot_to_action(goal: str, max_steps: int = 10):
    for step in range(max_steps):
        screenshot = capture_screen()

        response = vlm.analyze(
            image=screenshot,
            prompt=f"""
            Goal: {goal}
            Current step: {step + 1}

            Describe what you see on screen.
            Determine the next action to take toward the goal.
            Respond with JSON: {{"action": "click|type|scroll|done",
                                "target": "...", "value": "..."}}
            """
        )

        action = parse_json(response)

        if action["action"] == "done":
            return action["value"]

        execute_action(action)
```

---

## Audio Pipelines

### Whisper: Transcription

OpenAI's Whisper is the current standard for speech-to-text:

```python
import openai

# Transcribe audio file
with open("recording.mp3", "rb") as audio_file:
    transcript = openai.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="verbose_json",  # includes word-level timestamps
        timestamp_granularities=["word"]
    )

print(transcript.text)
# Word-level timestamps for speaker diarization or karaoke-style display:
for word in transcript.words:
    print(f"{word.start:.2f}s - {word.end:.2f}s: {word.word}")
```

Whisper's strengths: multilingual (99 languages), robust to accents and background noise, good punctuation.

### Text-to-Speech Pipelines

```
Text → [TTS model] → Audio file or streaming audio

Common choices:
  OpenAI TTS-1: fast, 6 voices, good for real-time
  ElevenLabs: highest quality, voice cloning, emotional range
  Coqui TTS: open source, self-hostable
  Azure Neural TTS: enterprise, SSML support, many languages
```

### End-to-End Voice Pipeline

```
User speaks
    ↓
[Whisper] transcription → text
    ↓
[LLM] processes text → response text
    ↓
[TTS model] text → speech audio
    ↓
User hears response
```

Latency is the challenge: each step adds delay. Streaming responses from the LLM and streaming audio output from TTS reduces perceived latency significantly.

---

## Document AI

### The Real-World Problem

Documents in production are messy:
- PDFs with embedded text (easy case)
- Scanned PDFs with only images (hard case)
- Forms with structured fields (medium case)
- Documents with tables, charts, mixed layouts (hardest case)

### The OCR + VLM Pipeline

```
PDF → pdf2image (render pages as images)
           ↓
    ┌──────┴──────┐
    │             │
[Tesseract OCR]  [VLM analysis]
text extraction   layout + context
    │             │
    └──────┬──────┘
           ↓
    Combine: structure from VLM,
             raw text from OCR
           ↓
    Pass combined context to LLM
           ↓
    Structured output (JSON/table)
```

### When to Use What

| Input Type | Best Approach |
|-----------|---------------|
| Searchable PDF (text layer) | Direct text extraction (PyMuPDF) |
| Scanned image PDF | OCR → text extraction |
| Complex layout (tables, charts) | VLM + OCR combined |
| Invoice/form extraction | VLM with structured output schema |
| Academic paper | PyMuPDF + section parsing |
| Handwritten notes | VLM (Tesseract fails on handwriting) |

```python
import base64
from pathlib import Path
import openai

def extract_invoice_data(image_path: str) -> dict:
    """Extract structured data from an invoice image using VLM."""
    image_data = base64.b64encode(Path(image_path).read_bytes()).decode()

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                },
                {
                    "type": "text",
                    "text": """Extract the following from this invoice as JSON:
                    {
                      "invoice_number": "...",
                      "date": "YYYY-MM-DD",
                      "vendor_name": "...",
                      "total_amount": 0.00,
                      "currency": "...",
                      "line_items": [
                        {"description": "...", "quantity": 0, "unit_price": 0.00, "total": 0.00}
                      ]
                    }
                    Return only valid JSON, no explanation."""
                }
            ]
        }],
        response_format={"type": "json_object"}
    )

    import json
    return json.loads(response.choices[0].message.content)
```

---

## Mental Model

**Images are documents written in pixels, not letters.**

A VLM doesn't "see" the way humans see. It converts an image into a sequence of patch tokens — which is just another way to serialize information into a form a transformer can process. The model attends across image patches and text tokens using the same mechanism it uses to attend across words.

The implication: all the prompting intuitions from text apply. Be specific about what you want. Provide examples. Specify the output format. The model is pattern-matching on visual tokens just as it pattern-matches on word tokens.

---

## Common Mistakes

| Mistake | Why It Happens | Fix |
|---------|---------------|-----|
| Sending low-resolution images | Assuming the model compensates | Resize to at least 512px minimum dimension before sending |
| Using VLM alone for text-heavy documents | OCR accuracy matters for dense text | Pre-extract text with Tesseract, combine with VLM layout understanding |
| Asking VLMs to count precisely | Fundamental attention limitation | Use specialized detection models (YOLO, Detectron2) for counting |
| Sending raw PDF bytes to the API | Most APIs only accept images | Render PDF pages to PNG/JPEG first (pdf2image) |
| No structured output schema | Inconsistent extraction across runs | Always provide a JSON schema for document extraction tasks |
| Assuming VLM output is factual | Models hallucinate about image content | Cross-check critical fields with OCR or targeted CV models |

---

## Quiz

> **Q1: A user uploads a scanned medical form (image-only PDF). You need to extract 20 specific field values reliably. Should you use a VLM alone, OCR alone, or both? Why?**
>
> <details><summary>Show Answer</summary>
>
> **Use both: OCR for raw text accuracy + VLM for layout understanding.**
>
> OCR alone (Tesseract) extracts text accurately but doesn't understand structure — it can't distinguish which text belongs to which form field.
>
> VLM alone understands context and structure but can hallucinate exact values, especially for numbers, dates, and checkboxes where accuracy is critical.
>
> The optimal pipeline: render PDF page to image → run Tesseract for accurate text extraction → pass both the image and the OCR text to the VLM with the prompt "here is the OCR text: [text]. Use it as the source of truth for values, but use the image to understand which fields they belong to."
>
> This combination gives you structural understanding (VLM) + factual accuracy (OCR).
> </details>

> **Q2: You're building a real-time voice assistant. Whisper transcription takes 800ms, LLM response takes 1200ms, and TTS takes 600ms. Total: 2.6 seconds per turn. How would you reduce perceived latency?**
>
> <details><summary>Show Answer</summary>
>
> Three techniques:
>
> 1. **Stream LLM output to TTS**: Don't wait for the full LLM response. As tokens arrive, send complete sentences to the TTS model. The user starts hearing audio after the first sentence, not the last.
>
> 2. **Reduce Whisper latency**: Use a smaller Whisper model (whisper-tiny or whisper-base) for the first transcription pass, then optionally re-transcribe asynchronously for logging. Or use streaming ASR (Deepgram, AssemblyAI) that transcribes while the user is still speaking.
>
> 3. **Pre-compute TTS for common responses**: For greeting messages, clarification requests, or common error messages, pre-generate audio and cache it.
>
> Practical result: streaming LLM → TTS pipeline can reduce perceived latency from 2.6s to ~0.8–1.2s (time to first audio chunk).
> </details>

> **Q3: You're using GPT-4o to analyze product photos for an e-commerce site. The model is describing colors inaccurately — reporting "navy blue" as "dark blue" and "forest green" as "dark green". What's the problem and how do you fix it?**
>
> <details><summary>Show Answer</summary>
>
> **The problem**: VLMs perceive colors inconsistently and don't have precise color vocabulary matching your catalog. "Navy" vs "dark blue" is a naming distinction, not a perception distinction.
>
> **Fixes**:
>
> Option A — Provide color vocabulary in the prompt: "Classify the primary color using ONLY these exact labels: [navy, cobalt, sky blue, teal, ...]. Do not use any other color names."
>
> Option B — Use a CV library for color extraction: Extract the dominant RGB values using Pillow or OpenCV, then map to your catalog's color names using nearest-neighbor matching in color space. VLMs are not needed for this subtask.
>
> Option C — Hybrid: Use VLM for everything else (style, material, context), use color extraction code for color specifically.
>
> **Lesson**: Identify which subtasks benefit from VLM understanding vs deterministic code. Color extraction is better done with code.
> </details>

---

## Summary Table

| Concept | What It Is | When to Use |
|---------|-----------|------------|
| VLM (GPT-4o, Claude vision) | LLM that accepts image + text input | Document understanding, visual QA, chart reading |
| CLIP | Shared image-text embedding space | Image search, zero-shot classification, multimodal RAG |
| Diffusion model | Noise → image via learned denoising | Image generation, editing, inpainting |
| Whisper | Speech-to-text model | Transcription, voice input |
| TTS pipeline | Text → audio | Voice output, accessibility |
| OCR (Tesseract) | Image → text extraction | Text-heavy documents, when accuracy matters more than context |
| Document AI pipeline | OCR + VLM combined | Forms, invoices, mixed-layout PDFs |
| Patch embedding | How images become tokens | Internal VLM representation |
| Screenshot-to-action | Vision + reasoning + tool use loop | Computer use agents, UI automation |

---

## Next Steps

→ [Hands-On: Multimodal Systems](./hands-on)
