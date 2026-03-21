---
sidebar_position: 1
title: "Module 10: Multimodal Systems"
description: Work with inputs beyond text — images, audio, and documents. Covers vision-language models, diffusion models, document AI, and multimodal agents.
---

# Module 10: Multimodal Systems

Text-only AI is the tip of the iceberg.

The real world is multimodal: users send screenshots, upload invoices, share charts, and record voice memos. Limiting your AI to text-in → text-out means excluding a massive category of use cases — and leaving meaningful productivity gains unrealized.

This module covers the second generation of AI capabilities: vision-language models that see and reason, diffusion models that generate images, audio pipelines that transcribe and speak, and document AI that handles the structured chaos of PDFs, forms, and scanned pages.

## What You'll Learn

- How vision-language models encode images as patch embeddings and connect them to text decoders
- What VLMs can and cannot do reliably — where the hard limits are today
- Diffusion model architecture: how coherent images emerge from pure noise
- Text-to-image prompt engineering: getting consistent, high-quality results
- Multimodal agents: combining vision, text, and tool use in a single loop
- Audio pipelines: Whisper for transcription, text-to-speech for voice output
- Document AI: OCR + layout understanding + VLM for real-world document processing
- When to use which multimodal model for which problem

## Prerequisites

- [Module 1: LLM Fundamentals](/learn/ai-engineering/module-1-llm-fundamentals) — embeddings and tokenization concepts apply to images too
- [Module 3: Prompting & Reasoning](/learn/ai-engineering/module-3-prompting-reasoning) — prompt engineering skills transfer to multimodal models
- [Module 8: Agents & System Design](/learn/ai-engineering/module-8-agents-system-design) — for the screenshot-to-action agent exercise
- OpenAI API key with GPT-4o access, or Anthropic key with Claude 3+ access

## Time Estimate

~4 hours

## Module Structure

| Page | What's covered |
|------|----------------|
| [Overview](./overview) | VLM architecture (patch embeddings, cross-attention), diffusion models, CLIP, multimodal agents, Whisper audio pipelines, Document AI patterns |
| [Hands-On](./hands-on) | Build image Q&A, extract structured data from PDF invoices, benchmark OCR vs VLM vs combined pipeline, build a screenshot-to-action agent |
| [Resources](./resources) | Papers (CLIP, LLaVA, Flamingo, Whisper), model docs, tools (pdf2image, Tesseract, Pillow) |

---

**Ready to start? →** [Overview: Multimodal Systems](./overview)
