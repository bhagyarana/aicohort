---
sidebar_position: 3
title: "Hands-On"
description: Practical exercises — image Q&A, PDF invoice parsing, OCR vs VLM benchmark, screenshot-to-action agent
---

# Hands-On: Multimodal Systems

## Setup

```bash
pip install openai pillow pdf2image pytesseract requests

# On macOS/Linux, install Tesseract:
# brew install tesseract (macOS)
# apt-get install tesseract-ocr (Ubuntu)

# pdf2image requires poppler:
# brew install poppler (macOS)
# apt-get install poppler-utils (Ubuntu)
```

```python
import os
import openai

client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
```

---

## Exercise 1: Image Q&A System (Beginner)

**Goal:** Build a reusable function that sends an image and a question to GPT-4o and returns an answer.

**Time:** ~25 min

### Step 1: Send an image URL

```python
def ask_about_image_url(image_url: str, question: str) -> str:
    """Ask a question about an image from a URL."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url, "detail": "high"}
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ],
        max_tokens=500
    )
    return response.choices[0].message.content
```

Test it with a public image:
```python
answer = ask_about_image_url(
    image_url="https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
    question="What objects do you see in this image? Describe the colors."
)
print(answer)
```

### Step 2: Send a local image file

```python
import base64
from pathlib import Path

def ask_about_local_image(image_path: str, question: str) -> str:
    """Ask a question about a local image file."""
    image_data = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")

    # Detect MIME type from extension
    ext = Path(image_path).suffix.lower()
    mime_types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                  ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}
    mime_type = mime_types.get(ext, "image/jpeg")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_data}",
                            "detail": "high"  # "low" is cheaper but less accurate
                        }
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ],
        max_tokens=500
    )
    return response.choices[0].message.content
```

### Step 3: Build a multi-question image analyzer

```python
def analyze_image(image_path: str, questions: list[str]) -> dict[str, str]:
    """Ask multiple questions about an image in a single call."""
    numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
                        }
                    },
                    {
                        "type": "text",
                        "text": f"""Answer each question about this image.
Format your response as:
1. [answer]
2. [answer]
...

Questions:
{numbered}"""
                    }
                ]
            }
        ],
        max_tokens=1000
    )

    answers = response.choices[0].message.content.strip().split("\n")
    return {questions[i]: answers[i].lstrip("0123456789. ") for i in range(len(questions))}

def encode_image(image_path: str) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
```

**Try it on your own images** — screenshots, product photos, charts, anything. Notice where it gets things right and wrong.

---

## Exercise 2: PDF Invoice Parser (Intermediate)

**Goal:** Extract structured data from a PDF invoice using a VLM.

**Time:** ~40 min

### Step 1: Convert PDF pages to images

```python
from pdf2image import convert_from_path
from pathlib import Path
import tempfile
import os

def pdf_to_images(pdf_path: str, dpi: int = 200) -> list[str]:
    """Convert PDF pages to image files. Returns list of image paths."""
    images = convert_from_path(pdf_path, dpi=dpi)
    image_paths = []

    output_dir = Path(tempfile.mkdtemp())
    for i, image in enumerate(images):
        image_path = str(output_dir / f"page_{i+1}.png")
        image.save(image_path, "PNG")
        image_paths.append(image_path)

    return image_paths
```

### Step 2: Extract structured invoice data

```python
import json

INVOICE_SCHEMA = {
    "invoice_number": "string",
    "invoice_date": "YYYY-MM-DD format",
    "due_date": "YYYY-MM-DD format or null",
    "vendor": {
        "name": "string",
        "address": "string or null",
        "email": "string or null"
    },
    "bill_to": {
        "name": "string",
        "address": "string or null"
    },
    "line_items": [
        {
            "description": "string",
            "quantity": "number",
            "unit_price": "number",
            "total": "number"
        }
    ],
    "subtotal": "number",
    "tax": "number or null",
    "total_amount": "number",
    "currency": "3-letter code e.g. USD"
}

def extract_invoice_data(image_path: str) -> dict:
    """Extract structured invoice data from an invoice image."""
    image_data = encode_image(image_path)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": f"""Extract all invoice data from this image.
Return a JSON object matching this schema exactly:
{json.dumps(INVOICE_SCHEMA, indent=2)}

Rules:
- Use null for fields not present in the invoice
- All monetary values should be numbers (not strings)
- Return ONLY valid JSON, no explanation or markdown"""
                    }
                ]
            }
        ],
        response_format={"type": "json_object"},
        max_tokens=1000
    )

    return json.loads(response.choices[0].message.content)
```

### Step 3: Handle multi-page invoices

```python
def process_invoice_pdf(pdf_path: str) -> dict:
    """Process a multi-page PDF invoice."""
    image_paths = pdf_to_images(pdf_path)

    if len(image_paths) == 1:
        # Single page — direct extraction
        return extract_invoice_data(image_paths[0])

    # Multi-page: extract from each page, merge results
    all_data = []
    for image_path in image_paths:
        try:
            page_data = extract_invoice_data(image_path)
            all_data.append(page_data)
        except Exception as e:
            print(f"Failed to extract from {image_path}: {e}")

    # Merge: keep first page's header data, combine all line items
    merged = all_data[0].copy()
    all_line_items = []
    for page_data in all_data:
        if page_data.get("line_items"):
            all_line_items.extend(page_data["line_items"])

    merged["line_items"] = all_line_items
    return merged
```

**Test with a sample invoice**: Download a free sample PDF invoice from the web, or create one in Word/Google Docs and save as PDF.

---

## Exercise 3: OCR vs VLM vs Pipeline Comparison (Intermediate)

**Goal:** Quantitatively compare three approaches to document text extraction.

**Time:** ~45 min

### Step 1: Tesseract OCR extraction

```python
import pytesseract
from PIL import Image

def extract_with_ocr(image_path: str) -> str:
    """Extract text from image using Tesseract OCR."""
    image = Image.open(image_path)

    # Tesseract config: PSM 3 = automatic page segmentation
    custom_config = r'--oem 3 --psm 3'
    text = pytesseract.image_to_string(image, config=custom_config)

    return text.strip()
```

### Step 2: VLM extraction (text only)

```python
def extract_with_vlm(image_path: str) -> str:
    """Extract text from image using GPT-4o."""
    image_data = encode_image(image_path)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}", "detail": "high"}
                    },
                    {
                        "type": "text",
                        "text": "Transcribe all text visible in this image exactly as it appears, preserving layout where possible."
                    }
                ]
            }
        ],
        max_tokens=2000
    )
    return response.choices[0].message.content
```

### Step 3: Combined pipeline

```python
def extract_with_combined(image_path: str) -> str:
    """Extract using OCR for accuracy + VLM for structure."""
    ocr_text = extract_with_ocr(image_path)
    image_data = encode_image(image_path)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"}
                    },
                    {
                        "type": "text",
                        "text": f"""OCR has extracted the following text from this image:
---
{ocr_text}
---
Using both the image (for layout and structure) and the OCR text (for accuracy),
provide a clean, well-structured transcription of the document.
Fix any OCR errors you can identify from the image."""
                    }
                ]
            }
        ],
        max_tokens=2000
    )
    return response.choices[0].message.content
```

### Step 4: Benchmark

```python
def benchmark_extraction(image_path: str, ground_truth: str | None = None) -> dict:
    """Compare all three extraction methods."""
    import time

    results = {}

    for method_name, method in [
        ("OCR only", extract_with_ocr),
        ("VLM only", extract_with_vlm),
        ("Combined", extract_with_combined)
    ]:
        start = time.time()
        text = method(image_path)
        elapsed = time.time() - start

        result = {"text": text, "latency_seconds": round(elapsed, 2)}

        if ground_truth:
            # Simple character-level accuracy
            common = sum(1 for c in text if c in ground_truth)
            result["approx_recall"] = round(common / len(ground_truth), 3)

        results[method_name] = result

    return results
```

**Try it on**: a screenshot of a webpage, a photo of a printed form, or a scanned document with complex layout. The difference between methods is most visible on documents with mixed text, tables, and images.

---

## Mini-Project: Screenshot-to-Action Agent (Advanced)

**Goal:** Build an agent that takes a screenshot, understands the UI, and suggests (or executes) the next action toward a goal.

**Time:** ~60 min

### Step 1: Screenshot capture utility

```python
import subprocess
import platform

def capture_screenshot(output_path: str = "screenshot.png") -> str:
    """Capture the current screen to a file."""
    system = platform.system()

    if system == "Darwin":  # macOS
        subprocess.run(["screencapture", "-x", output_path], check=True)
    elif system == "Linux":
        subprocess.run(["scrot", output_path], check=True)
    elif system == "Windows":
        from PIL import ImageGrab
        img = ImageGrab.grab()
        img.save(output_path)

    return output_path
```

### Step 2: Screen analyzer

```python
def analyze_screen(screenshot_path: str, goal: str, step_number: int) -> dict:
    """Analyze a screenshot and determine the next action toward a goal."""
    image_data = encode_image(screenshot_path)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You are a UI navigation assistant.
Analyze screenshots and determine actions to achieve goals.
Always respond with valid JSON."""
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}", "detail": "high"}
                    },
                    {
                        "type": "text",
                        "text": f"""Goal: {goal}
Current step: {step_number}

Analyze what's currently on screen and determine the next action.

Respond with JSON:
{{
  "screen_description": "what you see on screen",
  "progress": "how close to goal (0-100%)",
  "next_action": {{
    "type": "click | type | scroll | wait | done",
    "description": "what to do and where",
    "coordinates": {{"x": 0, "y": 0}} // approximate if click
  }},
  "reasoning": "why this action moves toward the goal",
  "goal_complete": false
}}"""
                    }
                ]
            }
        ],
        response_format={"type": "json_object"},
        max_tokens=500
    )

    return json.loads(response.choices[0].message.content)
```

### Step 3: The agent loop (observation only — no execution)

```python
def screenshot_agent_observe(goal: str, max_steps: int = 5):
    """
    Observe-only agent: takes screenshots, describes what it sees,
    suggests actions but does NOT execute them.
    Safe for demonstration purposes.
    """
    print(f"\nGoal: {goal}")
    print("=" * 50)

    for step in range(1, max_steps + 1):
        print(f"\nStep {step}:")

        # Capture current screen state
        screenshot_path = f"step_{step}.png"
        capture_screenshot(screenshot_path)

        # Analyze and get next action
        analysis = analyze_screen(screenshot_path, goal, step)

        print(f"  Screen: {analysis['screen_description'][:100]}...")
        print(f"  Progress: {analysis['progress']}%")
        print(f"  Suggested action: {analysis['next_action']['type']} — {analysis['next_action']['description']}")
        print(f"  Reasoning: {analysis['reasoning']}")

        if analysis.get("goal_complete"):
            print(f"\n  Goal complete!")
            break

        # In a real agent, you would execute the action here:
        # execute_action(analysis["next_action"])
        # For now, just wait for manual action
        input("\n  Press Enter after taking the suggested action (or Ctrl+C to stop)...")

# Run it
screenshot_agent_observe("Find the Python documentation for the 'requests' library")
```

### Step 4: Add action history for context

```python
def screenshot_agent_with_history(goal: str, max_steps: int = 8):
    """Agent that maintains action history for better context."""
    history = []

    for step in range(1, max_steps + 1):
        screenshot_path = f"step_{step}.png"
        capture_screenshot(screenshot_path)
        image_data = encode_image(screenshot_path)

        history_text = "\n".join(
            f"Step {h['step']}: {h['action']} — {h['outcome']}"
            for h in history
        ) if history else "No previous steps."

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_data}"}
                        },
                        {
                            "type": "text",
                            "text": f"""Goal: {goal}

Previous steps:
{history_text}

Current step: {step}
Determine the next action. Respond as JSON with keys:
action_type, description, reasoning, goal_complete (boolean)"""
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=400
        )

        action = json.loads(response.choices[0].message.content)
        history.append({"step": step, "action": action["description"], "outcome": "pending"})

        print(f"Step {step}: {action['action_type']} — {action['description']}")

        if action.get("goal_complete"):
            print("Goal complete!")
            return history

    return history
```

---

## Checklist

- [ ] Exercise 1: Built `ask_about_image_url` and `ask_about_local_image`, tested on at least 3 different images
- [ ] Exercise 1: Observed at least one limitation (counting, spatial reasoning, or small text)
- [ ] Exercise 2: Successfully extracted structured JSON from a PDF invoice
- [ ] Exercise 2: Handled a multi-page or complex-layout invoice
- [ ] Exercise 3: Ran the benchmark on at least 2 different document types
- [ ] Exercise 3: Can explain when OCR alone is better and when VLM alone is better
- [ ] Mini-project: Built the observation loop for the screenshot agent
- [ ] Mini-project: Added action history to improve context across steps

---

**Next: →** [Resources: Multimodal Systems](./resources)
