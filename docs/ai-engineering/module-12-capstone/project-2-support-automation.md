---
sidebar_position: 3
title: "Project 2: Customer Support Automation"
description: Build a multi-agent support pipeline that classifies intent, routes to specialized agents, retrieves knowledge base articles, responds with confidence scores, and escalates to humans when needed.
---

# Project 2: Customer Support Automation

**Difficulty:** Intermediate
**Modules:** 3, 4, 8, 9, 11
**Time:** 10–14 hours

---

## Problem Statement

Build a customer support system that:
1. Receives a support message (email, chat, ticket)
2. Classifies the intent (billing, technical, account, general)
3. Routes to a specialized agent for each category
4. Retrieves relevant knowledge base articles using semantic search
5. Generates a helpful, grounded response
6. Escalates to a human agent when confidence is low or the issue is unresolvable

**What makes this hard**: The escalation logic. A system that tries to answer everything is worse than a system that knows when to stop. The product requirements are: never mislead a customer, never leave them stuck, and never waste a human agent's time on something the AI could resolve.

---

## Architecture

```
Incoming support message
          ↓
[Intent Classifier]
  → BILLING | TECHNICAL | ACCOUNT | GENERAL | ESCALATE
          ↓
[Router] dispatches to specialized agent
    ┌────┴────────────────────────┐
    │                             │
[Billing Agent]          [Technical Agent]
[Account Agent]          [General Agent]
    │                             │
    └────┬────────────────────────┘
         ↓
[Knowledge Base Retrieval]
  Vector search on KB articles
         ↓
[Response Generator]
  Grounded on KB articles
         ↓
[Confidence Evaluator]
  Score: 0.0–1.0
         ↓
  Score > 0.75 → send response
  Score < 0.75 → escalate to human
         ↓
     Output
```

---

## Implementation Guide

### Phase 1: Intent Classification (1–2 hours)

```python
import os
import json
import openai

client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

INTENT_SYSTEM_PROMPT = """You are a customer support intent classifier.

Classify the customer message into exactly one category:
- BILLING: payment issues, invoices, charges, refunds, subscription pricing
- TECHNICAL: bugs, errors, connectivity issues, feature not working
- ACCOUNT: login, password reset, profile changes, account access
- GENERAL: product questions, feature requests, how-to questions
- ESCALATE: angry customers, legal threats, sensitive situations, abuse

Return JSON: {"intent": "CATEGORY", "confidence": 0.0-1.0, "reason": "brief explanation"}"""


def classify_intent(message: str) -> dict:
    """Classify the intent of a support message."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": message}
        ],
        response_format={"type": "json_object"},
        max_tokens=100,
        temperature=0
    )
    return json.loads(response.choices[0].message.content)
```

### Phase 2: Knowledge Base Setup (2 hours)

```python
import chromadb
from chromadb.utils import embedding_functions

chroma = chromadb.PersistentClient(path="./support_kb")
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name="text-embedding-3-small"
)
kb = chroma.get_or_create_collection("knowledge_base", embedding_function=openai_ef)

# Sample knowledge base articles — replace with your actual KB
KB_ARTICLES = [
    {
        "id": "kb_001",
        "category": "BILLING",
        "title": "How to Update Payment Method",
        "content": "To update your payment method: 1) Go to Settings > Billing. 2) Click 'Update Payment Method'. 3) Enter your new card details. 4) Click Save. Changes take effect on your next billing date."
    },
    {
        "id": "kb_002",
        "category": "BILLING",
        "title": "Refund Policy",
        "content": "We offer full refunds within 14 days of purchase. To request a refund, contact support with your order number. Refunds are processed within 5-7 business days to your original payment method."
    },
    {
        "id": "kb_003",
        "category": "TECHNICAL",
        "title": "App Not Loading Fix",
        "content": "If the app is not loading: 1) Clear browser cache and cookies. 2) Try an incognito/private window. 3) Disable browser extensions. 4) Check status.ourapp.com for outages. 5) If problem persists, contact support with your browser version and OS."
    },
    {
        "id": "kb_004",
        "category": "ACCOUNT",
        "title": "Password Reset Instructions",
        "content": "To reset your password: 1) Click 'Forgot Password' on the login page. 2) Enter your email address. 3) Check your email for a reset link (check spam if not received). 4) Click the link within 24 hours. 5) Enter and confirm your new password."
    },
    {
        "id": "kb_005",
        "category": "ACCOUNT",
        "title": "Changing Your Email Address",
        "content": "To change your email: Go to Settings > Profile > Email. Enter your new email and verify it. You'll need to confirm both your old and new email addresses. This change takes effect immediately after confirmation."
    },
    {
        "id": "kb_006",
        "category": "GENERAL",
        "title": "Available Subscription Plans",
        "content": "We offer three plans: Free (basic features, 5 projects), Pro ($19/month, unlimited projects, priority support), and Enterprise (custom pricing, dedicated support, SSO, SLA). Annual billing saves 20%."
    },
]


def setup_knowledge_base():
    """Load KB articles into ChromaDB."""
    if kb.count() > 0:
        return  # Already loaded

    kb.add(
        documents=[a["content"] for a in KB_ARTICLES],
        ids=[a["id"] for a in KB_ARTICLES],
        metadatas=[{"category": a["category"], "title": a["title"]} for a in KB_ARTICLES]
    )
    print(f"Loaded {len(KB_ARTICLES)} KB articles")


def retrieve_kb_articles(query: str, category: str | None = None, n: int = 3) -> list[dict]:
    """Retrieve relevant KB articles for a query."""
    where_filter = {"category": category} if category else None

    results = kb.query(
        query_texts=[query],
        n_results=n,
        where=where_filter
    )

    articles = []
    for i, doc_id in enumerate(results["ids"][0]):
        articles.append({
            "id": doc_id,
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "relevance_score": 1 - results["distances"][0][i]
        })

    return articles
```

### Phase 3: Specialized Agents (2–3 hours)

```python
AGENT_PROMPTS = {
    "BILLING": """You are a billing support specialist.
Answer billing questions based only on the provided knowledge base articles.
Be clear about refund timelines, payment processing, and billing cycles.
If the issue isn't covered in the KB, say so clearly.""",

    "TECHNICAL": """You are a technical support specialist.
Provide step-by-step troubleshooting based on the knowledge base.
Ask for clarifying information if you need the customer's OS or browser version.
If you can't diagnose from available information, escalate.""",

    "ACCOUNT": """You are an account support specialist.
Help with account access, profile changes, and authentication issues.
Never ask for passwords. Guide customers through self-service steps first.""",

    "GENERAL": """You are a customer success specialist.
Answer product and feature questions based on the knowledge base.
Be helpful and suggest relevant features the customer might not know about.""",
}


def run_specialist_agent(message: str, intent: str, kb_articles: list[dict]) -> dict:
    """Run a specialized support agent for a given intent."""
    system_prompt = AGENT_PROMPTS.get(intent, AGENT_PROMPTS["GENERAL"])

    kb_context = "\n\n".join(
        f"KB Article: {a['metadata']['title']}\n{a['content']}"
        for a in kb_articles
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""Knowledge Base Articles:
{kb_context}

Customer Message:
{message}

Provide a helpful, concise response based on the knowledge base.
If the answer isn't in the knowledge base, clearly state that."""
            }
        ],
        max_tokens=400
    )

    return {
        "response": response.choices[0].message.content,
        "intent": intent,
        "kb_articles_used": [a["metadata"]["title"] for a in kb_articles]
    }
```

### Phase 4: Confidence Evaluation and Escalation (1–2 hours)

```python
CONFIDENCE_EVAL_PROMPT = """Evaluate this customer support response.

Customer message: {message}
Agent response: {response}
KB articles available: {kb_titles}

Rate confidence (0.0–1.0) based on:
- Does the response directly address the customer's issue?
- Is the response grounded in the KB articles?
- Does the response provide actionable next steps?
- Is there any ambiguity or missing information?

Return JSON:
{{
  "confidence_score": 0.0-1.0,
  "is_fully_resolved": true/false,
  "missing_info": "what additional info would help, or null",
  "escalation_reason": "reason to escalate, or null"
}}"""


def evaluate_response_confidence(message: str, response: str, kb_articles: list[dict]) -> dict:
    """Evaluate confidence in the generated response."""
    kb_titles = [a["metadata"]["title"] for a in kb_articles]

    eval_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": CONFIDENCE_EVAL_PROMPT.format(
                message=message,
                response=response,
                kb_titles=json.dumps(kb_titles)
            )
        }],
        response_format={"type": "json_object"},
        max_tokens=200,
        temperature=0
    )

    return json.loads(eval_response.choices[0].message.content)


ESCALATION_THRESHOLD = 0.75


def handle_support_ticket(message: str) -> dict:
    """Full support pipeline: classify → retrieve → respond → evaluate → route."""

    # Step 1: Classify intent
    classification = classify_intent(message)
    intent = classification["intent"]

    # Immediate escalation for flagged intents
    if intent == "ESCALATE":
        return {
            "status": "ESCALATED",
            "reason": "Flagged for immediate human review",
            "intent": intent,
            "original_message": message
        }

    # Step 2: Retrieve KB articles
    kb_articles = retrieve_kb_articles(message, category=intent, n=3)

    # Step 3: Generate response
    agent_result = run_specialist_agent(message, intent, kb_articles)

    # Step 4: Evaluate confidence
    confidence = evaluate_response_confidence(
        message,
        agent_result["response"],
        kb_articles
    )

    # Step 5: Route based on confidence
    if confidence["confidence_score"] >= ESCALATION_THRESHOLD and confidence["is_fully_resolved"]:
        return {
            "status": "RESOLVED",
            "response": agent_result["response"],
            "intent": intent,
            "confidence": confidence["confidence_score"],
            "kb_used": agent_result["kb_articles_used"]
        }
    else:
        return {
            "status": "ESCALATED",
            "reason": confidence.get("escalation_reason", "Low confidence"),
            "draft_response": agent_result["response"],  # Share with human agent
            "intent": intent,
            "confidence": confidence["confidence_score"],
            "missing_info": confidence.get("missing_info")
        }
```

### Phase 5: Test and Evaluate (1–2 hours)

```python
def run_evaluation():
    """Test the system on a set of support messages."""
    setup_knowledge_base()

    test_tickets = [
        # Should resolve:
        "Hi, I need to reset my password. I've been locked out of my account.",
        "How do I update my credit card on file?",
        "What's your refund policy?",
        "What's the difference between the Free and Pro plans?",

        # Should escalate:
        "Your product ruined my business. I'm contacting my lawyer.",
        "I was charged twice this month and nobody is helping me. This is fraud.",
        "I need to change my email but I no longer have access to my old email account.",
        "Why is feature X not working? I've tried everything and nothing works.",
    ]

    results = []
    resolved = 0
    escalated = 0

    for ticket in test_tickets:
        result = handle_support_ticket(ticket)
        results.append(result)

        if result["status"] == "RESOLVED":
            resolved += 1
        else:
            escalated += 1

        print(f"\nTicket: '{ticket[:60]}...'")
        print(f"  Status: {result['status']}")
        if result["status"] == "RESOLVED":
            print(f"  Confidence: {result['confidence']:.2f}")
            print(f"  KB Used: {result['kb_used']}")
        else:
            print(f"  Reason: {result.get('reason')}")

    print(f"\nSummary: {resolved} resolved, {escalated} escalated")
    print(f"Auto-resolution rate: {resolved/len(test_tickets)*100:.0f}%")

run_evaluation()
```

---

## Evaluation Criteria

| Criterion | Target | How to Measure |
|-----------|--------|----------------|
| Auto-resolution rate | 60–75% of tickets | Count resolved vs escalated |
| False resolution rate | < 5% | Manual review of "RESOLVED" tickets |
| Intent classification accuracy | > 90% | Test 20 labeled tickets |
| Escalation recall (catches hard cases) | 100% | All ESCALATE-labeled tickets escalated |
| Response relevance | KB articles match issue | Manual review of top 3 retrieved articles |

---

## Stretch Goals

1. **Conversation history**: Handle multi-turn conversations where context from prior messages matters
2. **Feedback loop**: Add a thumbs-up/thumbs-down mechanism; store feedback in DB for later fine-tuning
3. **Template generation**: For the top 10 most common ticket types, generate response templates
4. **SLA tracking**: Track response time and alert when a ticket is approaching its SLA deadline
5. **Analytics dashboard**: Build a simple dashboard showing resolution rate, escalation rate, and top issues

---

## Common Failure Modes

- **Confident but wrong**: High confidence score on an incorrect answer → add ground-truth test set, human review sample
- **Over-escalation**: Too many tickets escalated → lower threshold or add a "not enough information" response type
- **Under-escalation**: Sensitive tickets not caught → improve ESCALATE intent examples in the classifier prompt
- **KB miss**: Right intent, wrong articles → improve KB coverage or use hybrid BM25 + vector search
