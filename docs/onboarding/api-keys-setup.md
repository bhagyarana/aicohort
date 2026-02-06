---
sidebar_position: 5
title: API Keys Setup
description: Configure API keys for LLM providers
---

# API Keys Setup

This guide walks you through obtaining and configuring API keys for various LLM providers used in the training.

## Overview

We'll use multiple LLM providers throughout the training:

| Provider | Use Case | Cost |
|----------|----------|------|
| **OpenAI** | Primary LLM (GPT-4) | Pay per token |
| **Google AI** | Gemini models | Free tier available |
| **Groq** | Fast inference | Free tier available |
| **Anthropic** | Claude models | Pay per token |

:::tip Start Free
Begin with **Groq** or **Google AI** for free-tier access during learning.
:::

## Obtaining API Keys

### OpenAI

1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign up or log in
3. Navigate to **API Keys** section
4. Click **Create new secret key**
5. Copy and save the key securely

```
Your key will look like: sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### Google AI (Gemini)

1. Go to [ai.google.dev](https://ai.google.dev)
2. Click **Get API key in Google AI Studio**
3. Sign in with Google account
4. Click **Create API key**
5. Copy the key

```
Your key will look like: AIzaxxxxxxxxxxxxxxxxxxxxxxxxx
```

### Groq

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for an account
3. Navigate to **API Keys**
4. Create a new key
5. Copy and save it

```
Your key will look like: gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### Anthropic

1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Sign up for an account
3. Navigate to **API Keys**
4. Create a new key
5. Copy and save it

```
Your key will look like: sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## Configuring Environment Variables

### Create .env File

In your project root, create a `.env` file:

```bash title=".env"
# OpenAI
OPENAI_API_KEY=sk-your-openai-key-here

# Google AI
GOOGLE_API_KEY=AIza-your-google-key-here

# Groq
GROQ_API_KEY=gsk_your-groq-key-here

# Anthropic
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# Optional: LangSmith for tracing
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=your-langsmith-key-here
```

:::danger Security Warning
**Never commit your .env file to git!** Make sure `.env` is in your `.gitignore`.
:::

### Load Environment Variables

In your Python code, load the variables:

```python
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access keys
openai_key = os.getenv("OPENAI_API_KEY")
google_key = os.getenv("GOOGLE_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")
```

## Testing Your Keys

### Test OpenAI

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke("Say hello!")
print(response.content)
```

### Test Google AI

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
response = llm.invoke("Say hello!")
print(response.content)
```

### Test Groq

```python
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.1-8b-instant")
response = llm.invoke("Say hello!")
print(response.content)
```

### Complete Test Script

```python title="test_api_keys.py"
"""Test all configured API keys."""
from dotenv import load_dotenv
import os

load_dotenv()

def test_openai():
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini")
        response = llm.invoke("Say 'OpenAI working!'")
        print(f"OpenAI: {response.content}")
        return True
    except Exception as e:
        print(f"OpenAI: Failed - {e}")
        return False

def test_google():
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        response = llm.invoke("Say 'Google AI working!'")
        print(f"Google AI: {response.content}")
        return True
    except Exception as e:
        print(f"Google AI: Failed - {e}")
        return False

def test_groq():
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(model="llama-3.1-8b-instant")
        response = llm.invoke("Say 'Groq working!'")
        print(f"Groq: {response.content}")
        return True
    except Exception as e:
        print(f"Groq: Failed - {e}")
        return False

if __name__ == "__main__":
    print("Testing API Keys...\n")

    results = {
        "OpenAI": test_openai(),
        "Google AI": test_google(),
        "Groq": test_groq(),
    }

    print("\n" + "="*40)
    print("Summary:")
    for provider, status in results.items():
        emoji = "OK" if status else "FAILED"
        print(f"  {provider}: {emoji}")
```

## Best Practices

### 1. Use Environment Variables

Never hardcode API keys:

```python
# Bad - Never do this!
api_key = "sk-abc123..."

# Good - Use environment variables
api_key = os.getenv("OPENAI_API_KEY")
```

### 2. Key Rotation

Periodically rotate your API keys:
- Create a new key
- Update your `.env` file
- Delete the old key from the provider's dashboard

### 3. Monitor Usage

Set up usage alerts to avoid unexpected charges:

| Provider | Billing Page |
|----------|--------------|
| OpenAI | [Usage Dashboard](https://platform.openai.com/usage) |
| Google AI | [Cloud Console](https://console.cloud.google.com) |
| Anthropic | [Console Billing](https://console.anthropic.com) |

### 4. Use Spending Limits

Most providers allow you to set spending limits:

- **OpenAI**: Set monthly budget in billing settings
- **Google AI**: Configure quotas in Cloud Console
- **Anthropic**: Set usage limits in console

## Troubleshooting

### "Invalid API Key" Error

```python
# Check if key is loaded
import os
print(f"Key loaded: {bool(os.getenv('OPENAI_API_KEY'))}")
print(f"Key prefix: {os.getenv('OPENAI_API_KEY', '')[:10]}...")
```

### "Insufficient Quota" Error

- Check your billing settings
- Add payment method if on free tier
- Wait for quota reset (usually monthly)

### Environment Variables Not Loading

```python
# Ensure you're loading from the correct path
from dotenv import load_dotenv
import os

# Explicit path to .env file
load_dotenv(dotenv_path='/path/to/your/.env')

# Or load from current directory
load_dotenv()
```

## Security Checklist

Before proceeding, verify:

- [ ] `.env` file is in `.gitignore`
- [ ] API keys are not in any committed files
- [ ] Spending limits are configured
- [ ] At least one provider is working

## Next Steps

With your API keys configured, you're ready to start learning!

1. Head to [Module 1: LangChain Fundamentals](/learn/modules/module-1)
2. Run your first LangChain notebook
3. Start building AI agents!

---

:::info Need Help?
If you're having trouble with API keys, reach out in our Discord community for assistance.
:::
