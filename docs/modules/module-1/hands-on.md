---
sidebar_position: 3
title: "Hands-on"
description: Module 1 practical exercises
---

# Module 1 Hands-on Exercises

Practice the concepts from Module 1 with these guided exercises.

## Exercise 1: Basic LLM Interaction

Set up your first LangChain application with OpenAI.

### Setup

```python
# Install required packages
# pip install langchain langchain-openai python-dotenv

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)
```

### Basic Invocation

```python
# Simple text input
response = llm.invoke("What is Agentic AI?")
print(response.content)
```

### Using Messages

```python
from langchain_core.messages import SystemMessage, HumanMessage

messages = [
    SystemMessage(content="You are an expert in AI and machine learning."),
    HumanMessage(content="Explain what makes an AI system 'agentic'.")
]

response = llm.invoke(messages)
print(response.content)
```

### Exercise Task

:::tip Try It Yourself
1. Create an LLM instance with `temperature=0`
2. Ask it to explain "prompt engineering"
3. Compare the output with `temperature=1.0`
:::

---

## Exercise 2: Prompt Templates

Create reusable prompts for consistent interactions.

### Basic Template

```python
from langchain_core.prompts import ChatPromptTemplate

# Create a template with placeholders
template = ChatPromptTemplate.from_template(
    "Explain {topic} as if I'm a {audience}."
)

# Format the template
prompt = template.invoke({
    "topic": "neural networks",
    "audience": "5-year-old"
})

print(prompt)
```

### Template with System Message

```python
template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Always be {style}."),
    ("human", "{question}")
])

prompt = template.invoke({
    "role": "senior software engineer",
    "style": "concise and practical",
    "question": "How do I structure a Python project?"
})
```

### Template with Multiple Examples

```python
from langchain_core.prompts import FewShotChatMessagePromptTemplate

examples = [
    {"input": "happy", "output": "joyful"},
    {"input": "sad", "output": "melancholic"},
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])

few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "Provide a more sophisticated synonym."),
    few_shot,
    ("human", "{word}")
])
```

### Exercise Task

:::tip Try It Yourself
Create a prompt template that:
1. Sets a system role as "code reviewer"
2. Takes a `language` and `code` parameter
3. Asks for feedback on the code
:::

---

## Exercise 3: LCEL Chains

Build composable chains with LangChain Expression Language.

### Simple Chain

```python
from langchain_core.output_parsers import StrOutputParser

# Create the chain
chain = template | llm | StrOutputParser()

# Invoke with parameters
result = chain.invoke({
    "role": "Python expert",
    "style": "detailed",
    "question": "What are decorators?"
})

print(result)
```

### Chain with RunnablePassthrough

```python
from langchain_core.runnables import RunnablePassthrough

# Pass input through unchanged
chain = (
    {"question": RunnablePassthrough()}
    | template
    | llm
    | StrOutputParser()
)

result = chain.invoke("What is LCEL?")
```

### Parallel Chains

```python
from langchain_core.runnables import RunnableParallel

# Run multiple chains in parallel
parallel_chain = RunnableParallel(
    summary=summary_chain,
    keywords=keywords_chain,
    sentiment=sentiment_chain
)

results = parallel_chain.invoke({"text": "Your input text here"})
# results["summary"], results["keywords"], results["sentiment"]
```

### Streaming

```python
# Stream responses token by token
for chunk in chain.stream({"question": "Explain transformers"}):
    print(chunk, end="", flush=True)
```

### Exercise Task

:::tip Try It Yourself
Create a chain that:
1. Takes a topic as input
2. Generates both an explanation AND a quiz question
3. Returns both results in a structured format
:::

---

## Exercise 4: Output Parsers

Structure LLM responses into usable formats.

### String Parser

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
chain = prompt | llm | parser

# Returns plain string
result = chain.invoke({"topic": "AI"})
```

### JSON Parser

```python
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()

prompt = ChatPromptTemplate.from_template("""
Extract information from this text and return as JSON:

Text: {text}

Return a JSON object with: title, main_points (array), summary
""")

chain = prompt | llm | parser
result = chain.invoke({"text": "Your text here..."})
# result is now a Python dict
```

### Pydantic Parser (Structured Output)

```python
from pydantic import BaseModel, Field
from typing import List

class TaskAnalysis(BaseModel):
    """Analysis of a task description."""
    task_name: str = Field(description="Name of the task")
    complexity: str = Field(description="low, medium, or high")
    estimated_hours: int = Field(description="Estimated hours to complete")
    subtasks: List[str] = Field(description="List of subtasks")

# Use structured output
structured_llm = llm.with_structured_output(TaskAnalysis)

result = structured_llm.invoke(
    "Analyze this task: Build a REST API for user authentication"
)

print(f"Task: {result.task_name}")
print(f"Complexity: {result.complexity}")
print(f"Hours: {result.estimated_hours}")
print(f"Subtasks: {result.subtasks}")
```

### Exercise Task

:::tip Try It Yourself
Create a structured output parser that:
1. Defines a `CodeReview` Pydantic model
2. Has fields for: `issues`, `suggestions`, `rating`
3. Use it to analyze a code snippet
:::

---

## Exercise 5: Putting It All Together

Build a complete mini-application.

### Code Explanation Assistant

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import List

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# Define structured output
class CodeExplanation(BaseModel):
    summary: str = Field(description="Brief summary of what the code does")
    concepts: List[str] = Field(description="Key programming concepts used")
    complexity: str = Field(description="Beginner, Intermediate, or Advanced")
    suggestions: List[str] = Field(description="Improvement suggestions")

# Create the chain
explanation_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert code reviewer and teacher.
    Analyze the provided code and explain it clearly."""),
    ("human", "Please analyze this {language} code:\n\n```\n{code}\n```")
])

# Structured output chain
structured_llm = llm.with_structured_output(CodeExplanation)
chain = explanation_prompt | structured_llm

# Use the chain
code_sample = '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
'''

result = chain.invoke({
    "language": "Python",
    "code": code_sample
})

print(f"Summary: {result.summary}")
print(f"Concepts: {', '.join(result.concepts)}")
print(f"Complexity: {result.complexity}")
print(f"Suggestions:")
for suggestion in result.suggestions:
    print(f"  - {suggestion}")
```

---

## Notebook Labs

Access the Jupyter notebooks for interactive practice:

| Notebook | Topics Covered |
|----------|---------------|
| `1_langchain_openai.ipynb` | LLM setup, basic invocation |
| `2_prompt_templates.ipynb` | Template creation, few-shot |
| `3_lcel.ipynb` | Chain composition, streaming |
| `4_output_parser.ipynb` | Structured outputs |

## Checklist

Before moving to Module 2, ensure you can:

- [ ] Initialize LLMs with different providers
- [ ] Create prompt templates with variables
- [ ] Build chains using LCEL
- [ ] Parse outputs into structured formats
- [ ] Handle streaming responses

## Next Steps

Once you've completed these exercises:

1. Review the [Resources](./resources) for additional learning
2. Proceed to [Module 2: Advanced LangChain](/learn/modules/module-2)
