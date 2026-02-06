---
sidebar_position: 3
title: Implementation
description: Capstone implementation guide
---

# Capstone Implementation

A step-by-step guide to building your capstone project.

## Phase 1: Setup & Planning

### Day 1: Project Setup

```bash
# Create project structure
mkdir capstone && cd capstone
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Create directories
mkdir -p src/{agents,tools,rag,graph}
mkdir -p data tests notebooks
touch src/__init__.py requirements.txt .env.example README.md
```

### Requirements File

```txt title="requirements.txt"
langchain>=0.3.0
langchain-openai>=0.3.0
langgraph>=0.2.0
chromadb>=1.0.0
python-dotenv>=1.0.0
pydantic>=2.0.0
```

### Environment Setup

```bash title=".env.example"
OPENAI_API_KEY=your-key-here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=capstone
```

## Phase 2: Core Components

### Step 1: Define State

```python title="src/graph/state.py"
from typing import TypedDict, Annotated, List, Optional
from operator import add

class CapstoneState(TypedDict):
    """State for the capstone workflow."""
    # Input
    query: str
    user_id: Optional[str]

    # Processing
    query_type: Optional[str]
    retrieved_docs: List[str]
    tool_outputs: Annotated[List[str], add]

    # Agent work
    reasoning: Optional[str]
    plan: Optional[str]

    # Output
    response: Optional[str]
    error: Optional[str]

    # Metadata
    iteration: int
```

### Step 2: Build RAG Pipeline

```python title="src/rag/retriever.py"
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class KnowledgeBase:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.embeddings = OpenAIEmbeddings()
        self.persist_dir = persist_dir
        self.vectorstore = None

    def load_documents(self, data_dir: str):
        """Load and index documents."""
        loader = DirectoryLoader(data_dir, glob="**/*.txt")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )

    def search(self, query: str, k: int = 4) -> List[str]:
        """Search for relevant documents."""
        if not self.vectorstore:
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )

        results = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in results]
```

### Step 3: Create Tools

```python title="src/tools/custom_tools.py"
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional

class SearchParams(BaseModel):
    query: str = Field(description="Search query")
    category: Optional[str] = Field(default=None)

@tool(args_schema=SearchParams)
def search_knowledge(query: str, category: Optional[str] = None) -> str:
    """Search the knowledge base for information."""
    from src.rag.retriever import KnowledgeBase

    kb = KnowledgeBase()
    results = kb.search(query)

    if not results:
        return "No relevant information found."

    return "\n\n".join(results)

@tool
def analyze_query(query: str) -> str:
    """Analyze the user's query to understand intent."""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = f"""Analyze this query and identify:
    1. Main intent
    2. Key entities
    3. Required actions

    Query: {query}"""

    return llm.invoke(prompt).content

@tool
def format_response(content: str, style: str = "professional") -> str:
    """Format content into a polished response."""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = f"""Format this content in a {style} style:

    {content}

    Make it clear, well-structured, and helpful."""

    return llm.invoke(prompt).content
```

### Step 4: Build Agents

```python title="src/agents/main_agent.py"
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from src.tools.custom_tools import search_knowledge, analyze_query, format_response

class MainAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.tools = [search_knowledge, analyze_query, format_response]
        self.agent = self._create_agent()

    def _create_agent(self) -> AgentExecutor:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant.

            Use your tools to:
            1. Search for relevant information
            2. Analyze queries to understand needs
            3. Format responses professionally

            Always be helpful and accurate."""),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])

        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)

    def run(self, query: str) -> str:
        result = self.agent.invoke({
            "input": query,
            "chat_history": []
        })
        return result["output"]
```

### Step 5: Create Workflow

```python title="src/graph/workflow.py"
from langgraph.graph import StateGraph, END
from src.graph.state import CapstoneState
from src.rag.retriever import KnowledgeBase
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
kb = KnowledgeBase()

def understand_query(state: CapstoneState) -> dict:
    """Understand what the user is asking."""
    query = state["query"]

    prompt = f"""Classify this query:
    Query: {query}

    Return one of: question, request, task, other"""

    response = llm.invoke(prompt)
    return {"query_type": response.content.strip().lower()}

def retrieve_context(state: CapstoneState) -> dict:
    """Retrieve relevant documents."""
    docs = kb.search(state["query"])
    return {"retrieved_docs": docs}

def generate_response(state: CapstoneState) -> dict:
    """Generate final response."""
    context = "\n".join(state.get("retrieved_docs", []))
    tool_info = "\n".join(state.get("tool_outputs", []))

    prompt = f"""Based on this context and information:

    Context: {context}
    Additional Info: {tool_info}

    Answer this query: {state['query']}"""

    response = llm.invoke(prompt)
    return {"response": response.content}

def should_use_tools(state: CapstoneState) -> str:
    """Decide if tools are needed."""
    query_type = state.get("query_type", "")
    if query_type in ["task", "request"]:
        return "tools"
    return "respond"

# Build graph
graph = StateGraph(CapstoneState)
graph.add_node("understand", understand_query)
graph.add_node("retrieve", retrieve_context)
graph.add_node("respond", generate_response)

graph.set_entry_point("understand")
graph.add_edge("understand", "retrieve")
graph.add_conditional_edges(
    "retrieve",
    should_use_tools,
    {"tools": "respond", "respond": "respond"}  # Simplified
)
graph.add_edge("respond", END)

workflow = graph.compile()
```

### Step 6: Main Entry Point

```python title="src/main.py"
from dotenv import load_dotenv
from src.graph.workflow import workflow
from src.graph.state import CapstoneState

load_dotenv()

def run_capstone(query: str, user_id: str = None) -> str:
    """Run the capstone project."""
    initial_state: CapstoneState = {
        "query": query,
        "user_id": user_id,
        "query_type": None,
        "retrieved_docs": [],
        "tool_outputs": [],
        "reasoning": None,
        "plan": None,
        "response": None,
        "error": None,
        "iteration": 0
    }

    try:
        result = workflow.invoke(initial_state)
        return result.get("response", "Unable to generate response.")
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    query = input("Enter your query: ")
    response = run_capstone(query)
    print(f"\nResponse: {response}")
```

## Phase 3: Testing

### Unit Tests

```python title="tests/test_tools.py"
import pytest
from src.tools.custom_tools import analyze_query

def test_analyze_query():
    result = analyze_query.invoke({"query": "What is AI?"})
    assert isinstance(result, str)
    assert len(result) > 0
```

### Integration Test

```python title="tests/test_workflow.py"
from src.main import run_capstone

def test_full_workflow():
    response = run_capstone("What can you help me with?")
    assert response is not None
    assert len(response) > 0
```

## Phase 4: Documentation

### README Template

```markdown title="README.md"
# Capstone Project: [Your Project Name]

## Overview
[Brief description of what your project does]

## Features
- Feature 1
- Feature 2
- Feature 3

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
```

## Usage
```python
from src.main import run_capstone

response = run_capstone("Your query here")
print(response)
```

## Architecture
[Include your architecture diagram]

## Example Queries
1. "Example query 1" → Expected result
2. "Example query 2" → Expected result
```

## Next Steps

Complete your implementation and prepare for the [Review](./review).
