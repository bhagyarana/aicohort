---
sidebar_position: 2
title: Requirements
description: Capstone project requirements
---

# Capstone Requirements

Detailed requirements for your capstone project.

## Functional Requirements

### Core Features

Your system must:

1. **Accept user queries** in natural language
2. **Retrieve relevant information** from a knowledge base
3. **Reason about the task** using LangGraph
4. **Execute actions** using custom tools
5. **Provide clear responses** to users

### Minimum Viable Product (MVP)

At minimum, implement:

```
User Query → Agent → [Tools/RAG] → Response
```

| Feature | Required | Description |
|---------|----------|-------------|
| Query handling | Yes | Accept and parse user input |
| RAG pipeline | Yes | Retrieve from documents |
| Tool execution | Yes | At least 2 custom tools |
| Error handling | Yes | Graceful failure recovery |
| Conversation | Optional | Multi-turn support |

## Technical Requirements

### Architecture

```python
# Your project should include these components

class CapstoneProject:
    def __init__(self):
        self.llm = ChatOpenAI(...)           # LLM setup
        self.embeddings = OpenAIEmbeddings()  # Embeddings
        self.vectorstore = Chroma(...)        # Vector store
        self.tools = [...]                    # Custom tools
        self.graph = StateGraph(...)          # LangGraph workflow

    def run(self, query: str) -> str:
        """Main entry point."""
        pass
```

### Required Technologies

| Category | Technology | Purpose |
|----------|------------|---------|
| LLM | OpenAI/Groq/Google | Core reasoning |
| Framework | LangChain + LangGraph | Agent building |
| Vector Store | ChromaDB or FAISS | Document retrieval |
| State | LangGraph State | Workflow management |

### Code Structure

```
capstone/
├── src/
│   ├── agents/           # Agent implementations
│   │   ├── __init__.py
│   │   ├── main_agent.py
│   │   └── specialist_agents.py
│   ├── tools/            # Custom tools
│   │   ├── __init__.py
│   │   ├── search_tools.py
│   │   └── action_tools.py
│   ├── rag/              # RAG components
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── retriever.py
│   ├── graph/            # LangGraph workflow
│   │   ├── __init__.py
│   │   ├── state.py
│   │   └── workflow.py
│   └── main.py           # Entry point
├── data/                 # Sample data
├── tests/                # Tests
├── notebooks/            # Development notebooks
├── requirements.txt
├── .env.example
└── README.md
```

## Documentation Requirements

### README.md

Must include:
- Project description
- Setup instructions
- Usage examples
- Architecture overview

### Code Documentation

- Docstrings for all functions
- Type hints
- Inline comments for complex logic

## Sample Implementation Outline

### 1. State Definition

```python
from typing import TypedDict, Annotated, List, Optional
from operator import add

class ProjectState(TypedDict):
    query: str
    context: Optional[str]
    research: Annotated[List[str], add]
    tool_results: Annotated[List[str], add]
    response: Optional[str]
    error: Optional[str]
```

### 2. Tool Examples

```python
@tool
def search_knowledge_base(query: str) -> str:
    """Search the project's knowledge base."""
    pass

@tool
def execute_action(action: str, params: dict) -> str:
    """Execute a specific action."""
    pass

@tool
def get_user_context(user_id: str) -> str:
    """Get context about a user."""
    pass
```

### 3. Workflow Structure

```python
# Graph nodes
graph.add_node("understand", understand_query)
graph.add_node("retrieve", retrieve_context)
graph.add_node("reason", reasoning_agent)
graph.add_node("act", action_agent)
graph.add_node("respond", generate_response)

# Edges
graph.set_entry_point("understand")
graph.add_edge("understand", "retrieve")
graph.add_conditional_edges("retrieve", route_after_retrieval)
graph.add_edge("reason", "act")
graph.add_edge("act", "respond")
graph.add_edge("respond", END)
```

## Deliverables

### Required Submissions

1. **Source Code**
   - Complete, working implementation
   - Clean, documented code

2. **Documentation**
   - README with setup instructions
   - Architecture diagram

3. **Demo**
   - Working demonstration
   - Example queries and responses

4. **Presentation**
   - 5-10 minute walkthrough
   - Technical decisions explained

## Tips for Success

1. **Start Simple**
   - Get basic flow working first
   - Add complexity iteratively

2. **Test Early**
   - Test each component separately
   - Integration test frequently

3. **Handle Errors**
   - Expect failures
   - Provide helpful error messages

4. **Document As You Go**
   - Don't leave docs for the end
   - Keep notes on decisions

## Questions?

If you're unsure about requirements:
1. Ask in the community Discord
2. Check with your instructor
3. Propose alternatives with justification
