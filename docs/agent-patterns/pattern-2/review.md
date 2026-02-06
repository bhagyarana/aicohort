---
sidebar_position: 3
title: "Review"
description: Pattern 2 review and discussion
---

# Pattern 2: Review

Review your tool-using agent and discuss advanced patterns.

## Code Review Points

### 1. Tool Schema Design

**Well-designed schema:**
```python
class OrderParams(BaseModel):
    product_id: str = Field(
        description="Unique product identifier (e.g., P001)"
    )
    quantity: int = Field(
        ge=1, le=100,  # Validation
        description="Number of items to order"
    )
```

### 2. Error Handling

All tools should handle:
- Invalid inputs
- External service failures
- Timeout scenarios

### 3. Tool Composition

Consider how tools work together:
```python
# Good: Tools that complement each other
search_products  # Find items
get_details      # Learn more
place_order      # Take action
```

## Discussion Questions

1. **Tool Granularity**
   - When to split tools?
   - When to combine?

2. **Input Validation**
   - Where to validate?
   - How to communicate errors?

3. **Side Effects**
   - How to handle actions?
   - Confirmation patterns?

## Advanced Patterns

### Parallel Tool Execution

```python
from langchain_core.runnables import RunnableParallel

parallel = RunnableParallel(
    price=get_price,
    inventory=check_inventory,
    shipping=estimate_shipping
)

results = parallel.invoke({"product_id": "P001"})
```

### Tool Result Caching

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_search(query: str, category: str) -> str:
    return expensive_search(query, category)
```

## Best Practices

1. **Clear tool descriptions** help the LLM choose correctly
2. **Validate inputs** before processing
3. **Return structured data** when possible
4. **Handle errors gracefully**
5. **Log tool usage** for debugging

## Next Pattern

Ready for the final pattern? Continue to [Pattern 3: Multi-Agent System](/learn/agent-patterns/pattern-3).
