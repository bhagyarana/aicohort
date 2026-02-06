---
sidebar_position: 2
title: "Hands-on"
description: Build a tool-using agent
---

# Pattern 2: Hands-on

Build a sophisticated agent with multiple custom tools.

## Step 1: Define Tool Schemas

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from langchain_core.tools import tool

class ProductSearchParams(BaseModel):
    query: str = Field(description="Search query")
    category: Optional[str] = Field(default=None, description="Product category")
    max_price: Optional[float] = Field(default=None, description="Maximum price")
    limit: int = Field(default=5, description="Number of results")

class OrderParams(BaseModel):
    product_id: str = Field(description="Product ID to order")
    quantity: int = Field(default=1, description="Quantity to order")
    shipping: str = Field(default="standard", description="Shipping speed")
```

## Step 2: Implement Tools

```python
# Mock data
PRODUCTS = [
    {"id": "P001", "name": "Laptop", "category": "electronics", "price": 999.99},
    {"id": "P002", "name": "Headphones", "category": "electronics", "price": 199.99},
    {"id": "P003", "name": "Desk Chair", "category": "furniture", "price": 299.99},
    {"id": "P004", "name": "Monitor", "category": "electronics", "price": 449.99},
]

@tool(args_schema=ProductSearchParams)
def search_products(
    query: str,
    category: Optional[str] = None,
    max_price: Optional[float] = None,
    limit: int = 5
) -> str:
    """Search for products in the catalog."""
    results = PRODUCTS

    # Filter by query
    results = [p for p in results if query.lower() in p["name"].lower()]

    # Filter by category
    if category:
        results = [p for p in results if p["category"] == category]

    # Filter by price
    if max_price:
        results = [p for p in results if p["price"] <= max_price]

    # Limit results
    results = results[:limit]

    if not results:
        return "No products found matching your criteria."

    return "\n".join([
        f"- {p['name']} (ID: {p['id']}) - ${p['price']}"
        for p in results
    ])

@tool
def get_product_details(product_id: str) -> str:
    """Get detailed information about a product."""
    for product in PRODUCTS:
        if product["id"] == product_id:
            return f"""
Product: {product['name']}
ID: {product['id']}
Category: {product['category']}
Price: ${product['price']}
In Stock: Yes
Rating: 4.5/5
"""
    return f"Product {product_id} not found."

@tool(args_schema=OrderParams)
def place_order(product_id: str, quantity: int = 1, shipping: str = "standard") -> str:
    """Place an order for a product."""
    for product in PRODUCTS:
        if product["id"] == product_id:
            total = product["price"] * quantity
            shipping_cost = 0 if shipping == "standard" else 15.99

            return f"""
Order Placed Successfully!
Product: {product['name']}
Quantity: {quantity}
Subtotal: ${total:.2f}
Shipping ({shipping}): ${shipping_cost:.2f}
Total: ${total + shipping_cost:.2f}
Order ID: ORD-{product_id}-001
"""
    return f"Cannot place order: Product {product_id} not found."

@tool
def check_order_status(order_id: str) -> str:
    """Check the status of an order."""
    # Mock status
    return f"""
Order {order_id}:
Status: Processing
Estimated Delivery: 3-5 business days
Tracking: Will be available once shipped
"""
```

## Step 3: Create the Agent

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

tools = [
    search_products,
    get_product_details,
    place_order,
    check_order_status
]

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful shopping assistant.

Your capabilities:
- Search for products
- Get product details
- Place orders
- Check order status

Be helpful, concise, and guide users through their shopping experience.
Always confirm before placing orders."""),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

## Step 4: Test the Agent

```python
# Test 1: Search products
result = executor.invoke({
    "input": "I'm looking for electronics under $500",
    "chat_history": []
})
print(result["output"])
```

```python
# Test 2: Get details and order
result = executor.invoke({
    "input": "Tell me more about the headphones and place an order for 2",
    "chat_history": []
})
print(result["output"])
```

```python
# Test 3: Complex query
result = executor.invoke({
    "input": "Find the cheapest electronics, get details, and order it with express shipping",
    "chat_history": []
})
print(result["output"])
```

## Step 5: Add API Integration

```python
import aiohttp
import asyncio

@tool
async def fetch_live_price(product_id: str) -> str:
    """Fetch live pricing from external API."""
    # Mock API call
    async with aiohttp.ClientSession() as session:
        # In real implementation, call actual API
        await asyncio.sleep(0.1)  # Simulate API latency
        return f"Live price for {product_id}: $499.99 (updated just now)"

@tool
def get_shipping_estimate(zip_code: str, product_id: str) -> str:
    """Get shipping estimate for a zip code."""
    # Mock calculation
    return f"""
Shipping to {zip_code}:
- Standard (5-7 days): Free
- Express (2-3 days): $15.99
- Overnight: $29.99
"""
```

## Step 6: Error Handling

```python
from functools import wraps

def handle_tool_errors(func):
    """Decorator to handle tool errors gracefully."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return f"Error: {str(e)}. Please try again or contact support."
    return wrapper

@tool
@handle_tool_errors
def risky_operation(data: str) -> str:
    """An operation that might fail."""
    # Implementation that could raise errors
    pass
```

## Exercises

### Exercise 1: Add Inventory Tool

```python
@tool
def check_inventory(product_id: str) -> str:
    """Check inventory levels for a product."""
    # Implement inventory check
    pass
```

### Exercise 2: Multi-Tool Workflow

Create a workflow that:
1. Searches for products
2. Compares prices
3. Checks inventory
4. Places order for best option

### Exercise 3: Conversational Context

Maintain context across turns:
```python
# User: "Search for laptops"
# User: "What about the first one?"
# User: "Order it"
```

## Checklist

- [ ] Defined structured tool inputs
- [ ] Implemented multiple tools
- [ ] Created tool-calling agent
- [ ] Tested multi-tool scenarios
- [ ] Added error handling

## Next Steps

Continue to the [Review](./review) session.
