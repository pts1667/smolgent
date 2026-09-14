"""A Python tool called by the Rust agent loop. Requires OPENROUTER_API_KEY."""

from smolgent import Agent, tool


@tool(parameters={
    "type": "object",
    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    "required": ["a", "b"],
})
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


agent = Agent.openrouter("openrouter/auto", tools=[add])
print(agent.run("Use add to calculate 12345 + 67890."))
