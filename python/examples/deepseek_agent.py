"""Set DEEPSEEK_API_KEY, then run: python python/examples/deepseek_agent.py"""

from smolgent import Agent, tool


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


if __name__ == "__main__":
    agent = Agent.deepseek(tools=[add])
    print(agent.run("Use add to calculate 123 + 456."))
