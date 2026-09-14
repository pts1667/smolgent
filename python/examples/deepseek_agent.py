"""Set DEEPSEEK_API_KEY, then run: python python/examples/deepseek_agent.py.

Optionally set DEEPSEEK_MODEL to choose a model (default: deepseek-flash).
"""

import os

from smolgent import Agent, tool


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


if __name__ == "__main__":
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
    agent = Agent.deepseek(model, tools=[add])
    print(f"Waiting for {model} (up to 120 seconds per API request)...", flush=True)
    try:
        print(agent.run("Use add to calculate 123 + 456."))
    except KeyboardInterrupt:
        print("\nCancelled.")
        raise SystemExit(130)
