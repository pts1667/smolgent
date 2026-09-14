"""Set DEEPSEEK_API_KEY, then run: python python/examples/streaming_agent.py."""

import os
from contextlib import closing

from smolgent import Agent, tool


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


if __name__ == "__main__":
    agent = Agent.deepseek(os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"), tools=[add])
    try:
        with closing(agent.stream("Use add to calculate 123 + 456.")) as events:
            for event in events:
                if event.type == "text_delta":
                    print(event.text, end="", flush=True)
                elif event.type == "tool_started":
                    print(f"\n[Calling {event.data['call']['function']['name']}]", flush=True)
        print()
    except KeyboardInterrupt:
        print("\nCancelled.")
        raise SystemExit(130)
