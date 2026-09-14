"""Set OPENROUTER_API_KEY, then run this from the project you want to inspect."""

from smolgent import Agent

agent = Agent.openrouter("openrouter/auto", read_roots=["."])
print(agent.run("Read README.md and summarize this project in three sentences."))
