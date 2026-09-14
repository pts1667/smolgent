# smolgent for Python

A small Python API over smolgent's Rust provider transport, conversation history,
tool loop, filesystem tools, and context compaction. Requires Python 3.10 or newer.

## Install from this checkout

With Rust and Python installed, run from the repository root:

```shell
python -m pip install .
```

For development, create and activate a virtual environment, then:

```shell
python -m pip install maturin
maturin develop
python -m unittest discover -s python/tests -v
```

`maturin build --release` produces a wheel in `target/wheels`. A prebuilt wheel
does not require a Rust compiler on the user's machine. Build wheels separately
for each target OS/architecture. This package is not yet published to PyPI.

## Three-line OpenRouter agent

Set `OPENROUTER_API_KEY` in your environment (`$env:OPENROUTER_API_KEY = "..."`
in PowerShell, or `export OPENROUTER_API_KEY="..."` in a POSIX shell):

```python
from smolgent import Agent
agent = Agent.openrouter("openrouter/auto", read_roots=["."])
print(agent.run("Read README.md and summarize this project in three sentences."))
```

Choose a tool-capable model your account can access, or use OpenRouter's automatic
routing as above. Run subsequent prompts on the same object to retain history.
`run()` returns a `Response`: printing it displays the final answer; `.text`,
`.message`, `.reasoning`, and `.raw` expose the text and provider metadata.

No filesystem tools are enabled unless roots are supplied. `read_roots` enables
`read` and `ripgrep`; `write_roots` also enables create, delete, and patch tools.
Write roots grant read access too. Paths resolve against the process working
directory. The search tool requires `rg` on PATH. Custom Python tools execute
with the application's permissions; filesystem roots apply to built-in tools.

## Async applications and notebooks

```python
from smolgent import Agent

agent = Agent.openrouter("openrouter/auto")
response = await agent.arun("Say hello in one sentence.")
print(response.text)
```

In scripts, put this in an async function and call it using `asyncio.run()`.
`run()` detects an already-running event loop and directs you to `arun()`.
Separate agents can run concurrently; overlapping calls on the same agent raise
`RuntimeError`. Reading history or resetting while a run is active also raises.

Failed/cancelled runs leave the previous conversation intact. Tool side effects
that already occurred remain. Async Python tools receive cancellation; a running
synchronous Python tool cannot be forcibly stopped. `timeout=120.0` limits each HTTP request. Wrap `arun()`
in `asyncio.wait_for()` for an overall deadline. `max_tool_rounds=32` bounds the
tool loop. `compaction=False` disables the Rust core's default context compaction.

## Python tools

Tools use an explicit JSON Schema. Both ordinary and `async def` functions work:

```python
from smolgent import Agent, tool

@tool(parameters={
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
})
def weather(city: str):
    """Get an illustrative weather report for a city."""
    return {"city": city, "temperature_c": 20}

agent = Agent.openrouter("openrouter/auto", tools=[weather])
print(agent.run("Use weather to get the report for London."))
```

The decorator returns a `Tool` object. Alternatively construct
`Tool(name, description, parameters, handler)` directly. Functions receive keyword
arguments and return JSON-serializable values. Async tools run on the caller's
event loop; synchronous tools run in its thread pool. Both preserve context
variables. Tool exceptions become model-visible errors, matching Rust behavior.
Provider/transport failures raise `SmolgentError`, a subclass of `RuntimeError`.

## Credentials and other endpoints

Pass `api_key="..."` explicitly, or use the Rust keyring backend with an
application-specific ID. Explicit credentials take precedence over the environment:

```python
from smolgent import Agent, set_api_key

set_api_key("my-app/openrouter", "...")  # Store once, in a separate setup step.
agent = Agent.openrouter("openrouter/auto", keyring_id="my-app/openrouter")
```

The keyring service is `smolgent`, shared with the Rust examples; native stores
are configured on Windows and Linux. Other platforms require backend support.

```python
agent = Agent.llama_cpp("local-model", base_url="http://127.0.0.1:8080")
agent = Agent.compatible(
    "my-model",
    chat_completions_url="http://localhost:8000/v1/chat/completions",
    api_key="optional-key",
)
```

## History and multimodal input

`agent.messages` returns a detached list of message dictionaries, preserving
reasoning, annotations, and tool results. `agent.reset()` restores the initial
system prompt. Override that prompt with `system_prompt="..."` at construction.

Pass ordered content dictionaries for direct multimodal input:

```python
response = agent.run([
    {"type": "text", "text": "Describe this image."},
    {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
])
```

The model must support the requested media. The core also accepts file, audio,
and video content parts; see the main README for formats. This first Python API
uses the text file reader. Model-aware media reading, event/telemetry subscriptions,
and automatic JSON Schema generation from Python annotations are not exposed yet.
