# smolgent for Python

A small Python API over smolgent's Rust provider transport, conversation history,
tool loop, filesystem tools, and context compaction. Requires Python 3.10 or newer.

## Install from this checkout

With Rust and Python installed, run from the repository root:

```shell
python -m pip install .
```

For Pillow image objects, install the optional image extra:

```shell
python -m pip install ".[images]"
```

Audio, video, files, and images supplied as encoded bytes or URLs work without Pillow.

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

## DeepSeek

Set `DEEPSEEK_API_KEY`, then use the same agent and tool API:

```python
from smolgent import Agent
agent = Agent.deepseek("deepseek-flash", read_roots=["."])
print(agent.run("Read README.md and summarize this project in three sentences."))
```

The model argument defaults to `deepseek-flash`; any model ID can be supplied.
Explicit `api_key="..."` or `keyring_id="my-app/deepseek"` takes precedence over
the environment. Store a keyring key with `set_api_key("my-app/deepseek", key)`.
`base_url="https://api.deepseek.com/v1"` optionally overrides the server root or
path prefix; `/chat/completions` is appended automatically.

Omitting thinking options uses the API defaults. DeepSeek currently enables
thinking by default. Use `Agent.deepseek(thinking=False)` to disable it, or
`Agent.deepseek(reasoning_effort="max")` to select effort. Supported Python values
are `"low"`, `"high"`, `"max"`, and `"none"` (disables thinking). Contradictory
thinking and effort settings raise `ValueError`. Reasoning is returned in
`response.reasoning` and preserved across tool calls and subsequent user turns,
as required by the [DeepSeek thinking guide](https://api-docs.deepseek.com/guides/thinking_mode/).

Custom `@tool` functions and `await agent.arun(...)` work identically. See the
[tool example](examples/deepseek_agent.py).

The tool example accepts `DEEPSEEK_MODEL` to select a model. For example, in
PowerShell:

```powershell
$env:DEEPSEEK_MODEL = "deepseek-v4-pro"
.\.venv\Scripts\python.exe python\examples\deepseek_agent.py
```

`Ctrl+C` cancels a pending request. If one model keeps waiting, try another model
available to your account; a working model-list endpoint alone does not confirm
that completion requests are being served promptly.

[deceive_test.py](examples/deceive_test.py) runs two separate DeepSeek agents in a
human-or-AI guessing game. Each is told its opponent may be human or AI and its
task is to win. They exchange messages until one calls `submit_verdict`: a correct
verdict wins, an incorrect verdict awards the win to the opponent. Both are AI,
but this ground truth is kept in the host. The first player is chosen randomly;
an immediate guess is allowed. Reasoning is never relayed to the opponent.

```shell
python python/examples/deceive_test.py
```

Use `--model MODEL` (or `DEEPSEEK_MODEL`) to choose a model. Play continues until
a verdict by default; `--max-turns 20` optionally stops after 20 individual turns
with no winner if neither submits. Each turn makes API requests using
`DEEPSEEK_API_KEY`.

The game allows 600 seconds per API request to accommodate extended reasoning.
Use `--timeout 900` to change this. This deadline includes waiting for the full
response, even if the provider sends keep-alive whitespace while processing.
Other Python agents still default to `timeout=120.0`; pass a larger `timeout`
to `Agent.deepseek(...)` when needed. DeepSeek describes its keep-alive behavior
in the [rate-limit documentation](https://api-docs.deepseek.com/quick_start/rate_limit/).

For models with vision, use the existing `Image` wrapper or Pillow objects:

```python
from smolgent import Image
print(agent.run(["Describe this image.", Image.from_file("image.png")]))
```

DeepSeek documents image input for `deepseek-flash` in JPEG, PNG, GIF, and WebP
formats. `File` wrappers containing inline image data are translated to DeepSeek's
flat file-part format. This does not add audio/video, PDF parsing, file uploads,
or streaming. See [DeepSeek's vision guide](https://api-docs.deepseek.com/guides/vision/)
for current model and format limits. The chat API reference also lists image
parts in tool messages; availability is subject to the selected model and service
([chat API reference](https://api-docs.deepseek.com/api/create-chat-completion/)).
Built-in `read` remains text-only; use custom media tools for images.

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

Tools infer their JSON Schema from argument type hints. Both ordinary and
`async def` functions work:

```python
from smolgent import Agent, tool

@tool
def weather(city: str):
    """Get an illustrative weather report for a city."""
    return {"city": city, "temperature_c": 20}

agent = Agent.openrouter("openrouter/auto", tools=[weather])
print(agent.run("Use weather to get the report for London."))
```

`@tool`, `@tool()`, and `@tool(name="...", description="...")` all support inference.
The function's docstring supplies the description unless overridden. Supported
argument hints are `str`, `int`, `float`, `bool`, `None`, `Any`, `list[T]`,
`dict[str, T]`, unions (`T | U` or `Union[T, U]`), `Optional[T]`, and `Literal`
of strings, integers, booleans, or `None`. Nested containers and their `typing`
equivalents work; bare `list`/`dict` allow arbitrary contents. `Annotated[T, ...]`
uses `T`, ignoring metadata. Quoted/postponed hints must resolve in the function's
module. Return hints do not affect the input schema.

Arguments without defaults are required, including nullable arguments such as
`city: str | None`. A default (`city: str | None = None`) allows omission and
Python applies it when the tool runs. Keyword-only arguments work too. Inference
does not coerce or validate values at runtime.

Every argument needs a supported hint for inference. Missing/unsupported hints,
positional-only arguments, `*args`, and `**kwargs` raise an error when decorating.
Use `@tool(parameters={...})` to provide an explicit schema instead; it takes
precedence over hints and also supports functions without annotations. For example:

```python
@tool(parameters={
    "type": "object",
    "properties": {"city": {"type": "string", "minLength": 1}},
    "required": ["city"],
})
def weather(city):
    """Get an illustrative weather report for a city."""
    return {"city": city, "temperature_c": 20}
```

The decorator returns a `Tool` object. Alternatively construct
`Tool(name, description, parameters, handler)` directly. Functions receive keyword
arguments and return JSON values or media (see below). Async tools run on the caller's
event loop; synchronous tools run in its thread pool. Both preserve context
variables. Tool exceptions become model-visible errors, matching Rust behavior.
Provider/transport failures raise `SmolgentError`, a subclass of `RuntimeError`.

## Multimodal tool output

Return a Pillow image directly from a tool; the binding encodes it as PNG:

```python
from PIL import ImageGrab, Image
from smolgent import tool

@tool
def screenshot() -> Image.Image:
    """Capture the current screen."""
    return ImageGrab.grab()
```

Add the tool to `Agent.openrouter(model, tools=[screenshot])`, using a model that
supports both tools and the media you supply. The actual return value determines
the attachment type; return annotations are optional. Sync and async tools work.

For audio, video, and documents, return a wrapper:

```python
from smolgent import Audio, Video, File, MultimodalResult, tool

@tool
def recording() -> Audio:
    """Retrieve the audio recording."""
    return Audio.from_file("recording.wav")

@tool
def evidence() -> MultimodalResult:
    """Retrieve the recording, clip, and report."""
    return MultimodalResult([
        "Recording and supporting material:",
        Audio.from_file("recording.wav"),
        Video.from_file("clip.mp4"),
        File.from_file("report.pdf"),
    ])
```

`MultimodalResult` preserves the order of text, wrappers, Pillow images, and raw
content dictionaries. Ordinary dictionary/list tool returns remain JSON text;
wrap content parts in `MultimodalResult` to send actual attachments. A tool can
return a single wrapper or Pillow image without the container.

| Wrapper | Local file | Encoded bytes | Remote URL |
| --- | --- | --- | --- |
| `Image` | `Image.from_file("photo.png")` | `Image.from_bytes(data, mime_type="image/png")` | `Image.from_url(url)` |
| `Audio` | `Audio.from_file("recording.wav")` | `Audio.from_bytes(data, format="wav")` | Not supported by the core audio format |
| `Video` | `Video.from_file("clip.mp4")` | `Video.from_bytes(data, mime_type="video/mp4")` | `Video.from_url(url)` |
| `File` | `File.from_file("report.pdf")` | `File.from_bytes(data, mime_type="application/pdf", filename="report.pdf")` | `File.from_url(url, filename="report.pdf")` |

Helpers read local files immediately and infer MIME type/audio format from the
extension. Supply `mime_type=` or `format=` explicitly when needed. Unknown
document extensions default to `application/octet-stream`. Encoded bytes must
already contain the appropriate file format, such as WAV or MP4; these helpers
do not record, synthesize, transcode, or extract video frames. URLs are forwarded
to the provider. Supported media, codecs, and attachment sizes depend on the
provider/model, including support for media in tool messages.

For explicit Pillow encoding options, use smolgent's `Image` wrapper:

```python
from smolgent import Image

attachment = Image.from_pil(pil_image, format="JPEG", max_size=(1280, 1280), quality=85)
```

`from_pil` encodes an independent snapshot, applies EXIF orientation, and preserves
aspect ratio when resizing. PNG is the default; JPEG flattens transparency onto
white. `Image` constructors also accept `detail="auto"`, `"low"`, or `"high"`.
Automatic Pillow encoding runs in a worker thread. Explicit `from_file` and
`from_pil` calls are synchronous; use `asyncio.to_thread` for expensive work inside
an async tool. Keep a directly returned Pillow image open until it is encoded,
or call `Image.from_pil` inside its file context manager.

Runnable examples: [image_tool.py](examples/image_tool.py) generates an image in
memory; [multimodal_tool.py](examples/multimodal_tool.py) returns a selected file.

## Streaming

Use `stream()` for a blocking iterator or `astream()` for an async iterator. Both
run the complete Rust agent loop, including custom tools and subsequent model
requests. Set `DEEPSEEK_API_KEY` for this example (or supply `keyring_id`):

```python
from smolgent import Agent

agent = Agent.deepseek("deepseek-flash")
for event in agent.stream("Explain why the sky is blue."):
    if event.type == "text_delta":
        print(event.text, end="", flush=True)
    elif event.type == "completed":
        response = event.response  # Same Response type as run()/arun().
```

In notebooks or async applications:

```python
from contextlib import aclosing

async with aclosing(agent.astream("Tell me more.")) as events:
    async for event in events:
        if event.type == "text_delta":
            print(event.text, end="", flush=True)
```

| Event type | Payload |
| --- | --- |
| `model_started` | A model request is starting. |
| `text_delta` | `event.text`: assistant text as it arrives. |
| `reasoning_delta` | `event.text`: reasoning text supplied by the provider, kept separate. |
| `tool_call_delta` | `event.data["index"]` and `["delta"]`: partial tool-call fields; arguments may be incomplete JSON. |
| `model_completed` | `event.response`: one assembled model response, before tool execution. |
| `tool_started` | `event.data["call"]`: the complete tool call. |
| `tool_result` | `event.data`: `tool_call_id`, `name`, and `content` (text or multimodal parts). Tool failures are returned to the model as ordinary error content. |
| `completed` | `event.response`: the final answer; session history is committed. |

Text deltas include intermediate assistant messages before tool calls. Use
`completed.response.text` when you need only the final answer. Streamed
`response.raw` is an assembled Chat Completions object, not the original SSE
chunks; it includes usage when the provider supplies it. Reasoning metadata is
preserved for subsequent tool rounds. Automatic context compaction uses ordinary
requests and does not appear as assistant text in the stream.

Streaming supports text/reasoning output and tool calls over Chat Completions SSE
with DeepSeek, OpenRouter, and compatible servers (including llama.cpp). Existing
multimodal prompts and tool results remain supported according to the model's
capabilities. Generated audio/video/image output streams are not implemented.
Keep-alive comments are ignored; they do not require a reply. Protocol references:
[DeepSeek](https://api-docs.deepseek.com/api/create-chat-completion/) and
[OpenRouter](https://openrouter.ai/docs/api/reference/streaming).

Errors raise `SmolgentError`, including disconnects before the end marker and
provider errors delivered inside the stream. Partial output may already have been
displayed. History is committed only on successful completion; already executed
tool effects remain. Consume the iterator through `completed`. If stopping early,
call `close()` on a synchronous stream (or use `contextlib.closing`), or await
`aclose()` on an async stream (or use `contextlib.aclosing` as above). Cancellation
also cancels active async Python tools; a synchronous tool already running in a
thread cannot be forcibly stopped. Event delivery uses bounded queues.

The configured `timeout` remains a total deadline per HTTP request, including
streaming and consumer delays. Keep-alives do not extend it. Streaming improves
visibility but does not remove provider queueing time. `run()` and `arun()` retain
their non-streaming behavior.

Run the [streaming tool example](examples/streaming_agent.py):

```powershell
.\.venv\Scripts\python.exe python\examples\streaming_agent.py
```

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

The same wrappers, Pillow images, and `MultimodalResult` work in `run()` and
`arun()`. Combine attachments and text in a list:

```python
from smolgent import Image, Audio, Video

response = agent.run([
    "Compare this image with the clip.",
    Image.from_file("photo.png"),
    Video.from_file("clip.mp4"),
])
response = await agent.arun(["Transcribe this.", Audio.from_file("recording.wav")])
```

Raw content dictionaries remain supported:

```python
response = agent.run([
    {"type": "text", "text": "Describe this image."},
    {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
])
```

The model must support the requested media. The built-in `read` tool still uses
the text file reader; use custom tools to return media. Model-aware media reading and event/telemetry
subscriptions are not exposed yet.
