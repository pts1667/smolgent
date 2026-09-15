# smolgent

smolgent is a tiny LLM harness library. It implements an easily configurable tool calling agent.
It is designed to be small, secure and easy to use. It has built-in tools for reading, writing, and using ripgrep.
It preserves reasoning whenever possible.

I made this because, despite massive popular LLM harnesses like OpenCode, Codex and Claude Code existing, I could not find something that easily integrates into a user app.
Support for other popular endpoints will happen later, but I prefer to keep the list of "officially" supported endpoints small.

Note: the ripgrep tool requires access to the ripgrep (or `rg`) command. If you have git CLI installed or are on Linux, you probably have this.
If your user doesn't have access to ripgrep, you can just disable the built-in ripgrep tool- but you'll have to provide your own substitutions.

## Examples

### Python

Install from this checkout with `python -m pip install .` (requires Rust to build),
then set `OPENROUTER_API_KEY`:

```python
from smolgent import Agent
agent = Agent.openrouter("openrouter/auto", read_roots=["."])
print(agent.run("Read README.md and summarize this project in three sentences."))
```

The same agent retains conversation history. Use `await agent.arun(...)` in async
applications. See [the Python guide](python/README.md) for custom Python tools,
keyring credentials, local providers, packaging, and development instructions.

### Rust

See the [full_cli.rs](examples/full_cli.rs) example for a full TUI app exposing the agent harness.

Run the example: `cargo run --release --example full_cli -- . --read-only`
Replace `.` with any directory you wish to read from.

### Streaming (Rust and Python)

Python exposes synchronous `agent.stream(prompt)` and asynchronous
`agent.astream(prompt)` event iterators:

```python
from smolgent import Agent

agent = Agent.deepseek("deepseek-flash")  # Set DEEPSEEK_API_KEY.
for event in agent.stream("Explain why the sky is blue."):
    if event.type == "text_delta":
        print(event.text, end="", flush=True)
```

Rust exposes `ChatProvider::send_request_stream` for a single completion and
`ChatSession::stream_user_message_with_tools` for the full agent loop. Both return
an async stream of `Result<StreamEvent>`; tools execute automatically in the session
stream. The final `Completed` event carries the assembled response.

Runnable examples: [Python](python/examples/streaming_agent.py) and
[Rust](examples/streaming_chat.rs) (`cargo run --example streaming_chat`). See the
[Python streaming guide](python/README.md#streaming) for events and cancellation.

## Providing API Keys

Providing API keys should be done using keyring secrets.
It is the app's responsibility to choose the keyring id it wants to use. Do not assume a single global OpenRouter key is appropriate for every app using this library.
Use `ProviderConfig::openrouter_with_keyring(model, "my-app/openrouter")` and store the key under the same id.

The examples use `SMOLGENT_OPENROUTER_KEY_ID`, defaulting to `smolgent/examples/openrouter`:

```powershell
$env:SMOLGENT_OPENROUTER_KEY_ID = "my-app/openrouter"
cargo run --example openrouter_keyring_setup -- set-from-env
cargo run --example openrouter_chat
```

You can always just use a .env file or even environment variables if you don't care about any of that.

## Supported Endpoints

- llama.cpp
- Openrouter
- DeepSeek

### DeepSeek

Python: set `DEEPSEEK_API_KEY`, then call `Agent.deepseek("deepseek-flash")`.
It supports the same tools, history, and async methods as the other presets.
See the [Python guide](python/README.md#deepseek) and
[tool example](python/examples/deepseek_agent.py).

Rust: use `ProviderConfig::deepseek(model)` or
`ProviderConfig::deepseek_with_keyring(model, "my-app/deepseek")`, and attach a
secret store containing the matching key. For environment credentials, assign
`config.api_key = ApiKeyRef::Literal(key)`; the Rust preset itself does not read
environment variables. The [minimal example](examples/deepseek_chat.rs) does:

```powershell
$env:DEEPSEEK_API_KEY = "..."
cargo run --example deepseek_chat
# Optional: $env:DEEPSEEK_MODEL = "deepseek-v4-pro"
$env:SMOLGENT_PROVIDER = "deepseek"
cargo run --release --example full_cli -- . --read-only
```

The preset uses `https://api.deepseek.com/chat/completions`. Omitted reasoning
configuration uses the API defaults. To disable thinking in Rust:

```rust
use smolgent::{ProviderConfig, ReasoningConfig};
let mut config = ProviderConfig::deepseek("deepseek-flash")?;
config.reasoning = Some(ReasoningConfig {
    enabled: Some(false),
    ..Default::default()
});
```

## Context Compaction

Context compaction is enabled by default.
Compaction can be done with a capable LLM during the running session, but is also requested at certain tokens counts.
This behaviour is configurable, and can be disabled.

Configure this with `SessionConfig::default().compaction`, or use `CompactionConfig::disabled()` to turn it off.

Each compaction round uses the updated conversation and usage breakdown. If an automatic attempt does not reach its target, further automatic attempts are deferred until the next managed agent run. The model can still use compaction tools during normal tool rounds when `always_offer_tools` is enabled.

Token estimates cover text and metadata only. Media and parsed PDF annotations are excluded because their token costs depend on the model, resolution, duration, and document contents. Track provider usage for media-heavy sessions; automatic compaction cannot reliably predict those costs.

## Telemetry

Telemetry is opt-in and dependency-free.
Configure `SessionConfig::telemetry` with `TelemetryConfig::messages()`, `TelemetryConfig::tools()`, or `TelemetryConfig::all()`, then create the session with `ChatSession::with_config_and_telemetry`.
Payload capture is disabled unless you choose `TelemetryConfig::all()` or otherwise enable the payload flags.
It is the responsibility of the user to actually log the provided content; the library only emits events.

## Multimodal Input

Pass an ordered `Vec<ContentPart>` wherever you would pass a user message. It works with direct provider requests, `ChatSession::send_user_message`, and `ChatSession::run_user_message_with_tools`.

```rust
use smolgent::{ChatMessage, ContentPart};

let message = ChatMessage::user(vec![
    ContentPart::text("Describe this image."),
    ContentPart::image_url("https://example.com/image.png"),
]);
// provider.send_messages(&[message]).await?;
```

Following [OpenRouter's multimodal documentation](https://openrouter.ai/docs/guides/overview/multimodal/overview), supported input parts are:

| Input | URL or pre-encoded data | Encode local bytes |
| --- | --- | --- |
| Image | `ContentPart::image_url(url_or_data_url)` | `ContentPart::image_bytes("image/png", &bytes)` |
| PDF | `ContentPart::file("document.pdf", url_or_data_url)` | `ContentPart::pdf_bytes("document.pdf", &bytes)` |
| Audio | `ContentPart::input_audio(raw_base64, "wav")` | `ContentPart::audio_bytes("wav", &bytes)` |
| Video | `ContentPart::video_url(url_or_data_url)` | `ContentPart::video_bytes("video/mp4", &bytes)` |

Byte helpers encode data in memory; the application reads files and chooses MIME types/formats. Audio requires raw base64 rather than a URL. Image, PDF, and video byte helpers produce base64 data URLs. Multiple attachments and text parts retain their order. Image detail can be set through `ContentPart::ImageUrl` and `ImageDetail`.

Choose a model that supports the requested modalities. Video URL and media format support depend on the upstream provider/model. OpenRouter handles PDF parsing with its default configuration. Returned PDF annotations and reasoning are preserved in session history for follow-up requests. OpenRouter and llama.cpp input formats are covered by HTTP mock tests; support at other compatible endpoints depends on their capabilities. Media generation and dedicated speech endpoints are outside this input API.

The image/video example uses `z-ai/glm-5.3-flash` on OpenRouter and the same keyring entry as `openrouter_chat`. Pass one or more local paths, each followed by its MIME type:

```powershell
cargo run --example openrouter_multimodal -- ./image.png image/png
cargo run --example openrouter_multimodal -- ./clip.mp4 video/mp4
cargo run --example openrouter_multimodal -- --prompt "Does this image appear in the video?" ./image.png image/png ./clip.mp4 video/mp4
```

It encodes media as base64 data URLs and sends the prompt and attachments in one message. Use `--help` for usage. This example selects the model explicitly; `OPENROUTER_MODEL` does not override it.

### llama.cpp

Use the same `ContentPart` constructors with `ProviderConfig::llama_cpp(base_url, model)`. Images and audio use their existing wire formats; the provider converts video parts to llama.cpp's `input_video` format when sending a request, including media in tool results. Session history retains the original parts. llama.cpp does not provide OpenRouter's PDF parsing.

Start a [multimodal llama-server](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md) with a supported model and its matching projector, for example `llama-server -m model.gguf --mmproj mmproj.gguf --alias local-model --port 8080`. According to the [server API documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), image decoding supports PNG/JPEG/GIF/BMP/TGA, and audio decoding supports WAV/MP3/FLAC. WebP is not supported by the standard image decoder; convert it to PNG or JPEG first. Video requires a recent server with video support and FFmpeg/ffprobe available on the server.

The example discovers capabilities and uses the bounded media reader to attach one or more files. File extensions select formats:

```powershell
$env:LLAMA_CPP_BASE_URL = "http://127.0.0.1:8080"
$env:LLAMA_CPP_MODEL = "local-model"
cargo run --example llama_cpp_multimodal -- ./image.png
cargo run --example llama_cpp_multimodal -- --prompt "Summarize this clip" ./clip.mp4
cargo run --example llama_cpp_multimodal -- --prompt "Transcribe this" ./sound.wav
```

Set `LLAMA_CPP_API_KEY` if the server requires authentication. Local bytes are embedded in the request, so the server does not need access to the client's filesystem. The server root may include a reverse-proxy/API path prefix; do not append `/v1`.

`ChatMessage::content` and `SessionTurn::content` now use `MessageContent` (`Text` or `Parts`) instead of `String`. Existing string constructors and text-only JSON stay the same. Use `.content.text()` to read text, `.content.to_string()` for an owned display string, or match `MessageContent::Parts` to inspect attachments. Struct literals need `content: text.into()` and `annotations: Vec::new()`. Display and message telemetry include text only; full media payloads are included in model telemetry only when `model_payloads` is enabled.

## Model-aware Read Tool

Use `builtin_registry_for_provider(state, &provider).await?` instead of `builtin_registry(state)` to enable model-aware reading. For OpenRouter it uses the [model metadata endpoint](https://openrouter.ai/docs/api/api-reference/models/get-a-model-by-its-slug) and `architecture.input_modalities`. For llama.cpp it queries `/props?model=<configured-model>` and maps `modalities.vision`, `modalities.audio`, and `modalities.video` to supported inputs. Video is enabled only when explicitly advertised; vision alone does not imply video support. The model query also supports llama.cpp's model router (which may load the selected model). The registry captures capabilities once; rebuild it when switching models or reloading the server.

Supported inputs enable matching local file types: images, video, audio, and PDFs when `file` input is advertised. Media reads infer the format from the extension, enforce allowed read roots, and attach the complete file as content parts. Each media file is capped at 20 MiB before encoding; text offsets/counts are rejected for media. Text reading keeps its existing pagination and limits. Upstream model limits can be lower.

Unknown models, automatic routing (`openrouter/auto`), missing metadata (including older llama.cpp servers), and generic compatible providers retain the text-only reader. Discovery errors are returned to the application. `full_cli` reports these errors and falls back to text-only reading, so a failed lookup does not block normal text use.

For example, with `OPENROUTER_API_KEY` configured in the environment or `.env`:

```powershell
$env:OPENROUTER_MODEL = "z-ai/glm-5.3-flash"
cargo run --release --example full_cli -- ./media --read-only
```

Then ask the agent to read and compare files such as `image.png` and `clip.mp4` in that directory. The CLI discovers capabilities at startup. You can also query `provider.model_capabilities().await?` directly or construct `read_tool_with_capabilities(state, capabilities)` with a previously fetched snapshot.

To use the same CLI with llama.cpp, select the provider explicitly (OpenRouter remains the default):

```powershell
$env:SMOLGENT_PROVIDER = "llama-cpp"
$env:LLAMA_CPP_BASE_URL = "http://127.0.0.1:8080"
$env:LLAMA_CPP_MODEL = "local-model"
cargo run --release --example full_cli -- ./media --read-only
```

This mode requires no OpenRouter key. The provider-aware registry also restricts the reader to llama.cpp's decoder formats. The lower-level `read_tool_with_capabilities` helper uses OpenRouter's format set because its capability snapshot contains modalities only.

`ToolResult::content` now also uses `MessageContent`; use `.text()` or `.to_string()` when consuming its text. Custom media tools use `Tool::new_multimodal` and return `MessageContent`. Existing `Tool::new` handlers and macro-generated tools keep their JSON-to-text behavior, so ordinary JSON arrays are never mistaken for media. Tool telemetry exposes text only; full attachments are available through model payload telemetry.
