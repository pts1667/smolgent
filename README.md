# smolgent

smolagent is a tiny LLM harness library. It implements an easily configurable tool calling agent.
It is designed to be small, secure and easy to use. It has built-in tools for reading, writing, and using ripgrep.
It preserves reasoning whenever possible.

I made this because, despite massive popular LLM harnesses like OpenCode, Codex and Claude Code existing, I could not find something that easily integrates into a user app.
Support for other popular endpoints will happen later, but I prefer to keep the list of "officially" supported endpoints small.

Note: the ripgrep tool requires access to the ripgrep (or `rg`) command. If you have git CLI installed or are on Linux, you probably have this.
If your user doesn't have access to ripgrep, you can just disable the built-in ripgrep tool- but you'll have to provide your own substitutions.

## Examples

See the [full_cli.rs](examples/full_cli.rs) example for a full TUI app exposing the agent harness.

Run the example: `cargo run --release --example full_cli -- . --read-only`
Replace `.` with any directory you wish to read from.

## Providing API Keys

Providing API keys should be done using keyring secrets.
It is the user's responsibility to provide these to the library, or preferably store them in the keyring themselves, in a secure fashion.
You can always just use a .env file or even environment variables if you don't care about any of that.

## Supported Endpoints

- llama.cpp
- Openrouter

## Context Compaction

Context compaction is enabled by default.
Compaction can be done with a capable LLM during the running session, but is also requested at certain tokens counts.
This behaviour is configurable, and can be disabled.

Configure this with `SessionConfig::default().compaction`, or use `CompactionConfig::disabled()` to turn it off.

## Telemetry

Telemetry is opt-in and dependency-free.
Configure `SessionConfig::telemetry` with `TelemetryConfig::messages()`, `TelemetryConfig::tools()`, or `TelemetryConfig::all()`, then create the session with `ChatSession::with_config_and_telemetry`.
Payload capture is disabled unless you choose `TelemetryConfig::all()` or otherwise enable the payload flags.
It is the responsibility of the user to actually log the provided content; the library only emits events.

## TODO

- Multimodal input
