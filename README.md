# smolgent

smolagent is a tiny LLM harness library. It implements an easily configurable tool calling agent.
It is designed to be small, secure and easy to use.
It preserves reasoning whenever possible.

I made this because, despite massive popular LLM harnesses like OpenCode, Codex and Claude Code existing, I could not find something that easily integrates into a user app.
Support for other popular endpoints will happen later, but I prefer to keep the list of "officially" supported endpoints small.

## Examples

See the [full_cli.rs](examples/full_cli.rs) example for a full TUI app exposing the agent harness.

## Providing API Keys

Providing API keys should be done using keyring secrets.
It is the user's responsibility to provide these to the library, or preferably store them in the keyring themselves, in a secure fashion.
You can always just use a .env file or even environment variables if you don't care about any of that.

## Supported Endpoints

- llama.cpp
- Openrouter

## TODO

- Context compaction
- Logging
- Multimodal input