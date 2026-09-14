//! A small, extensible harness for OpenAI-compatible LLM agents.
//!
//! `smolgent` provides a few building blocks:
//!
//! - [`ChatProvider`] for OpenRouter, DeepSeek, llama.cpp, and other chat-completions endpoints.
//! - [`ChatSession`] for preserving conversation history, reasoning payloads, tool calls, and
//!   tool results.
//! - [`ToolRegistry`] plus the [`tool`] macro for exposing Rust functions as LLM tools.
//! - [`AgentState`] and [`builtin_registry`] for basic read/write/search file tools with root
//!   validation.
//! - [`KeyringCoreSecretStore`] for storing API keys through `keyring-core`.
//!
//! ## Quick Start
//!
//! Store your OpenRouter key once:
//!
//! ```text
//! # optional: choose an app-specific key id
//! set SMOLGENT_OPENROUTER_KEY_ID=my-app/openrouter
//! cargo run --example openrouter_keyring_setup -- set <OPENROUTER_API_KEY>
//! ```
//!
//! Then create a provider and session:
//!
//! ```no_run
//! use std::sync::Arc;
//!
//! use smolgent::{
//!     ChatProvider, ChatSession, KeyringCoreSecretStore, ProviderConfig,
//! };
//!
//! #[tokio::main]
//! async fn main() -> smolgent::Result<()> {
//!     let keyring_id = "my-app/openrouter";
//!     let secrets = Arc::new(KeyringCoreSecretStore::smolgent()?);
//!     let provider = ChatProvider::new(
//!         ProviderConfig::openrouter_with_keyring(
//!             "deepseek/deepseek-v4-flash",
//!             keyring_id,
//!         )?,
//!     )
//!     .with_secrets(secrets);
//!
//!     let mut session = ChatSession::with_system_prompt("You are concise and helpful.");
//!     let response = session
//!         .send_user_message(&provider, "Say hello in one sentence.")
//!         .await?;
//!
//!     println!("{}", response.message.content);
//!     Ok(())
//! }
//! ```
//!
//! To run an agent with built-in file tools:
//!
//! ```no_run
//! # use std::path::PathBuf;
//! # use smolgent::{AgentState, builtin_registry};
//! let root = PathBuf::from("my-project");
//! let state = AgentState::new([root.clone()], [root]);
//! let tools = builtin_registry(state);
//! ```
//!
//! `examples/full_cli.rs` shows a fuller interactive CLI agent. `examples/telemetry.rs`
//! demonstrates telemetry without requiring a real API key, and `examples/openrouter_chat.rs`
//! demonstrates a minimal OpenRouter request.
//! `examples/llama_cpp_multimodal.rs` discovers a local server's media capabilities and
//! sends images, audio, or video using the same content API.
//!
//! ## Events And Telemetry
//!
//! Session event and telemetry streams use bounded `std::sync::mpsc` channels. If enabled, the
//! receiver must be drained by your application. Sending blocks when the channel is full, which is
//! intentional: subscribed [`AgentEvent`] and [`TelemetryEvent`] streams are treated as user-owned
//! observability pipelines rather than best-effort logs.
//!
//! Use [`NotificationConfig`] for high-level agent progress such as tool calls and compaction, and
//! [`TelemetryConfig`] for structured observability such as messages, tool payloads, and model
//! request/response payloads.
//!
//! ```no_run
//! use std::thread;
//!
//! use smolgent::{ChatSession, NotificationConfig, SessionConfig, TelemetryConfig};
//!
//! let (session, events, telemetry) = ChatSession::with_config_and_telemetry(SessionConfig {
//!     notifications: NotificationConfig::tools(),
//!     telemetry: TelemetryConfig::messages(),
//!     ..SessionConfig::default()
//! });
//!
//! if let Some(events) = events {
//!     thread::spawn(move || {
//!         while let Ok(event) = events.recv() {
//!             eprintln!("agent event: {event:?}");
//!         }
//!     });
//! }
//!
//! if let Some(telemetry) = telemetry {
//!     thread::spawn(move || {
//!         while let Ok(event) = telemetry.recv() {
//!             eprintln!("telemetry: {event:?}");
//!         }
//!     });
//! }
//! # drop(session);
//! ```
//!
//! ## Defining Tools
//!
//! For simple functions, [`tool`] derives a JSON Schema from the named parameters. For handlers
//! with application state or an existing argument type, mark those parameters explicitly:
//!
//! ```
//! use schemars::JsonSchema;
//! use serde::Deserialize;
//!
//! #[derive(Clone)]
//! struct Counter(i64);
//!
//! #[derive(Deserialize, JsonSchema)]
//! struct AddArgs {
//!     /// Amount to add to the counter.
//!     amount: i64,
//! }
//!
//! #[smolgent::tool(fallible)]
//! fn add(
//!     #[tool(context)] counter: &Counter,
//!     #[tool(arguments)] args: AddArgs,
//! ) -> smolgent::Result<i64> {
//!     Ok(counter.0 + args.amount)
//! }
//!
//! let tool = add_tool(Counter(10));
//! assert_eq!(tool.definition().function.name, "add");
//! ```
//!
//! The argument type's Rustdoc and `schemars`/`serde` attributes are retained in the generated
//! schema. Each expansion also generates `<handler>_tool_definition()` for callers that only need
//! provider metadata. [`tool_definition`] provides the definition-only form used by tools whose
//! execution is managed separately.
//!
//! ## Examples
//!
//! - `cargo run --example openrouter_keyring_setup -- check`
//! - `cargo run --example openrouter_chat`
//! - `cargo run --example openrouter_multimodal -- <image-or-video-path> <mime-type>`
//! - `cargo run --example llama_cpp_chat`
//! - `cargo run --example tool_registry`
//! - `cargo run --example telemetry`
//! - `cargo run --example full_cli -- <input-directory>`
//!
extern crate self as smolgent;

/// Chat-completions request, response, message, reasoning, and tool-call types.
pub mod chat;
/// Multimodal content parts and helpers for encoding local media bytes.
pub mod content;
/// Crate-wide error and result types.
pub mod error;
/// OpenAI-compatible provider configuration and HTTP transport.
pub mod provider;
/// API-key storage abstractions and `keyring-core` integration.
pub mod secrets;
/// Conversation/session management, tool loops, compaction, events, and telemetry.
pub mod session;
/// Read/write root state used by built-in file tools.
pub mod state;
/// Tool definitions, registries, built-ins, and compaction tool schemas.
pub mod tools;

pub use chat::{
    ChatMessage, ChatRequest, ChatResponse, MessageRole, ReasoningConfig, ReasoningPayload,
    ToolCall, ToolCallFunction,
};
pub use content::{
    ContentPart, FileInput, ImageDetail, ImageUrl, InputAudio, MessageContent, VideoUrl,
};
pub use error::{Error, Result};
pub use provider::{ApiKeyRef, ChatProvider, ModelCapabilities, ProviderConfig, ProviderKind};
pub use secrets::{KeyringCoreSecretStore, SecretStore, native_credential_store};
pub use session::{
    AgentEvent, AgentEventReceiver, ChatSession, CompactionConfig, ContextUsage,
    ContextUsageBreakdown, NotificationConfig, SessionConfig, SessionTurn, TelemetryConfig,
    TelemetryEvent, TelemetryEventReceiver, ToolCallUsage,
};
pub use state::AgentState;
pub use tools::builtin::{
    ApplyPatchArgs, CreateFileArgs, DeleteFileArgs, ReadArgs, RgArgs, apply_patch_tool,
    builtin_registry, create_file_tool, delete_file_tool, read_tool, ripgrep_tool,
};
pub use tools::media::{
    READ_MAX_MEDIA_BYTES, builtin_registry_for_provider, read_tool_with_capabilities,
};
pub use tools::{FunctionToolDefinition, Tool, ToolDefinition, ToolRegistry, ToolResult};

pub use smolgent_macros::{tool, tool_definition};
