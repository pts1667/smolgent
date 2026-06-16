extern crate self as smolgent;

pub mod chat;
pub mod error;
pub mod provider;
pub mod secrets;
pub mod session;
pub mod state;
pub mod tools;

pub use chat::{
    ChatMessage, ChatRequest, ChatResponse, MessageRole, ReasoningConfig, ReasoningPayload,
    ToolCall, ToolCallFunction,
};
pub use error::{Error, Result};
pub use provider::{ApiKeyRef, ChatProvider, ProviderConfig, ProviderKind};
pub use secrets::{KeyringCoreSecretStore, SecretStore, native_credential_store};
pub use session::{
    AgentEvent, AgentEventReceiver, ChatSession, NotificationConfig, SessionConfig, SessionTurn,
};
pub use state::AgentState;
pub use tools::builtin::{
    ApplyPatchArgs, ReadArgs, RgArgs, apply_patch_tool, builtin_registry, read_tool, ripgrep_tool,
};
pub use tools::{FunctionToolDefinition, Tool, ToolDefinition, ToolRegistry, ToolResult};

pub use smolgent_macros::tool;
