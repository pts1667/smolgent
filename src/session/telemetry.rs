use std::sync::mpsc::Receiver;

use crate::chat::{ChatRequest, ChatResponse};

/// Receiver for structured telemetry events.
///
/// This is a bounded `std::sync::mpsc` channel. If you enable telemetry, drain this receiver while
/// the agent runs; session sends block when the channel is full.
pub type TelemetryEventReceiver = Receiver<TelemetryEvent>;

/// Selects which [`TelemetryEvent`] values a session emits.
///
/// Payload fields are opt-in. [`TelemetryConfig::tools`] records tool-call metadata without
/// argument/result payloads, while [`TelemetryConfig::all`] captures full tool and model payloads.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TelemetryConfig {
    /// Emit user and assistant message events, including message text.
    pub messages: bool,
    /// Emit tool-call metadata.
    pub tool_calls: bool,
    /// Include tool arguments and tool result content.
    pub tool_payloads: bool,
    /// Emit provider/model request and response metadata.
    pub model_metadata: bool,
    /// Include full model request and response payloads.
    pub model_payloads: bool,
}

impl TelemetryConfig {
    /// Disable all telemetry.
    pub fn none() -> Self {
        Self::default()
    }

    /// Capture user and assistant message events.
    pub fn messages() -> Self {
        Self {
            messages: true,
            ..Self::default()
        }
    }

    /// Capture message and tool-call metadata, without tool payloads.
    pub fn tools() -> Self {
        Self {
            messages: true,
            tool_calls: true,
            ..Self::default()
        }
    }

    /// Capture all telemetry, including model and tool payloads.
    pub fn all() -> Self {
        Self {
            messages: true,
            tool_calls: true,
            tool_payloads: true,
            model_metadata: true,
            model_payloads: true,
        }
    }

    /// Whether this configuration enables a telemetry stream.
    pub fn is_enabled(&self) -> bool {
        self.messages
            || self.tool_calls
            || self.tool_payloads
            || self.model_metadata
            || self.model_payloads
    }
}

/// Structured observability events emitted by a [`crate::ChatSession`].
///
/// The application owns logging/export. `smolgent` only sends events through the telemetry
/// receiver returned at session creation.
#[derive(Clone, Debug, PartialEq)]
pub enum TelemetryEvent {
    /// A user message was added to the session.
    UserMessage {
        turn_id: u64,
        content_len: usize,
        content: Option<String>,
    },
    /// An assistant message was added to the session.
    AssistantMessage {
        turn_id: u64,
        content_len: usize,
        tool_calls: usize,
        content: Option<String>,
    },
    /// A tool result message was added to the session.
    ToolMessage {
        turn_id: u64,
        tool_call_id: String,
        name: String,
        content_len: usize,
        content: Option<String>,
    },
    /// A provider request is being sent.
    ModelRequest {
        provider: String,
        model: String,
        message_count: usize,
        tool_count: usize,
        request: Option<ChatRequest>,
    },
    /// A provider response was received and parsed.
    ModelResponse {
        provider: String,
        model: String,
        content_len: usize,
        tool_calls: usize,
        response: Option<ChatResponse>,
    },
    /// A tool call is starting.
    ToolCallStarted {
        tool_call_id: String,
        name: String,
        arguments_len: usize,
        arguments: Option<String>,
    },
    /// A tool call finished successfully.
    ToolCallFinished {
        tool_call_id: String,
        name: String,
        content_len: usize,
        content: Option<String>,
    },
    /// A tool call failed.
    ToolCallFailed {
        tool_call_id: String,
        name: String,
        error: String,
    },
}

impl TelemetryConfig {
    pub(crate) fn should_emit(&self, event: &TelemetryEvent) -> bool {
        match event {
            TelemetryEvent::UserMessage { .. } | TelemetryEvent::AssistantMessage { .. } => {
                self.messages
            }
            TelemetryEvent::ToolMessage { .. } => self.tool_calls || self.tool_payloads,
            TelemetryEvent::ModelRequest { .. } | TelemetryEvent::ModelResponse { .. } => {
                self.model_metadata || self.model_payloads
            }
            TelemetryEvent::ToolCallStarted { .. }
            | TelemetryEvent::ToolCallFinished { .. }
            | TelemetryEvent::ToolCallFailed { .. } => self.tool_calls || self.tool_payloads,
        }
    }
}
