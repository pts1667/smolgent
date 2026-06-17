use std::sync::mpsc::Receiver;

use crate::chat::{ChatRequest, ChatResponse};

pub type TelemetryEventReceiver = Receiver<TelemetryEvent>;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TelemetryConfig {
    pub messages: bool,
    pub tool_calls: bool,
    pub tool_payloads: bool,
    pub model_metadata: bool,
    pub model_payloads: bool,
}

impl TelemetryConfig {
    pub fn none() -> Self {
        Self::default()
    }

    pub fn messages() -> Self {
        Self {
            messages: true,
            ..Self::default()
        }
    }

    pub fn tools() -> Self {
        Self {
            messages: true,
            tool_calls: true,
            ..Self::default()
        }
    }

    pub fn all() -> Self {
        Self {
            messages: true,
            tool_calls: true,
            tool_payloads: true,
            model_metadata: true,
            model_payloads: true,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.messages
            || self.tool_calls
            || self.tool_payloads
            || self.model_metadata
            || self.model_payloads
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum TelemetryEvent {
    UserMessage {
        turn_id: u64,
        content_len: usize,
        content: Option<String>,
    },
    AssistantMessage {
        turn_id: u64,
        content_len: usize,
        tool_calls: usize,
        content: Option<String>,
    },
    ToolMessage {
        turn_id: u64,
        tool_call_id: String,
        name: String,
        content_len: usize,
        content: Option<String>,
    },
    ModelRequest {
        provider: String,
        model: String,
        message_count: usize,
        tool_count: usize,
        request: Option<ChatRequest>,
    },
    ModelResponse {
        provider: String,
        model: String,
        content_len: usize,
        tool_calls: usize,
        response: Option<ChatResponse>,
    },
    ToolCallStarted {
        tool_call_id: String,
        name: String,
        arguments_len: usize,
        arguments: Option<String>,
    },
    ToolCallFinished {
        tool_call_id: String,
        name: String,
        content_len: usize,
        content: Option<String>,
    },
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
