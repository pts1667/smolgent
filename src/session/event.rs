use std::sync::mpsc::Receiver;

use super::compaction::{ContextUsage, ToolCallUsage};

/// Receiver for high-level agent progress events.
///
/// This is a bounded `std::sync::mpsc` channel. If you enable notifications, drain this receiver
/// while the agent runs; session sends block when the channel is full.
pub type AgentEventReceiver = Receiver<AgentEvent>;

/// Selects which [`AgentEvent`] notifications a session emits.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct NotificationConfig {
    /// Emit before a model request is sent.
    pub model_requests: bool,
    /// Emit after a model response is received.
    pub model_responses: bool,
    /// Emit when a managed tool round starts or the round limit is reached.
    pub tool_rounds: bool,
    /// Emit when a normal tool call starts.
    pub tool_calls: bool,
    /// Emit when a normal tool call succeeds.
    pub tool_results: bool,
    /// Emit when a normal tool call fails.
    pub tool_errors: bool,
    /// Emit context compaction progress.
    pub compaction: bool,
}

impl NotificationConfig {
    /// Disable all agent events.
    pub fn none() -> Self {
        Self::default()
    }

    /// Emit normal tool-loop events, without model or compaction progress.
    pub fn tools() -> Self {
        Self {
            tool_rounds: true,
            tool_calls: true,
            tool_results: true,
            tool_errors: true,
            compaction: false,
            ..Self::default()
        }
    }

    /// Emit every agent event.
    pub fn all() -> Self {
        Self {
            model_requests: true,
            model_responses: true,
            tool_rounds: true,
            tool_calls: true,
            tool_results: true,
            tool_errors: true,
            compaction: true,
        }
    }

    /// Whether this configuration enables any event stream.
    pub fn is_enabled(&self) -> bool {
        self.model_requests
            || self.model_responses
            || self.tool_rounds
            || self.tool_calls
            || self.tool_results
            || self.tool_errors
            || self.compaction
    }
}

/// High-level progress events from a managed agent session.
///
/// These are intended for UI/status updates such as "calling tool", "reading file", or
/// "compacting context". For lower-level logging, payload capture, or audit trails, use
/// [`crate::TelemetryEvent`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AgentEvent {
    /// A model request is about to be sent.
    ModelRequestStarted { completed_tool_rounds: usize },
    /// A model response was received.
    ModelResponseReceived {
        completed_tool_rounds: usize,
        tool_calls: usize,
        content_len: usize,
    },
    /// A managed tool round is starting.
    ToolRoundStarted { round: usize, tool_calls: usize },
    /// A normal registry tool call is starting.
    ToolCallStarted {
        round: usize,
        tool_call_id: String,
        name: String,
        arguments: String,
    },
    /// A normal registry tool call succeeded.
    ToolCallFinished {
        round: usize,
        tool_call_id: String,
        name: String,
        content_len: usize,
    },
    /// A normal registry tool call failed and was reported back to the model.
    ToolCallFailed {
        round: usize,
        tool_call_id: String,
        name: String,
        error: String,
    },
    /// The managed loop stopped because the round limit was reached.
    MaxToolRoundsReached { max_tool_rounds: usize },
    /// Explicit context compaction has started.
    CompactionStarted {
        estimated_tokens: usize,
        target_estimated_tokens: usize,
    },
    /// Token-consumer breakdown emitted at the start of explicit compaction.
    CompactionBreakdown {
        total_estimated_tokens: usize,
        largest_turns: Vec<ContextUsage>,
        largest_tool_calls: Vec<ToolCallUsage>,
    },
    /// A built-in compaction tool call is starting.
    CompactionToolCallStarted {
        tool_call_id: String,
        name: String,
        arguments: String,
    },
    /// Context compaction finished.
    CompactionFinished {
        before_estimated_tokens: usize,
        after_estimated_tokens: usize,
        rounds: usize,
    },
    /// Context compaction did not run or ignored an invalid compaction request.
    CompactionSkipped { reason: String },
}

impl NotificationConfig {
    pub(crate) fn should_emit(&self, event: &AgentEvent) -> bool {
        match event {
            AgentEvent::ModelRequestStarted { .. } => self.model_requests,
            AgentEvent::ModelResponseReceived { .. } => self.model_responses,
            AgentEvent::ToolRoundStarted { .. } => self.tool_rounds,
            AgentEvent::ToolCallStarted { .. } => self.tool_calls,
            AgentEvent::ToolCallFinished { .. } => self.tool_results,
            AgentEvent::ToolCallFailed { .. } => self.tool_errors,
            AgentEvent::MaxToolRoundsReached { .. } => self.tool_rounds,
            AgentEvent::CompactionStarted { .. }
            | AgentEvent::CompactionBreakdown { .. }
            | AgentEvent::CompactionToolCallStarted { .. }
            | AgentEvent::CompactionFinished { .. }
            | AgentEvent::CompactionSkipped { .. } => self.compaction,
        }
    }
}
