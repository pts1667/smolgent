use std::sync::mpsc::Receiver;

use super::compaction::{ContextUsage, ToolCallUsage};

pub type AgentEventReceiver = Receiver<AgentEvent>;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct NotificationConfig {
    pub model_requests: bool,
    pub model_responses: bool,
    pub tool_rounds: bool,
    pub tool_calls: bool,
    pub tool_results: bool,
    pub tool_errors: bool,
    pub compaction: bool,
}

impl NotificationConfig {
    pub fn none() -> Self {
        Self::default()
    }

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

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AgentEvent {
    ModelRequestStarted {
        completed_tool_rounds: usize,
    },
    ModelResponseReceived {
        completed_tool_rounds: usize,
        tool_calls: usize,
        content_len: usize,
    },
    ToolRoundStarted {
        round: usize,
        tool_calls: usize,
    },
    ToolCallStarted {
        round: usize,
        tool_call_id: String,
        name: String,
        arguments: String,
    },
    ToolCallFinished {
        round: usize,
        tool_call_id: String,
        name: String,
        content_len: usize,
    },
    ToolCallFailed {
        round: usize,
        tool_call_id: String,
        name: String,
        error: String,
    },
    MaxToolRoundsReached {
        max_tool_rounds: usize,
    },
    CompactionStarted {
        estimated_tokens: usize,
        target_estimated_tokens: usize,
    },
    CompactionBreakdown {
        total_estimated_tokens: usize,
        largest_turns: Vec<ContextUsage>,
        largest_tool_calls: Vec<ToolCallUsage>,
    },
    CompactionToolCallStarted {
        tool_call_id: String,
        name: String,
        arguments: String,
    },
    CompactionFinished {
        before_estimated_tokens: usize,
        after_estimated_tokens: usize,
        rounds: usize,
    },
    CompactionSkipped {
        reason: String,
    },
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
