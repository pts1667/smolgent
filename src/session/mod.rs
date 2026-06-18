mod compaction;
mod event;
mod telemetry;

pub use compaction::{CompactionConfig, ContextUsage, ContextUsageBreakdown, ToolCallUsage};
pub use event::{AgentEvent, AgentEventReceiver, NotificationConfig};
pub use telemetry::{TelemetryConfig, TelemetryEvent, TelemetryEventReceiver};

use std::fmt;
use std::sync::mpsc::{SyncSender, sync_channel};

use crate::chat::{
    ChatMessage, ChatRequest, ChatResponse, MessageRole, ReasoningPayload, ToolCall,
};
use crate::provider::ChatProvider;
use crate::tools::compact::{
    COMPACT_REMOVE_MESSAGES, COMPACT_SUMMARIZE_MESSAGES,
    acknowledgement as compaction_acknowledgement, definitions as compaction_tool_definitions,
    is_compaction_tool, parse_remove_args, parse_summarize_args,
};
use crate::tools::{ToolDefinition, ToolRegistry, ToolResult};
use crate::{Error, Result};

/// One persisted turn in a [`ChatSession`].
///
/// This is the session-side representation of a chat message. It keeps provider reasoning payloads
/// and tool-call metadata intact so the conversation can be replayed to providers that support
/// preserved reasoning or tool-call continuations.
#[derive(Clone, Debug, PartialEq)]
pub struct SessionTurn {
    pub id: u64,
    pub role: MessageRole,
    pub content: String,
    pub reasoning: Option<ReasoningPayload>,
    pub tool_calls: Vec<ToolCall>,
    pub tool_call_id: Option<String>,
    pub name: Option<String>,
}

impl From<ChatMessage> for SessionTurn {
    fn from(message: ChatMessage) -> Self {
        let reasoning = ReasoningPayload {
            reasoning: message.reasoning,
            reasoning_content: message.reasoning_content,
            reasoning_details: message.reasoning_details,
        }
        .into_option();

        Self {
            id: 0,
            role: message.role,
            content: message.content,
            reasoning,
            tool_calls: message.tool_calls,
            tool_call_id: message.tool_call_id,
            name: message.name,
        }
    }
}

impl From<SessionTurn> for ChatMessage {
    fn from(turn: SessionTurn) -> Self {
        let mut message = ChatMessage::new(turn.role, turn.content);
        message.tool_calls = turn.tool_calls;
        message.tool_call_id = turn.tool_call_id;
        message.name = turn.name;
        match turn.reasoning {
            Some(reasoning) => message.with_reasoning(reasoning),
            None => message,
        }
    }
}

/// Runtime configuration for [`ChatSession`].
///
/// Event and telemetry channels are bounded. If `notifications` or `telemetry` are enabled, drain
/// the returned receivers from [`ChatSession::with_config`],
/// [`ChatSession::with_config_and_telemetry`], or
/// [`ChatSession::with_system_prompt_config_and_telemetry`] while the agent is running.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SessionConfig {
    /// Maximum normal tool-call rounds before the managed loop stops.
    pub max_tool_rounds: usize,
    /// Optional capacity for the [`AgentEventReceiver`]. Defaults to `4 * max_tool_rounds`.
    pub event_channel_capacity: Option<usize>,
    /// Which high-level agent progress events to emit.
    pub notifications: NotificationConfig,
    /// Context compaction behavior.
    pub compaction: CompactionConfig,
    /// Which structured telemetry events to emit.
    pub telemetry: TelemetryConfig,
    /// Optional capacity for the [`TelemetryEventReceiver`]. Defaults to `8 * max_tool_rounds`.
    pub telemetry_channel_capacity: Option<usize>,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            max_tool_rounds: 8,
            event_channel_capacity: None,
            notifications: NotificationConfig::default(),
            compaction: CompactionConfig::default(),
            telemetry: TelemetryConfig::default(),
            telemetry_channel_capacity: None,
        }
    }
}

impl SessionConfig {
    /// Effective bounded channel capacity for [`AgentEvent`] notifications.
    pub fn event_channel_capacity(&self) -> usize {
        self.event_channel_capacity
            .unwrap_or_else(|| self.max_tool_rounds.saturating_mul(4).max(1))
    }

    /// Effective bounded channel capacity for [`TelemetryEvent`] notifications.
    pub fn telemetry_channel_capacity(&self) -> usize {
        self.telemetry_channel_capacity
            .unwrap_or_else(|| self.max_tool_rounds.saturating_mul(8).max(8))
    }
}

/// Stateful chat transcript and managed agent loop.
///
/// A session owns the message history, preserves reasoning/tool-call state, optionally emits
/// [`AgentEvent`] and [`TelemetryEvent`] values, and can run model/tool loops with a [`ToolRegistry`].
#[derive(Clone)]
pub struct ChatSession {
    turns: Vec<SessionTurn>,
    next_turn_id: u64,
    config: SessionConfig,
    events: Option<SyncSender<AgentEvent>>,
    telemetry: Option<SyncSender<TelemetryEvent>>,
}

impl Default for ChatSession {
    fn default() -> Self {
        Self {
            turns: Vec::new(),
            next_turn_id: 1,
            config: SessionConfig::default(),
            events: None,
            telemetry: None,
        }
    }
}

impl fmt::Debug for ChatSession {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ChatSession")
            .field("turns", &self.turns)
            .field("next_turn_id", &self.next_turn_id)
            .field("config", &self.config)
            .field("events_enabled", &self.events.is_some())
            .field("telemetry_enabled", &self.telemetry.is_some())
            .finish()
    }
}

impl PartialEq for ChatSession {
    fn eq(&self, other: &Self) -> bool {
        self.turns == other.turns
            && self.next_turn_id == other.next_turn_id
            && self.config == other.config
    }
}

impl ChatSession {
    /// Create an empty session with default configuration and no event streams.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a session seeded with a system prompt.
    pub fn with_system_prompt(prompt: impl Into<String>) -> Self {
        let mut session = Self::new();
        session.push_system(prompt);
        session
    }

    /// Create a configured session and optional [`AgentEventReceiver`].
    ///
    /// The receiver is `Some` only when [`SessionConfig::notifications`] enables at least one
    /// event kind. Drain it while the agent runs; sends block when the bounded channel is full.
    pub fn with_config(config: SessionConfig) -> (Self, Option<AgentEventReceiver>) {
        let (session, events, _telemetry) = Self::with_config_and_telemetry(config);
        (session, events)
    }

    /// Create a configured session plus optional event and telemetry receivers.
    ///
    /// The telemetry receiver is `Some` only when [`SessionConfig::telemetry`] enables at least one
    /// event kind. Drain enabled receivers while the agent runs; both channels are bounded.
    pub fn with_config_and_telemetry(
        config: SessionConfig,
    ) -> (
        Self,
        Option<AgentEventReceiver>,
        Option<TelemetryEventReceiver>,
    ) {
        let (events, receiver) = if config.notifications.is_enabled() {
            let (sender, receiver) = sync_channel(config.event_channel_capacity());
            (Some(sender), Some(receiver))
        } else {
            (None, None)
        };
        let (telemetry, telemetry_receiver) = if config.telemetry.is_enabled() {
            let (sender, receiver) = sync_channel(config.telemetry_channel_capacity());
            (Some(sender), Some(receiver))
        } else {
            (None, None)
        };

        (
            Self {
                turns: Vec::new(),
                next_turn_id: 1,
                config,
                events,
                telemetry,
            },
            receiver,
            telemetry_receiver,
        )
    }

    /// Create a configured session seeded with a system prompt.
    pub fn with_system_prompt_and_config(
        prompt: impl Into<String>,
        config: SessionConfig,
    ) -> (Self, Option<AgentEventReceiver>) {
        let (mut session, receiver) = Self::with_config(config);
        session.push_system(prompt);
        (session, receiver)
    }

    /// Create a configured session seeded with a system prompt, with event and telemetry receivers.
    pub fn with_system_prompt_config_and_telemetry(
        prompt: impl Into<String>,
        config: SessionConfig,
    ) -> (
        Self,
        Option<AgentEventReceiver>,
        Option<TelemetryEventReceiver>,
    ) {
        let (mut session, events, telemetry) = Self::with_config_and_telemetry(config);
        session.push_system(prompt);
        (session, events, telemetry)
    }

    /// Current session configuration.
    pub fn config(&self) -> &SessionConfig {
        &self.config
    }

    /// Mutable session configuration.
    pub fn config_mut(&mut self) -> &mut SessionConfig {
        &mut self.config
    }

    /// Persisted session turns.
    pub fn turns(&self) -> &[SessionTurn] {
        &self.turns
    }

    /// Estimated context usage and largest consumers.
    pub fn context_usage_breakdown(&self) -> ContextUsageBreakdown {
        self.context_usage_breakdown_with_limit(self.turns.len())
    }

    /// Rough token estimate for the active context.
    pub fn estimated_context_tokens(&self) -> usize {
        self.context_usage_breakdown().total_estimated_tokens
    }

    /// Convert the session history into provider chat messages.
    pub fn messages(&self) -> Vec<ChatMessage> {
        self.turns.iter().cloned().map(ChatMessage::from).collect()
    }

    /// Append a system message.
    pub fn push_system(&mut self, content: impl Into<String>) {
        self.push(ChatMessage::system(content));
    }

    /// Append a user message.
    pub fn push_user(&mut self, content: impl Into<String>) {
        self.push(ChatMessage::user(content));
    }

    /// Append a plain assistant message.
    pub fn push_assistant(&mut self, content: impl Into<String>) {
        self.push(ChatMessage::assistant(content));
    }

    /// Append any chat message, preserving reasoning and tool-call fields.
    pub fn push(&mut self, message: ChatMessage) {
        self.push_turn(message.into());
    }

    fn push_turn(&mut self, mut turn: SessionTurn) -> u64 {
        if turn.id == 0 {
            turn.id = self.allocate_turn_id();
        } else {
            self.next_turn_id = self.next_turn_id.max(turn.id.saturating_add(1));
        }
        let id = turn.id;
        self.emit_message_telemetry(&turn);
        self.turns.push(turn);
        id
    }

    fn allocate_turn_id(&mut self) -> u64 {
        let id = self.next_turn_id;
        self.next_turn_id = self.next_turn_id.saturating_add(1).max(1);
        id
    }

    fn definitions_with_compaction(&self, registry: &ToolRegistry) -> Vec<ToolDefinition> {
        let mut definitions = registry.definitions();
        if self.config.compaction.enabled && self.config.compaction.always_offer_tools {
            definitions.extend(compaction_tool_definitions());
        }
        definitions
    }

    fn context_usage_breakdown_with_limit(&self, limit: usize) -> ContextUsageBreakdown {
        compaction::usage_breakdown(&self.turns, limit)
    }

    fn compact_remove_messages(&mut self, turn_ids: &[u64], reason: &str) -> Result<String> {
        compaction::remove_messages(&mut self.turns, &self.config.compaction, turn_ids, reason)
    }

    fn compact_summarize_messages(
        &mut self,
        turn_ids: &[u64],
        summary: &str,
        reason: &str,
    ) -> Result<String> {
        compaction::summarize_messages(
            &mut self.turns,
            &mut self.next_turn_id,
            &self.config.compaction,
            turn_ids,
            summary,
            reason,
        )
    }

    fn remove_previous_compaction_tool_messages(&mut self) -> usize {
        compaction::remove_previous_tool_messages(&mut self.turns)
    }

    /// Add a user message, send one provider request without tools, and persist the assistant reply.
    pub async fn send_user_message(
        &mut self,
        provider: &ChatProvider,
        content: impl Into<String>,
    ) -> Result<ChatResponse> {
        self.push_user(content);
        self.complete(provider).await
    }

    /// Send one provider request using the current session history, without tools.
    pub async fn complete(&mut self, provider: &ChatProvider) -> Result<ChatResponse> {
        let response = self
            .send_provider_request(provider, self.messages(), Vec::new())
            .await?;
        self.push(response.message.clone());
        Ok(response)
    }

    /// Send one provider request with tool definitions but do not execute returned tool calls.
    ///
    /// Use [`ChatSession::run_with_tools`] or [`ChatSession::run_user_message_with_tools`] when you
    /// want the session to execute tools and continue until the model returns a final answer.
    pub async fn complete_with_tools(
        &mut self,
        provider: &ChatProvider,
        registry: &ToolRegistry,
    ) -> Result<ChatResponse> {
        let response = self
            .send_provider_request(provider, self.messages(), registry.definitions())
            .await?;
        self.push(response.message.clone());
        Ok(response)
    }

    /// Add a user message and run the managed tool loop.
    ///
    /// This repeatedly sends requests, executes registry and compaction tool calls, records tool
    /// results, and stops when the model returns an assistant message without tool calls.
    pub async fn run_user_message_with_tools(
        &mut self,
        provider: &ChatProvider,
        registry: &ToolRegistry,
        content: impl Into<String>,
    ) -> Result<ChatResponse> {
        self.push_user(content);
        self.run_with_tools(provider, registry).await
    }

    /// Run the managed tool loop from the current session state.
    ///
    /// Built-in compaction tools are offered according to [`CompactionConfig`]. Normal tool rounds
    /// are bounded by [`SessionConfig::max_tool_rounds`].
    pub async fn run_with_tools(
        &mut self,
        provider: &ChatProvider,
        registry: &ToolRegistry,
    ) -> Result<ChatResponse> {
        let mut completed_tool_rounds = 0;

        loop {
            self.maybe_compact(provider).await?;
            self.emit(AgentEvent::ModelRequestStarted {
                completed_tool_rounds,
            });
            let response = self
                .send_provider_request(
                    provider,
                    self.messages(),
                    self.definitions_with_compaction(registry),
                )
                .await?;
            self.emit(AgentEvent::ModelResponseReceived {
                completed_tool_rounds,
                tool_calls: response.message.tool_calls.len(),
                content_len: response.message.content.len(),
            });

            if response.message.tool_calls.is_empty() {
                self.push(response.message.clone());
                return Ok(response);
            }

            if completed_tool_rounds >= self.config.max_tool_rounds {
                self.emit(AgentEvent::MaxToolRoundsReached {
                    max_tool_rounds: self.config.max_tool_rounds,
                });
                return Err(Error::Tool(format!(
                    "stopped after {} tool rounds without a final answer",
                    self.config.max_tool_rounds
                )));
            }

            let round = completed_tool_rounds + 1;
            self.emit(AgentEvent::ToolRoundStarted {
                round,
                tool_calls: response.message.tool_calls.len(),
            });
            self.push(response.message.clone());
            for call in &response.message.tool_calls {
                if is_compaction_tool(&call.function.name) {
                    let result = self.execute_compaction_tool_call(call);
                    self.push(result.into_message());
                } else {
                    self.execute_tool_call_reporting_errors(registry, call, Some(round))
                        .await;
                }
            }
            completed_tool_rounds = round;
        }
    }

    async fn maybe_compact(&mut self, provider: &ChatProvider) -> Result<()> {
        if !self.config.compaction.enabled {
            return Ok(());
        }

        let initial_estimate = self.estimated_context_tokens();
        if initial_estimate < self.config.compaction.trigger_estimated_tokens {
            return Ok(());
        }
        self.remove_previous_compaction_tool_messages();
        let before = self.estimated_context_tokens();

        self.emit(AgentEvent::CompactionStarted {
            estimated_tokens: before,
            target_estimated_tokens: self.config.compaction.target_estimated_tokens,
        });
        let breakdown =
            self.context_usage_breakdown_with_limit(self.config.compaction.top_consumers);
        self.emit(AgentEvent::CompactionBreakdown {
            total_estimated_tokens: breakdown.total_estimated_tokens,
            largest_turns: breakdown.largest_turns.clone(),
            largest_tool_calls: breakdown.largest_tool_calls.clone(),
        });

        let mut transient_messages = self.messages();
        transient_messages.push(ChatMessage::system(compaction::instruction(
            &self.config.compaction,
            &breakdown,
        )));

        let mut rounds = 0;
        while rounds < self.config.compaction.max_compaction_rounds {
            let response = self
                .send_provider_request(
                    provider,
                    transient_messages.clone(),
                    compaction_tool_definitions(),
                )
                .await?;
            if response.message.tool_calls.is_empty() {
                self.emit(AgentEvent::CompactionSkipped {
                    reason: "model did not call a compaction tool".to_string(),
                });
                break;
            }

            self.push(response.message.clone());
            transient_messages.push(response.message.clone());
            let mut result_messages = Vec::new();
            for call in &response.message.tool_calls {
                let result = self.execute_compaction_tool_call(call);
                let message = result.into_message();
                self.push(message.clone());
                result_messages.push(message);
            }
            transient_messages.extend(result_messages);
            rounds += 1;

            if self.estimated_context_tokens() <= self.config.compaction.target_estimated_tokens {
                break;
            }
        }

        let after = self.estimated_context_tokens();
        self.emit(AgentEvent::CompactionFinished {
            before_estimated_tokens: before,
            after_estimated_tokens: after,
            rounds,
        });
        Ok(())
    }

    async fn send_provider_request(
        &self,
        provider: &ChatProvider,
        messages: Vec<ChatMessage>,
        tools: Vec<ToolDefinition>,
    ) -> Result<ChatResponse> {
        let request = provider.build_request(&messages, tools);
        self.emit_model_request_telemetry(provider, &request);
        let response = provider.send_request(request).await?;
        self.emit_model_response_telemetry(provider, &response);
        Ok(response)
    }

    #[cfg(test)]
    fn execute_compaction_response(&mut self, message: ChatMessage) -> Result<()> {
        self.push(message.clone());
        for call in &message.tool_calls {
            if is_compaction_tool(&call.function.name) {
                let result = self.execute_compaction_tool_call(call);
                self.push(result.into_message());
            } else {
                self.emit(AgentEvent::CompactionSkipped {
                    reason: format!(
                        "ignored non-compaction tool `{}` in a compaction response",
                        call.function.name
                    ),
                });
            }
        }
        Ok(())
    }

    fn execute_compaction_tool_call(&mut self, call: &ToolCall) -> ToolResult {
        self.emit(AgentEvent::CompactionToolCallStarted {
            tool_call_id: call.id.clone(),
            name: call.function.name.clone(),
            arguments: call.function.arguments.clone(),
        });
        self.emit_tool_call_started_telemetry(call);

        let result = match call.function.name.as_str() {
            COMPACT_REMOVE_MESSAGES => parse_remove_args(&call.function.arguments)
                .and_then(|args| self.compact_remove_messages(&args.turn_ids, &args.reason)),
            COMPACT_SUMMARIZE_MESSAGES => {
                parse_summarize_args(&call.function.arguments).and_then(|args| {
                    self.compact_summarize_messages(&args.turn_ids, &args.summary, &args.reason)
                })
            }
            other => Err(Error::UnknownTool(other.to_string())),
        };

        let result = match result {
            Ok(content) => ToolResult {
                tool_call_id: call.id.clone(),
                name: call.function.name.clone(),
                content: compaction_acknowledgement(&call.function.name, &content),
            },
            Err(error) => ToolResult::error(call, error),
        };
        self.emit_tool_result_telemetry(call, &result);
        result
    }

    pub async fn execute_tool_calls(
        &mut self,
        registry: &ToolRegistry,
        response: &ChatResponse,
    ) -> Result<Vec<ToolResult>> {
        let results = registry.execute_calls(&response.message.tool_calls).await?;
        for result in &results {
            self.push(result.clone().into_message());
        }
        Ok(results)
    }

    pub async fn execute_tool_calls_reporting_errors(
        &mut self,
        registry: &ToolRegistry,
        response: &ChatResponse,
    ) -> Vec<ToolResult> {
        let mut results = Vec::with_capacity(response.message.tool_calls.len());
        for call in &response.message.tool_calls {
            results.push(
                self.execute_tool_call_reporting_errors(registry, call, None)
                    .await,
            );
        }
        results
    }

    async fn execute_tool_call_reporting_errors(
        &mut self,
        registry: &ToolRegistry,
        call: &ToolCall,
        round: Option<usize>,
    ) -> ToolResult {
        if let Some(round) = round {
            self.emit(AgentEvent::ToolCallStarted {
                round,
                tool_call_id: call.id.clone(),
                name: call.function.name.clone(),
                arguments: call.function.arguments.clone(),
            });
        }
        self.emit_tool_call_started_telemetry(call);

        let result = match registry.execute_call(call).await {
            Ok(result) => {
                if let Some(round) = round {
                    self.emit(AgentEvent::ToolCallFinished {
                        round,
                        tool_call_id: result.tool_call_id.clone(),
                        name: result.name.clone(),
                        content_len: result.content.len(),
                    });
                }
                self.emit_tool_result_telemetry(call, &result);
                result
            }
            Err(error) => {
                if let Some(round) = round {
                    self.emit(AgentEvent::ToolCallFailed {
                        round,
                        tool_call_id: call.id.clone(),
                        name: call.function.name.clone(),
                        error: error.to_string(),
                    });
                }
                self.emit_tool_error_telemetry(call, &error);
                ToolResult::error(call, error)
            }
        };
        self.push(result.clone().into_message());
        result
    }

    fn emit(&self, event: AgentEvent) {
        if !self.config.notifications.should_emit(&event) {
            return;
        }

        if let Some(events) = &self.events {
            let _ = events.send(event);
        }
    }

    fn emit_telemetry(&self, event: TelemetryEvent) {
        if !self.config.telemetry.should_emit(&event) {
            return;
        }

        if let Some(telemetry) = &self.telemetry {
            let _ = telemetry.send(event);
        }
    }

    fn emit_message_telemetry(&self, turn: &SessionTurn) {
        match turn.role {
            MessageRole::User => self.emit_telemetry(TelemetryEvent::UserMessage {
                turn_id: turn.id,
                content_len: turn.content.len(),
                content: self.config.telemetry.messages.then(|| turn.content.clone()),
            }),
            MessageRole::Assistant => self.emit_telemetry(TelemetryEvent::AssistantMessage {
                turn_id: turn.id,
                content_len: turn.content.len(),
                tool_calls: turn.tool_calls.len(),
                content: self.config.telemetry.messages.then(|| turn.content.clone()),
            }),
            MessageRole::Tool => self.emit_telemetry(TelemetryEvent::ToolMessage {
                turn_id: turn.id,
                tool_call_id: turn.tool_call_id.clone().unwrap_or_default(),
                name: turn.name.clone().unwrap_or_default(),
                content_len: turn.content.len(),
                content: self
                    .config
                    .telemetry
                    .tool_payloads
                    .then(|| turn.content.clone()),
            }),
            MessageRole::System => {}
        }
    }

    fn emit_model_request_telemetry(&self, provider: &ChatProvider, request: &ChatRequest) {
        self.emit_telemetry(TelemetryEvent::ModelRequest {
            provider: provider.config().name.clone(),
            model: request.model.clone(),
            message_count: request.messages.len(),
            tool_count: request.tools.len(),
            request: self
                .config
                .telemetry
                .model_payloads
                .then(|| request.clone()),
        });
    }

    fn emit_model_response_telemetry(&self, provider: &ChatProvider, response: &ChatResponse) {
        self.emit_telemetry(TelemetryEvent::ModelResponse {
            provider: provider.config().name.clone(),
            model: provider.config().default_model.clone(),
            content_len: response.message.content.len(),
            tool_calls: response.message.tool_calls.len(),
            response: self
                .config
                .telemetry
                .model_payloads
                .then(|| response.clone()),
        });
    }

    fn emit_tool_call_started_telemetry(&self, call: &ToolCall) {
        self.emit_telemetry(TelemetryEvent::ToolCallStarted {
            tool_call_id: call.id.clone(),
            name: call.function.name.clone(),
            arguments_len: call.function.arguments.len(),
            arguments: self
                .config
                .telemetry
                .tool_payloads
                .then(|| call.function.arguments.clone()),
        });
    }

    fn emit_tool_result_telemetry(&self, call: &ToolCall, result: &ToolResult) {
        self.emit_telemetry(TelemetryEvent::ToolCallFinished {
            tool_call_id: result.tool_call_id.clone(),
            name: result.name.clone(),
            content_len: result.content.len(),
            content: self
                .config
                .telemetry
                .tool_payloads
                .then(|| result.content.clone()),
        });

        if result.content.starts_with("Tool error:") {
            self.emit_tool_error_telemetry(call, &Error::Tool(result.content.clone()));
        }
    }

    fn emit_tool_error_telemetry(&self, call: &ToolCall, error: &Error) {
        self.emit_telemetry(TelemetryEvent::ToolCallFailed {
            tool_call_id: call.id.clone(),
            name: call.function.name.clone(),
            error: error.to_string(),
        });
    }
}

#[cfg(test)]
mod tests;
