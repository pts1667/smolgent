use std::fmt;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};

use crate::chat::{ChatMessage, ChatResponse, MessageRole, ReasoningPayload, ToolCall};
use crate::provider::ChatProvider;
use crate::tools::{ToolRegistry, ToolResult};
use crate::{Error, Result};

#[derive(Clone, Debug, PartialEq)]
pub struct SessionTurn {
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

pub type AgentEventReceiver = Receiver<AgentEvent>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SessionConfig {
    pub max_tool_rounds: usize,
    pub event_channel_capacity: Option<usize>,
    pub notifications: NotificationConfig,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            max_tool_rounds: 8,
            event_channel_capacity: None,
            notifications: NotificationConfig::default(),
        }
    }
}

impl SessionConfig {
    pub fn event_channel_capacity(&self) -> usize {
        self.event_channel_capacity
            .unwrap_or_else(|| self.max_tool_rounds.saturating_mul(4).max(1))
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct NotificationConfig {
    pub model_requests: bool,
    pub model_responses: bool,
    pub tool_rounds: bool,
    pub tool_calls: bool,
    pub tool_results: bool,
    pub tool_errors: bool,
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
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.model_requests
            || self.model_responses
            || self.tool_rounds
            || self.tool_calls
            || self.tool_results
            || self.tool_errors
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
}

#[derive(Clone, Default)]
pub struct ChatSession {
    turns: Vec<SessionTurn>,
    config: SessionConfig,
    events: Option<SyncSender<AgentEvent>>,
}

impl fmt::Debug for ChatSession {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ChatSession")
            .field("turns", &self.turns)
            .field("config", &self.config)
            .field("events_enabled", &self.events.is_some())
            .finish()
    }
}

impl PartialEq for ChatSession {
    fn eq(&self, other: &Self) -> bool {
        self.turns == other.turns && self.config == other.config
    }
}

impl ChatSession {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_system_prompt(prompt: impl Into<String>) -> Self {
        let mut session = Self::new();
        session.push_system(prompt);
        session
    }

    pub fn with_config(config: SessionConfig) -> (Self, Option<AgentEventReceiver>) {
        let (events, receiver) = if config.notifications.is_enabled() {
            let (sender, receiver) = sync_channel(config.event_channel_capacity());
            (Some(sender), Some(receiver))
        } else {
            (None, None)
        };

        (
            Self {
                turns: Vec::new(),
                config,
                events,
            },
            receiver,
        )
    }

    pub fn with_system_prompt_and_config(
        prompt: impl Into<String>,
        config: SessionConfig,
    ) -> (Self, Option<AgentEventReceiver>) {
        let (mut session, receiver) = Self::with_config(config);
        session.push_system(prompt);
        (session, receiver)
    }

    pub fn config(&self) -> &SessionConfig {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut SessionConfig {
        &mut self.config
    }

    pub fn turns(&self) -> &[SessionTurn] {
        &self.turns
    }

    pub fn messages(&self) -> Vec<ChatMessage> {
        self.turns.iter().cloned().map(ChatMessage::from).collect()
    }

    pub fn push_system(&mut self, content: impl Into<String>) {
        self.push(ChatMessage::system(content));
    }

    pub fn push_user(&mut self, content: impl Into<String>) {
        self.push(ChatMessage::user(content));
    }

    pub fn push_assistant(&mut self, content: impl Into<String>) {
        self.push(ChatMessage::assistant(content));
    }

    pub fn push(&mut self, message: ChatMessage) {
        self.turns.push(message.into());
    }

    pub async fn send_user_message(
        &mut self,
        provider: &ChatProvider,
        content: impl Into<String>,
    ) -> Result<ChatResponse> {
        self.push_user(content);
        self.complete(provider).await
    }

    pub async fn complete(&mut self, provider: &ChatProvider) -> Result<ChatResponse> {
        let response = provider.send_messages(&self.messages()).await?;
        self.push(response.message.clone());
        Ok(response)
    }

    pub async fn complete_with_tools(
        &mut self,
        provider: &ChatProvider,
        registry: &ToolRegistry,
    ) -> Result<ChatResponse> {
        let response = provider
            .send_messages_with_tools(&self.messages(), registry.definitions())
            .await?;
        self.push(response.message.clone());
        Ok(response)
    }

    pub async fn run_user_message_with_tools(
        &mut self,
        provider: &ChatProvider,
        registry: &ToolRegistry,
        content: impl Into<String>,
    ) -> Result<ChatResponse> {
        self.push_user(content);
        self.run_with_tools(provider, registry).await
    }

    pub async fn run_with_tools(
        &mut self,
        provider: &ChatProvider,
        registry: &ToolRegistry,
    ) -> Result<ChatResponse> {
        let mut completed_tool_rounds = 0;

        loop {
            self.emit(AgentEvent::ModelRequestStarted {
                completed_tool_rounds,
            });
            let response = provider
                .send_messages_with_tools(&self.messages(), registry.definitions())
                .await?;
            self.push(response.message.clone());
            self.emit(AgentEvent::ModelResponseReceived {
                completed_tool_rounds,
                tool_calls: response.message.tool_calls.len(),
                content_len: response.message.content.len(),
            });

            if response.message.tool_calls.is_empty() {
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
            for call in &response.message.tool_calls {
                self.execute_tool_call_reporting_errors(registry, call, Some(round))
                    .await;
            }
            completed_tool_rounds = round;
        }
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
}

impl NotificationConfig {
    fn should_emit(&self, event: &AgentEvent) -> bool {
        match event {
            AgentEvent::ModelRequestStarted { .. } => self.model_requests,
            AgentEvent::ModelResponseReceived { .. } => self.model_responses,
            AgentEvent::ToolRoundStarted { .. } => self.tool_rounds,
            AgentEvent::ToolCallStarted { .. } => self.tool_calls,
            AgentEvent::ToolCallFinished { .. } => self.tool_results,
            AgentEvent::ToolCallFailed { .. } => self.tool_errors,
            AgentEvent::MaxToolRoundsReached { .. } => self.tool_rounds,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use httpmock::Method::POST;
    use httpmock::MockServer;
    use serde_json::json;

    use super::*;
    use crate::chat::ToolCallFunction;
    use crate::{ChatProvider, ProviderConfig, Tool, ToolDefinition};

    #[test]
    fn sessions_preserve_reasoning_in_history() {
        let mut session = ChatSession::new();
        session.push(
            ChatMessage::assistant("answer").with_reasoning(ReasoningPayload {
                reasoning: Some(json!("raw reasoning")),
                reasoning_content: Some("text reasoning".to_string()),
                reasoning_details: Some(json!([{ "type": "reasoning.text" }])),
            }),
        );

        let message = session.messages().pop().unwrap();
        assert_eq!(message.reasoning, Some(json!("raw reasoning")));
        assert_eq!(
            message.reasoning_content,
            Some("text reasoning".to_string())
        );
        assert_eq!(
            message.reasoning_details,
            Some(json!([{ "type": "reasoning.text" }]))
        );
    }

    #[test]
    fn sessions_preserve_tool_call_history() {
        let mut session = ChatSession::new();
        session.push(ChatMessage::assistant("").with_tool_calls(vec![ToolCall {
            id: "call_1".to_string(),
            kind: "function".to_string(),
            function: ToolCallFunction {
                name: "read".to_string(),
                arguments: r#"{"path":"Cargo.toml"}"#.to_string(),
            },
        }]));
        session.push(ChatMessage::tool_result(
            "call_1",
            "read",
            "[package]\nname = \"smolgent\"\n",
        ));

        let messages = session.messages();

        assert_eq!(messages[0].tool_calls.len(), 1);
        assert_eq!(messages[0].tool_calls[0].id, "call_1");
        assert_eq!(messages[1].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(messages[1].name.as_deref(), Some("read"));
    }

    #[tokio::test]
    async fn sessions_can_record_tool_errors_without_failing() {
        let mut session = ChatSession::new();
        let response = ChatResponse {
            message: ChatMessage::assistant("").with_tool_calls(vec![ToolCall {
                id: "call_1".to_string(),
                kind: "function".to_string(),
                function: ToolCallFunction {
                    name: "read".to_string(),
                    arguments: r#"{"path":"Cargo.toml"}"#.to_string(),
                },
            }]),
            reasoning: None,
            raw: json!({}),
        };

        let results = session
            .execute_tool_calls_reporting_errors(&ToolRegistry::new(), &response)
            .await;
        let messages = session.messages();

        assert_eq!(results.len(), 1);
        assert!(
            results[0]
                .content
                .contains("Tool error: unknown tool 'read'")
        );
        assert_eq!(messages[0].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(messages[0].name.as_deref(), Some("read"));
    }

    #[test]
    fn configured_sessions_create_bounded_event_streams() {
        let config = SessionConfig {
            max_tool_rounds: 3,
            notifications: NotificationConfig::tools(),
            ..SessionConfig::default()
        };
        assert_eq!(config.event_channel_capacity(), 12);

        let (_session, receiver) = ChatSession::with_config(config);

        assert!(receiver.is_some());

        let config = SessionConfig {
            max_tool_rounds: 3,
            event_channel_capacity: Some(2),
            notifications: NotificationConfig::tools(),
        };
        assert_eq!(config.event_channel_capacity(), 2);

        let (_session, receiver) = ChatSession::with_config(SessionConfig {
            notifications: NotificationConfig::none(),
            ..SessionConfig::default()
        });

        assert!(receiver.is_none());
    }

    #[tokio::test]
    async fn managed_tool_loop_runs_tools_and_emits_events() {
        let server = MockServer::start();
        let tool_call_mock = server.mock(|when, then| {
            when.method(POST)
                .path("/v1/chat/completions")
                .body_contains("\"messages\":[{\"role\":\"user\",\"content\":\"add 2 and 3\"}]")
                .body_contains("\"tools\":[{\"type\":\"function\"");
            then.status(200).json_body(json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "add",
                                "arguments": "{\"a\":2,\"b\":3}"
                            }
                        }]
                    }
                }]
            }));
        });
        let final_mock = server.mock(|when, then| {
            when.method(POST)
                .path("/v1/chat/completions")
                .body_contains("\"role\":\"tool\"")
                .body_contains("\"content\":\"5\"");
            then.status(200).json_body(json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "2 + 3 = 5"
                    }
                }]
            }));
        });

        let config = ProviderConfig::llama_cpp(server.base_url(), "local-model").unwrap();
        let provider = ChatProvider::new(config);
        let registry = ToolRegistry::new().with_tool(Tool::new(
            ToolDefinition::new(
                "add",
                "Add two numbers.",
                json!({
                    "type": "object",
                    "properties": {
                        "a": { "type": "integer" },
                        "b": { "type": "integer" }
                    },
                    "required": ["a", "b"]
                }),
            ),
            |arguments| {
                Box::pin(async move {
                    std::thread::sleep(Duration::from_millis(20));
                    Ok(json!(
                        arguments["a"].as_i64().unwrap() + arguments["b"].as_i64().unwrap()
                    ))
                })
            },
        ));
        let (mut session, receiver) = ChatSession::with_config(SessionConfig {
            notifications: NotificationConfig {
                tool_calls: true,
                tool_results: true,
                ..NotificationConfig::default()
            },
            ..SessionConfig::default()
        });
        let receiver = receiver.unwrap();
        let events = Arc::new(Mutex::new(Vec::new()));
        let thread_events = Arc::clone(&events);
        let collector = std::thread::spawn(move || {
            while let Ok(event) = receiver.recv_timeout(Duration::from_secs(1)) {
                thread_events.lock().unwrap().push(event);
            }
        });

        let response = session
            .run_user_message_with_tools(&provider, &registry, "add 2 and 3")
            .await
            .unwrap();
        drop(session);
        collector.join().unwrap();

        tool_call_mock.assert();
        final_mock.assert();
        assert_eq!(response.message.content, "2 + 3 = 5");
        let events = events.lock().unwrap();
        assert!(events.iter().any(
            |event| matches!(event, AgentEvent::ToolCallStarted { name, .. } if name == "add")
        ));
        assert!(events
            .iter()
            .any(|event| matches!(event, AgentEvent::ToolCallFinished { name, content_len, .. } if name == "add" && *content_len == 1)));
    }
}
