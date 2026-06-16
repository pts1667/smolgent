use crate::Result;
use crate::chat::{ChatMessage, ChatResponse, MessageRole, ReasoningPayload};
use crate::provider::ChatProvider;
use crate::tools::{ToolRegistry, ToolResult};

#[derive(Clone, Debug, PartialEq)]
pub struct SessionTurn {
    pub role: MessageRole,
    pub content: String,
    pub reasoning: Option<ReasoningPayload>,
}

impl From<ChatMessage> for SessionTurn {
    fn from(message: ChatMessage) -> Self {
        Self {
            role: message.role,
            content: message.content,
            reasoning: ReasoningPayload {
                reasoning: message.reasoning,
                reasoning_content: message.reasoning_content,
                reasoning_details: message.reasoning_details,
            }
            .into_option(),
        }
    }
}

impl From<SessionTurn> for ChatMessage {
    fn from(turn: SessionTurn) -> Self {
        let message = ChatMessage::new(turn.role, turn.content);
        match turn.reasoning {
            Some(reasoning) => message.with_reasoning(reasoning),
            None => message,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChatSession {
    turns: Vec<SessionTurn>,
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
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

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
}
