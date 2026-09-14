use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use crate::content::{ContentPart, MessageContent};
use crate::tools::ToolDefinition;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct ReasoningPayload {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_details: Option<Value>,
}

impl ReasoningPayload {
    pub fn is_empty(&self) -> bool {
        self.reasoning.is_none()
            && self.reasoning_content.is_none()
            && self.reasoning_details.is_none()
    }

    pub fn into_option(self) -> Option<Self> {
        (!self.is_empty()).then_some(self)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: MessageContent,
    /// Provider annotations, including parsed PDF content reused on subsequent requests.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub annotations: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_details: Option<Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    pub fn new(role: MessageRole, content: impl Into<MessageContent>) -> Self {
        Self {
            role,
            content: content.into(),
            annotations: Vec::new(),
            reasoning: None,
            reasoning_content: None,
            reasoning_details: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new(MessageRole::System, content.into())
    }

    pub fn user(content: impl Into<MessageContent>) -> Self {
        Self::new(MessageRole::User, content)
    }

    pub fn assistant(content: impl Into<MessageContent>) -> Self {
        Self::new(MessageRole::Assistant, content)
    }

    pub fn with_reasoning(mut self, reasoning: ReasoningPayload) -> Self {
        self.reasoning = reasoning.reasoning;
        self.reasoning_content = reasoning.reasoning_content;
        self.reasoning_details = reasoning.reasoning_details;
        self
    }

    pub fn reasoning_payload(&self) -> Option<ReasoningPayload> {
        ReasoningPayload {
            reasoning: self.reasoning.clone(),
            reasoning_content: self.reasoning_content.clone(),
            reasoning_details: self.reasoning_details.clone(),
        }
        .into_option()
    }

    pub fn without_reasoning(&self) -> Self {
        let mut message = self.clone();
        message.reasoning = None;
        message.reasoning_content = None;
        message.reasoning_details = None;
        message
    }

    pub fn with_tool_calls(mut self, tool_calls: Vec<ToolCall>) -> Self {
        self.tool_calls = tool_calls;
        self
    }

    pub fn tool_result(
        tool_call_id: impl Into<String>,
        name: impl Into<String>,
        content: impl Into<MessageContent>,
    ) -> Self {
        Self {
            role: MessageRole::Tool,
            content: content.into(),
            annotations: Vec::new(),
            reasoning: None,
            reasoning_content: None,
            reasoning_details: None,
            tool_calls: Vec::new(),
            tool_call_id: Some(tool_call_id.into()),
            name: Some(name.into()),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ToolCallFunction,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct ReasoningConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exclude: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
}

impl ReasoningConfig {
    pub fn openrouter_enabled() -> Self {
        Self {
            enabled: Some(true),
            exclude: Some(false),
            effort: None,
            max_tokens: None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ChatResponse {
    pub message: ChatMessage,
    pub reasoning: Option<ReasoningPayload>,
    pub raw: Value,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChatCompletionEnvelope {
    pub choices: Vec<ChatCompletionChoice>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChatCompletionChoice {
    pub message: AssistantWireMessage,
}

#[derive(Debug, Deserialize)]
pub(crate) struct AssistantWireMessage {
    pub content: Option<MessageContent>,
    #[serde(default)]
    pub annotations: Vec<Value>,
    #[serde(default)]
    pub reasoning: Option<Value>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub reasoning_details: Option<Value>,
    #[serde(default, deserialize_with = "null_tool_calls")]
    pub tool_calls: Vec<ToolCall>,
}

fn null_tool_calls<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<ToolCall>, D::Error> {
    Ok(Option::<Vec<ToolCall>>::deserialize(deserializer)?.unwrap_or_default())
}

impl AssistantWireMessage {
    pub fn into_chat_message(self) -> ChatMessage {
        let mut content = self.content.unwrap_or_default();
        if !self.tool_calls.is_empty() && contains_leaked_tool_markup(&content.text()) {
            match &mut content {
                MessageContent::Text(text) => text.clear(),
                MessageContent::Parts(parts) => parts.retain(|part| {
                    !matches!(part,
                    ContentPart::Text { text } if contains_leaked_tool_markup(text))
                }),
            }
        }
        ChatMessage {
            role: MessageRole::Assistant,
            content,
            annotations: self.annotations,
            reasoning: self.reasoning,
            reasoning_content: self.reasoning_content,
            reasoning_details: self.reasoning_details,
            tool_calls: self.tool_calls,
            tool_call_id: None,
            name: None,
        }
    }
}

fn contains_leaked_tool_markup(content: &str) -> bool {
    content.contains("DSML") && (content.contains("tool_calls") || content.contains("invoke"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structured_tool_calls_discard_leaked_dsml_content() {
        let message = AssistantWireMessage {
            annotations: Vec::new(),
            content: Some(
                "Let me check. <｜DSML｜tool_calls><｜DSML｜invoke name=\"read\">".into(),
            ),
            reasoning: None,
            reasoning_content: None,
            reasoning_details: None,
            tool_calls: vec![ToolCall {
                id: "call_1".into(),
                kind: "function".into(),
                function: ToolCallFunction {
                    name: "read".into(),
                    arguments: r#"{"path":"notes.txt"}"#.into(),
                },
            }],
        }
        .into_chat_message();

        assert!(message.content.is_empty());
        assert_eq!(message.tool_calls.len(), 1);
    }

    #[test]
    fn ordinary_tool_call_preambles_are_preserved() {
        let message = AssistantWireMessage {
            annotations: Vec::new(),
            content: Some("Let me check that file.".into()),
            reasoning: None,
            reasoning_content: None,
            reasoning_details: None,
            tool_calls: vec![ToolCall::default()],
        }
        .into_chat_message();

        assert_eq!(message.content, "Let me check that file.");
    }
}
