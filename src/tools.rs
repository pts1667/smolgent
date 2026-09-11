use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::chat::{ChatMessage, MessageContent, MessageRole, ToolCall};
use crate::{Error, Result};

pub mod builtin;
pub mod compact;
pub mod file;
pub mod media;

/// Async handler result type used by [`Tool`].
pub type ToolFuture = Pin<Box<dyn Future<Output = Result<Value>> + Send>>;

/// Async handler for tools that return text or actual multimodal content parts.
pub type ToolContentFuture = Pin<Box<dyn Future<Output = Result<MessageContent>> + Send>>;

#[derive(Clone)]
enum ToolHandler {
    Json(Arc<dyn Fn(Value) -> ToolFuture + Send + Sync>),
    Content(Arc<dyn Fn(Value) -> ToolContentFuture + Send + Sync>),
}

/// OpenAI-compatible function tool definition.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ToolDefinition {
    /// Tool type, normally `function`.
    #[serde(rename = "type")]
    pub kind: String,
    /// Function metadata exposed to the model.
    pub function: FunctionToolDefinition,
}

impl ToolDefinition {
    /// Create a function tool definition from a name, description, and JSON schema parameters.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: impl Serialize,
    ) -> Self {
        Self {
            kind: "function".to_string(),
            function: FunctionToolDefinition {
                name: name.into(),
                description: description.into(),
                parameters: serde_json::to_value(parameters).unwrap_or_else(|_| {
                    serde_json::json!({
                        "type": "object",
                        "properties": {},
                    })
                }),
            },
        }
    }

    /// Tool function name.
    pub fn name(&self) -> &str {
        &self.function.name
    }
}

/// Function metadata exposed in a [`ToolDefinition`].
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FunctionToolDefinition {
    /// Function name the model must call.
    pub name: String,
    /// Human-readable tool description.
    pub description: String,
    /// JSON schema for tool arguments.
    pub parameters: Value,
}

/// A callable tool with its model-facing definition.
#[derive(Clone)]
pub struct Tool {
    definition: ToolDefinition,
    handler: ToolHandler,
}

impl Tool {
    /// Create a tool from a definition and async handler.
    pub fn new<F>(definition: ToolDefinition, handler: F) -> Self
    where
        F: Fn(Value) -> ToolFuture + Send + Sync + 'static,
    {
        Self {
            definition,
            handler: ToolHandler::Json(Arc::new(handler)),
        }
    }

    /// Create a tool whose result preserves multimodal content without JSON stringification.
    pub fn new_multimodal<F>(definition: ToolDefinition, handler: F) -> Self
    where
        F: Fn(Value) -> ToolContentFuture + Send + Sync + 'static,
    {
        Self {
            definition,
            handler: ToolHandler::Content(Arc::new(handler)),
        }
    }

    /// Model-facing definition.
    pub fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    /// Tool function name.
    pub fn name(&self) -> &str {
        self.definition.name()
    }

    /// Call the tool handler with decoded JSON arguments. Multimodal results serialize
    /// as a string or content array. Use [`Self::call_content`] for typed content.
    pub async fn call(&self, arguments: Value) -> Result<Value> {
        match &self.handler {
            ToolHandler::Json(handler) => handler(arguments).await,
            ToolHandler::Content(handler) => Ok(serde_json::to_value(handler(arguments).await?)?),
        }
    }

    /// Execute a tool for replay to the model. Ordinary JSON tools still return JSON text.
    pub async fn call_content(&self, arguments: Value) -> Result<MessageContent> {
        match &self.handler {
            ToolHandler::Json(handler) => {
                Ok(stringify_tool_output(handler(arguments).await?).into())
            }
            ToolHandler::Content(handler) => handler(arguments).await,
        }
    }
}

/// Result of executing a tool call.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolResult {
    /// Provider tool-call id this result answers.
    pub tool_call_id: String,
    /// Tool name.
    pub name: String,
    /// Tool output sent back to the model, including media parts from multimodal tools.
    pub content: MessageContent,
}

impl ToolResult {
    /// Convert an execution error into a tool result the model can correct.
    pub fn error(call: &ToolCall, error: impl fmt::Display) -> Self {
        Self {
            tool_call_id: call.id.clone(),
            name: call.function.name.clone(),
            content: format!(
                "Tool error: {error}\nPlease correct the tool call arguments and try again."
            )
            .into(),
        }
    }

    /// Convert this result into a chat message with role `tool`.
    pub fn into_message(self) -> ChatMessage {
        ChatMessage {
            role: MessageRole::Tool,
            content: self.content,
            annotations: Vec::new(),
            reasoning: None,
            reasoning_content: None,
            reasoning_details: None,
            tool_calls: Vec::new(),
            tool_call_id: Some(self.tool_call_id),
            name: Some(self.name),
        }
    }
}

/// Storage and dispatcher for tools available to an agent.
#[derive(Clone, Default)]
pub struct ToolRegistry {
    tools: HashMap<String, Tool>,
}

impl ToolRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace a tool by name.
    pub fn insert(&mut self, tool: Tool) -> Option<Tool> {
        self.tools.insert(tool.name().to_string(), tool)
    }

    /// Builder-style insertion.
    pub fn with_tool(mut self, tool: Tool) -> Self {
        self.insert(tool);
        self
    }

    /// Look up a tool by name.
    pub fn get(&self, name: &str) -> Option<&Tool> {
        self.tools.get(name)
    }

    /// Definitions to send to a model.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .values()
            .map(|tool| tool.definition().clone())
            .collect()
    }

    /// Execute one provider tool call, returning an error if the tool or arguments are invalid.
    pub async fn execute_call(&self, call: &ToolCall) -> Result<ToolResult> {
        let tool = self
            .get(&call.function.name)
            .ok_or_else(|| Error::UnknownTool(call.function.name.clone()))?;
        let arguments = parse_tool_arguments(&call.function.arguments)?;
        let output = tool.call_content(arguments).await?;

        Ok(ToolResult {
            tool_call_id: call.id.clone(),
            name: call.function.name.clone(),
            content: output,
        })
    }

    /// Execute one provider tool call and convert any error into a model-visible tool result.
    pub async fn execute_call_reporting_errors(&self, call: &ToolCall) -> ToolResult {
        match self.execute_call(call).await {
            Ok(result) => result,
            Err(error) => ToolResult::error(call, error),
        }
    }

    /// Execute multiple tool calls, stopping on the first error.
    pub async fn execute_calls(&self, calls: &[ToolCall]) -> Result<Vec<ToolResult>> {
        let mut results = Vec::with_capacity(calls.len());
        for call in calls {
            results.push(self.execute_call(call).await?);
        }
        Ok(results)
    }

    /// Execute multiple tool calls and report each error as a tool result.
    pub async fn execute_calls_reporting_errors(&self, calls: &[ToolCall]) -> Vec<ToolResult> {
        let mut results = Vec::with_capacity(calls.len());
        for call in calls {
            results.push(self.execute_call_reporting_errors(call).await);
        }
        results
    }
}

fn parse_tool_arguments(arguments: &str) -> Result<Value> {
    if arguments.trim().is_empty() {
        Ok(Value::Object(Default::default()))
    } else {
        Ok(serde_json::from_str(arguments)?)
    }
}

fn stringify_tool_output(output: Value) -> String {
    match output {
        Value::String(text) => text,
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::chat::ToolCallFunction;

    #[tokio::test]
    async fn registry_executes_tool_calls() {
        let tool = Tool::new(
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
                    Ok(json!(
                        arguments["a"].as_i64().unwrap() + arguments["b"].as_i64().unwrap()
                    ))
                })
            },
        );
        let registry = ToolRegistry::new().with_tool(tool);
        let call = ToolCall {
            id: "call_1".to_string(),
            kind: "function".to_string(),
            function: ToolCallFunction {
                name: "add".to_string(),
                arguments: r#"{"a":2,"b":3}"#.to_string(),
            },
        };

        let result = registry.execute_call(&call).await.unwrap();

        assert_eq!(result.tool_call_id, "call_1");
        assert_eq!(result.content, "5");
    }

    #[tokio::test]
    async fn registry_reports_tool_errors_as_tool_results() {
        let registry = ToolRegistry::new();
        let call = ToolCall {
            id: "call_1".to_string(),
            kind: "function".to_string(),
            function: ToolCallFunction {
                name: "missing".to_string(),
                arguments: "{}".to_string(),
            },
        };

        let result = registry.execute_call_reporting_errors(&call).await;
        let message = result.clone().into_message();

        assert_eq!(result.tool_call_id, "call_1");
        assert_eq!(result.name, "missing");
        assert!(
            result
                .content
                .text()
                .contains("Tool error: unknown tool 'missing'")
        );
        assert_eq!(message.tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(message.name.as_deref(), Some("missing"));
    }
}
