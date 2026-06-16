use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::chat::{ChatMessage, MessageRole, ToolCall};
use crate::{Error, Result};

pub mod builtin;

pub type ToolFuture = Pin<Box<dyn Future<Output = Result<Value>> + Send>>;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionToolDefinition,
}

impl ToolDefinition {
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

    pub fn name(&self) -> &str {
        &self.function.name
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FunctionToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Clone)]
pub struct Tool {
    definition: ToolDefinition,
    handler: Arc<dyn Fn(Value) -> ToolFuture + Send + Sync>,
}

impl Tool {
    pub fn new<F>(definition: ToolDefinition, handler: F) -> Self
    where
        F: Fn(Value) -> ToolFuture + Send + Sync + 'static,
    {
        Self {
            definition,
            handler: Arc::new(handler),
        }
    }

    pub fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    pub fn name(&self) -> &str {
        self.definition.name()
    }

    pub async fn call(&self, arguments: Value) -> Result<Value> {
        (self.handler)(arguments).await
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub name: String,
    pub content: String,
}

impl ToolResult {
    pub fn error(call: &ToolCall, error: impl fmt::Display) -> Self {
        Self {
            tool_call_id: call.id.clone(),
            name: call.function.name.clone(),
            content: format!(
                "Tool error: {error}\nPlease correct the tool call arguments and try again."
            ),
        }
    }

    pub fn into_message(self) -> ChatMessage {
        ChatMessage {
            role: MessageRole::Tool,
            content: self.content,
            reasoning: None,
            reasoning_content: None,
            reasoning_details: None,
            tool_calls: Vec::new(),
            tool_call_id: Some(self.tool_call_id),
            name: Some(self.name),
        }
    }
}

#[derive(Clone, Default)]
pub struct ToolRegistry {
    tools: HashMap<String, Tool>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, tool: Tool) -> Option<Tool> {
        self.tools.insert(tool.name().to_string(), tool)
    }

    pub fn with_tool(mut self, tool: Tool) -> Self {
        self.insert(tool);
        self
    }

    pub fn get(&self, name: &str) -> Option<&Tool> {
        self.tools.get(name)
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .values()
            .map(|tool| tool.definition().clone())
            .collect()
    }

    pub async fn execute_call(&self, call: &ToolCall) -> Result<ToolResult> {
        let tool = self
            .get(&call.function.name)
            .ok_or_else(|| Error::UnknownTool(call.function.name.clone()))?;
        let arguments = parse_tool_arguments(&call.function.arguments)?;
        let output = tool.call(arguments).await?;

        Ok(ToolResult {
            tool_call_id: call.id.clone(),
            name: call.function.name.clone(),
            content: stringify_tool_output(output),
        })
    }

    pub async fn execute_call_reporting_errors(&self, call: &ToolCall) -> ToolResult {
        match self.execute_call(call).await {
            Ok(result) => result,
            Err(error) => ToolResult::error(call, error),
        }
    }

    pub async fn execute_calls(&self, calls: &[ToolCall]) -> Result<Vec<ToolResult>> {
        let mut results = Vec::with_capacity(calls.len());
        for call in calls {
            results.push(self.execute_call(call).await?);
        }
        Ok(results)
    }

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
                .contains("Tool error: unknown tool 'missing'")
        );
        assert_eq!(message.tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(message.name.as_deref(), Some("missing"));
    }
}
