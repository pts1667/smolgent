use std::sync::Arc;

use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde_json::{Value, json};
use url::Url;

use crate::chat::{
    ChatCompletionEnvelope, ChatMessage, ChatRequest, ChatResponse, ReasoningConfig,
};
use crate::secrets::SecretStore;
use crate::tools::ToolDefinition;
use crate::{Error, Result};

/// Supported provider presets.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProviderKind {
    /// OpenRouter's OpenAI-compatible chat-completions endpoint.
    OpenRouter,
    /// A local llama.cpp server exposing `/v1/chat/completions`.
    LlamaCpp,
    /// Generic OpenAI-compatible endpoint.
    OpenAiCompatible,
}

/// Where a provider should obtain its API key.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ApiKeyRef {
    /// No API key is sent.
    None,
    /// Use the provided literal API key.
    Literal(String),
    /// Look up an API key by provider id in the configured [`crate::SecretStore`].
    Keyring(String),
}

/// Configuration for an OpenAI-compatible chat-completions provider.
#[derive(Clone, Debug)]
pub struct ProviderConfig {
    /// Human-readable provider name used in telemetry.
    pub name: String,
    /// Provider preset.
    pub kind: ProviderKind,
    /// Full chat-completions URL.
    pub chat_completions_url: Url,
    /// API-key lookup behavior.
    pub api_key: ApiKeyRef,
    /// Model used when building requests.
    pub default_model: String,
    /// Extra HTTP headers to include on every request.
    pub headers: Vec<(String, String)>,
    /// Optional provider-specific reasoning configuration.
    pub reasoning: Option<ReasoningConfig>,
}

impl ProviderConfig {
    /// OpenRouter configuration using a compatibility key id named `openrouter`.
    ///
    /// Apps should usually prefer [`ProviderConfig::openrouter_with_keyring`] so each application
    /// can choose its own keyring id instead of sharing one universal OpenRouter key.
    pub fn openrouter(model: impl Into<String>) -> Result<Self> {
        Self::openrouter_with_keyring(model, "openrouter")
    }

    /// OpenRouter configuration using an app-provided keyring id.
    ///
    /// The same id must be used when storing the key through [`crate::SecretStore::set_api_key`].
    pub fn openrouter_with_keyring(
        model: impl Into<String>,
        keyring_id: impl Into<String>,
    ) -> Result<Self> {
        Ok(Self {
            name: "openrouter".to_string(),
            kind: ProviderKind::OpenRouter,
            chat_completions_url: Url::parse("https://openrouter.ai/api/v1/chat/completions")?,
            api_key: ApiKeyRef::Keyring(keyring_id.into()),
            default_model: model.into(),
            headers: Vec::new(),
            reasoning: Some(ReasoningConfig::openrouter_enabled()),
        })
    }

    /// llama.cpp server configuration.
    ///
    /// The base URL should be the server root, for example `http://127.0.0.1:8080`.
    pub fn llama_cpp(base_url: impl AsRef<str>, model: impl Into<String>) -> Result<Self> {
        let base = Url::parse(base_url.as_ref())?;
        Ok(Self {
            name: "llama-cpp".to_string(),
            kind: ProviderKind::LlamaCpp,
            chat_completions_url: base.join("/v1/chat/completions")?,
            api_key: ApiKeyRef::None,
            default_model: model.into(),
            headers: Vec::new(),
            reasoning: None,
        })
    }
}

/// HTTP client wrapper for sending chat-completions requests.
#[derive(Clone)]
pub struct ChatProvider {
    client: reqwest::Client,
    config: ProviderConfig,
    secrets: Option<Arc<dyn SecretStore>>,
}

impl ChatProvider {
    /// Create a provider using the default `reqwest` client.
    pub fn new(config: ProviderConfig) -> Self {
        Self {
            client: reqwest::Client::new(),
            config,
            secrets: None,
        }
    }

    /// Replace the HTTP client, useful for custom TLS/proxy settings or tests.
    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    /// Attach a secret store for providers configured with [`ApiKeyRef::Keyring`].
    pub fn with_secrets(mut self, secrets: Arc<dyn SecretStore>) -> Self {
        self.secrets = Some(secrets);
        self
    }

    /// Provider configuration.
    pub fn config(&self) -> &ProviderConfig {
        &self.config
    }

    /// Send messages without tools.
    pub async fn send_messages(&self, messages: &[ChatMessage]) -> Result<ChatResponse> {
        self.send_messages_with_tools(messages, Vec::new()).await
    }

    /// Send messages with tool definitions.
    pub async fn send_messages_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Vec<ToolDefinition>,
    ) -> Result<ChatResponse> {
        let request = self.build_request(messages, tools);
        self.send_request(request).await
    }

    /// Build a request without sending it.
    pub fn build_request(
        &self,
        messages: &[ChatMessage],
        tools: Vec<ToolDefinition>,
    ) -> ChatRequest {
        ChatRequest {
            model: self.config.default_model.clone(),
            messages: messages.to_vec(),
            tools,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            reasoning: self.config.reasoning.clone(),
            response_format: None,
        }
    }

    /// Send a pre-built request and parse the response.
    pub async fn send_request(&self, request: ChatRequest) -> Result<ChatResponse> {
        let raw = self
            .client
            .post(self.config.chat_completions_url.clone())
            .headers(self.headers()?)
            .json(&request)
            .send()
            .await?
            .error_for_status()?
            .json::<Value>()
            .await?;

        self.parse_response(raw)
    }

    fn headers(&self) -> Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        if let Some(api_key) = self.api_key()? {
            let value = HeaderValue::from_str(&format!("Bearer {api_key}"))?;
            headers.insert(AUTHORIZATION, value);
        }

        for (name, value) in &self.config.headers {
            let name = HeaderName::from_bytes(name.as_bytes())?;
            let value = HeaderValue::from_str(value)?;
            headers.insert(name, value);
        }

        Ok(headers)
    }

    fn api_key(&self) -> Result<Option<String>> {
        match &self.config.api_key {
            ApiKeyRef::None => Ok(None),
            ApiKeyRef::Literal(api_key) => Ok(Some(api_key.clone())),
            ApiKeyRef::Keyring(provider_id) => {
                let secrets = self
                    .secrets
                    .as_ref()
                    .ok_or_else(|| Error::MissingApiKey(provider_id.clone()))?;
                secrets
                    .get_api_key(provider_id)?
                    .ok_or_else(|| Error::MissingApiKey(provider_id.clone()))
                    .map(Some)
            }
        }
    }

    fn parse_response(&self, raw: Value) -> Result<ChatResponse> {
        let envelope: ChatCompletionEnvelope = serde_json::from_value(raw.clone())?;
        let message = envelope
            .choices
            .into_iter()
            .next()
            .ok_or(Error::MissingAssistantMessage)?
            .message
            .into_chat_message();
        let reasoning = message.reasoning_payload();

        Ok(ChatResponse {
            message,
            reasoning,
            raw,
        })
    }
}

pub fn json_object_response_format() -> Value {
    json!({ "type": "json_object" })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use httpmock::Method::POST;
    use httpmock::MockServer;
    use keyring_core::CredentialStore;
    use serde_json::json;

    use super::*;
    use crate::ReasoningPayload;
    use crate::secrets::{KeyringCoreSecretStore, SecretStore};

    #[tokio::test]
    async fn openrouter_request_includes_auth_and_reasoning() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/api/v1/chat/completions")
                .header("authorization", "Bearer sk-test")
                .body_contains("\"reasoning\":{\"enabled\":true,\"exclude\":false}");
            then.status(200).json_body(json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "hello",
                        "reasoning_content": "worked it out"
                    }
                }]
            }));
        });

        let keyring: Arc<CredentialStore> = keyring_core::mock::Store::new().unwrap();
        let secrets = Arc::new(KeyringCoreSecretStore::with_store("smolgent-test", keyring));
        secrets
            .set_api_key("test-app/openrouter", "sk-test")
            .unwrap();

        let mut config =
            ProviderConfig::openrouter_with_keyring("openai/gpt-oss-20b", "test-app/openrouter")
                .unwrap();
        config.chat_completions_url =
            Url::parse(&format!("{}/api/v1/chat/completions", server.base_url())).unwrap();

        let provider = ChatProvider::new(config).with_secrets(secrets);
        let response = provider
            .send_messages(&[ChatMessage::user("Say hello")])
            .await
            .unwrap();

        mock.assert();
        assert_eq!(response.message.content, "hello");
        assert_eq!(
            response.reasoning.unwrap().reasoning_content,
            Some("worked it out".to_string())
        );
    }

    #[tokio::test]
    async fn llama_cpp_request_preserves_replayed_reasoning() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/v1/chat/completions")
                .body_contains("\"content\":\"previous\"")
                .body_contains("\"reasoning_content\":\"hidden chain\"");
            then.status(200).json_body(json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "next"
                    }
                }]
            }));
        });

        let config = ProviderConfig::llama_cpp(server.base_url(), "local-model").unwrap();
        let provider = ChatProvider::new(config);
        let prior = ChatMessage::assistant("previous").with_reasoning(ReasoningPayload {
            reasoning: None,
            reasoning_content: Some("hidden chain".to_string()),
            reasoning_details: None,
        });

        let response = provider.send_messages(&[prior]).await.unwrap();

        mock.assert();
        assert_eq!(response.message.content, "next");
    }

    #[tokio::test]
    async fn request_sends_tool_definitions_and_parses_tool_calls() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/v1/chat/completions")
                .body_contains("\"tools\":[{\"type\":\"function\"")
                .body_contains("\"name\":\"add\"");
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

        let config = ProviderConfig::llama_cpp(server.base_url(), "local-model").unwrap();
        let provider = ChatProvider::new(config);
        let tool = ToolDefinition::new(
            "add",
            "Add two numbers.",
            json!({
                "type": "object",
                "properties": {
                    "a": { "type": "integer" },
                    "b": { "type": "integer" }
                }
            }),
        );

        let response = provider
            .send_messages_with_tools(&[ChatMessage::user("Add 2 and 3")], vec![tool])
            .await
            .unwrap();

        mock.assert();
        assert_eq!(response.message.tool_calls.len(), 1);
        assert_eq!(response.message.tool_calls[0].function.name, "add");
    }
}
