use std::fmt;
use std::sync::Arc;

use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use url::Url;

use crate::chat::{
    ChatCompletionEnvelope, ChatMessage, ChatRequest, ChatResponse, ReasoningConfig,
};
use crate::secrets::SecretStore;
use crate::tools::ToolDefinition;
use crate::{ContentPart, Error, MessageContent, Result};

const MAX_RESPONSE_BYTES: usize = 10 * 1024 * 1024;
const MAX_ERROR_BODY_BYTES: usize = 16 * 1024;

/// Input modalities advertised by the configured model, not its output capabilities.
/// Format/size limits and upstream availability still depend on the selected model.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ModelCapabilities {
    pub model: String,
    pub input_modalities: Vec<String>,
}

impl ModelCapabilities {
    pub fn supports_input(&self, modality: &str) -> bool {
        self.input_modalities.iter().any(|input| input == modality)
    }
}

/// Supported provider presets.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProviderKind {
    /// OpenRouter's OpenAI-compatible chat-completions endpoint.
    OpenRouter,
    /// DeepSeek's chat-completions endpoint, including thinking mode.
    DeepSeek,
    /// A local llama.cpp server exposing `/v1/chat/completions`.
    LlamaCpp,
    /// Generic OpenAI-compatible endpoint.
    OpenAiCompatible,
}

/// Where a provider should obtain its API key.
#[derive(Clone, Eq, PartialEq)]
pub enum ApiKeyRef {
    /// No API key is sent.
    None,
    /// Use the provided literal API key.
    Literal(String),
    /// Look up an API key by provider id in the configured [`crate::SecretStore`].
    Keyring(String),
}

impl fmt::Debug for ApiKeyRef {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => formatter.write_str("None"),
            Self::Literal(_) => formatter
                .debug_tuple("Literal")
                .field(&"[REDACTED]")
                .finish(),
            Self::Keyring(id) => formatter.debug_tuple("Keyring").field(id).finish(),
        }
    }
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
    /// DeepSeek configuration using the compatibility keyring id `deepseek`.
    /// Prefer [`Self::deepseek_with_keyring`] for an application-specific key id.
    /// With `reasoning: None`, the API's default thinking settings apply.
    pub fn deepseek(model: impl Into<String>) -> Result<Self> {
        Self::deepseek_with_keyring(model, "deepseek")
    }

    /// DeepSeek configuration using an app-provided keyring id.
    /// `reasoning.enabled` maps to `thinking.type`, and `reasoning.effort` maps
    /// to `reasoning_effort`. Excluding reasoning or setting a separate reasoning
    /// token budget is unsupported. Reasoning is retained for tool-call replay.
    pub fn deepseek_with_keyring(
        model: impl Into<String>,
        keyring_id: impl Into<String>,
    ) -> Result<Self> {
        Ok(Self {
            name: "deepseek".into(),
            kind: ProviderKind::DeepSeek,
            chat_completions_url: Url::parse("https://api.deepseek.com/chat/completions")?,
            api_key: ApiKeyRef::Keyring(keyring_id.into()),
            default_model: model.into(),
            headers: Vec::new(),
            reasoning: None,
        })
    }

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
    /// An optional reverse-proxy/API path prefix is preserved (with or without a trailing slash).
    pub fn llama_cpp(base_url: impl AsRef<str>, model: impl Into<String>) -> Result<Self> {
        let mut base = Url::parse(base_url.as_ref())?;
        base.path_segments_mut()
            .map_err(|_| Error::Tool("provider URL cannot contain endpoint paths".into()))?
            .pop_if_empty()
            .extend(["v1", "chat", "completions"]);
        base.set_query(None);
        base.set_fragment(None);
        Ok(Self {
            name: "llama-cpp".to_string(),
            kind: ProviderKind::LlamaCpp,
            chat_completions_url: base,
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

    /// Look up the configured model's input modalities on OpenRouter or llama.cpp.
    ///
    /// llama.cpp uses `/props?model=...`; vision, audio, and video are enabled only when
    /// explicitly advertised. Older servers without metadata return `None`.
    /// Returns `None` for DeepSeek, generic providers, automatic routing, unknown model IDs, or missing
    /// modality metadata. Known aliases and variant suffixes are resolved by OpenRouter.
    /// Fetch once when constructing tools; rebuild them if the application's model changes.
    pub async fn model_capabilities(&self) -> Result<Option<ModelCapabilities>> {
        match self.config.kind {
            ProviderKind::LlamaCpp => self.llama_cpp_capabilities().await,
            ProviderKind::OpenRouter => self.openrouter_capabilities().await,
            ProviderKind::DeepSeek | ProviderKind::OpenAiCompatible => Ok(None),
        }
    }

    async fn llama_cpp_capabilities(&self) -> Result<Option<ModelCapabilities>> {
        let mut url = self.config.chat_completions_url.join("../../props")?;
        url.query_pairs_mut()
            .append_pair("model", &self.config.default_model);
        let Some(raw) = self.fetch_model_metadata(url).await? else {
            return Ok(None);
        };
        let Some(modalities) = raw.get("modalities").filter(|value| !value.is_null()) else {
            return Ok(None);
        };
        #[derive(Deserialize)]
        struct Modalities {
            #[serde(default)]
            vision: bool,
            #[serde(default)]
            audio: bool,
            #[serde(default)]
            video: bool,
        }
        let modalities: Modalities = serde_json::from_value(modalities.clone())?;
        let mut inputs = vec!["text".to_string()];
        for (enabled, name) in [
            (modalities.vision, "image"),
            (modalities.audio, "audio"),
            (modalities.video, "video"),
        ] {
            if enabled {
                inputs.push(name.into());
            }
        }
        Ok(Some(ModelCapabilities {
            model: self.config.default_model.clone(),
            input_modalities: inputs,
        }))
    }

    async fn openrouter_capabilities(&self) -> Result<Option<ModelCapabilities>> {
        let model = &self.config.default_model;
        if model == "openrouter/auto" {
            return Ok(None);
        }
        let segments = model.split('/').collect::<Vec<_>>();
        if segments.len() != 2
            || segments
                .iter()
                .any(|segment| segment.is_empty() || *segment == "." || *segment == "..")
        {
            return Ok(None);
        }
        let mut url = self.config.chat_completions_url.join("../model/")?;
        url.path_segments_mut()
            .map_err(|_| Error::Tool("provider URL cannot contain model paths".into()))?
            .pop_if_empty()
            .extend(segments);
        let Some(raw) = self.fetch_model_metadata(url).await? else {
            return Ok(None);
        };
        let Some(modalities) = raw.pointer("/data/architecture/input_modalities") else {
            return Ok(None);
        };
        if modalities.is_null() {
            return Ok(None);
        }
        Ok(Some(ModelCapabilities {
            model: model.clone(),
            input_modalities: serde_json::from_value(modalities.clone())?,
        }))
    }

    async fn fetch_model_metadata(&self, url: Url) -> Result<Option<Value>> {
        let response = self
            .client
            .get(url)
            .headers(self.headers()?)
            .timeout(std::time::Duration::from_secs(15))
            .send()
            .await?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        Ok(Some(Self::read_response_json(response).await?))
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
    /// llama.cpp video parts are translated to `input_video` at the HTTP boundary;
    /// DeepSeek reasoning controls and file parts are adapted to its wire format.
    /// The request and session content types remain provider-independent.
    pub async fn send_request(&self, request: ChatRequest) -> Result<ChatResponse> {
        let builder = self
            .client
            .post(self.config.chat_completions_url.clone())
            .headers(self.headers()?);
        let needs_video_conversion = self.config.kind == ProviderKind::LlamaCpp
            && request.messages.iter().any(|message| {
                matches!(&message.content, MessageContent::Parts(parts)
                    if parts.iter().any(|part| matches!(part, ContentPart::VideoUrl { .. })))
            });
        let builder = if self.config.kind == ProviderKind::DeepSeek {
            builder.json(&deepseek_request_body(&request)?)
        } else if needs_video_conversion {
            let mut body = serde_json::to_value(&request)?;
            // Transform only typed message content, including tool results. Do not touch
            // tool arguments/schemas or other JSON that happens to contain similar keys.
            if let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) {
                for message in messages {
                    if let Some(parts) = message.get_mut("content").and_then(Value::as_array_mut) {
                        for part in parts {
                            if part["type"] == "video_url" {
                                let url = part["video_url"]["url"].take();
                                *part = json!({"type": "input_video", "input_video": {"url": url}});
                            }
                        }
                    }
                }
            }
            builder.json(&body)
        } else {
            builder.json(&request)
        };
        let response = builder.send().await?;

        self.parse_response(Self::read_response_json(response).await?)
    }

    async fn read_response_json(mut response: reqwest::Response) -> Result<Value> {
        let status = response.status();
        if !status.is_success() {
            // Read only a bounded prefix, including for non-JSON gateway errors.
            // One extra byte distinguishes a full body from a truncated one.
            let mut body = Vec::new();
            while let Some(chunk) = response.chunk().await? {
                let remaining = MAX_ERROR_BODY_BYTES + 1 - body.len();
                body.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
                if body.len() > MAX_ERROR_BODY_BYTES {
                    break;
                }
            }
            let truncated = body.len() > MAX_ERROR_BODY_BYTES;
            body.truncate(MAX_ERROR_BODY_BYTES);
            // Avoid replacing an otherwise valid character split at the byte limit.
            if truncated
                && let Err(error) = std::str::from_utf8(&body)
                && error.error_len().is_none()
            {
                body.truncate(error.valid_up_to());
            }
            let mut body = String::from_utf8_lossy(&body).into_owned();
            if truncated {
                body.push_str("\n[Provider error body truncated at 16 KiB]");
            }
            return Err(Error::Provider {
                status: status.as_u16(),
                body,
            });
        }

        if response
            .content_length()
            .is_some_and(|length| length > MAX_RESPONSE_BYTES as u64)
        {
            return Err(Error::ProviderResponseTooLarge {
                limit: MAX_RESPONSE_BYTES,
            });
        }

        let mut body = Vec::new();
        while let Some(chunk) = response.chunk().await? {
            if body.len().saturating_add(chunk.len()) > MAX_RESPONSE_BYTES {
                return Err(Error::ProviderResponseTooLarge {
                    limit: MAX_RESPONSE_BYTES,
                });
            }
            body.extend_from_slice(&chunk);
        }
        Ok(serde_json::from_slice::<Value>(&body)?)
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

// Keep the common request and session representation intact; adapt only the wire body.
fn deepseek_request_body(request: &ChatRequest) -> Result<Value> {
    let mut body = serde_json::to_value(request)?;
    body.as_object_mut().unwrap().remove("reasoning");
    if let Some(reasoning) = &request.reasoning {
        if reasoning.exclude == Some(true) || reasoning.max_tokens.is_some() {
            return Err(Error::InvalidProviderConfig(
                "DeepSeek does not support reasoning.exclude=true or reasoning.max_tokens".into(),
            ));
        }
        if let (Some(enabled), Some(effort)) = (reasoning.enabled, &reasoning.effort)
            && enabled == (effort == "none")
        {
            return Err(Error::InvalidProviderConfig(
                "DeepSeek thinking toggle conflicts with reasoning effort".into(),
            ));
        }
        if let Some(enabled) = reasoning.enabled {
            body["thinking"] = json!({"type": if enabled { "enabled" } else { "disabled" }});
        }
        if let Some(effort) = &reasoning.effort {
            body["reasoning_effort"] = json!(effort);
        }
    }
    for message in body["messages"].as_array_mut().unwrap() {
        if let Some(parts) = message["content"].as_array_mut() {
            for part in parts {
                // DeepSeek file parts have flat fields; our canonical format nests them.
                if part["type"] == "file" {
                    let mut file = part["file"].take();
                    file["type"] = json!("file");
                    *part = file;
                }
            }
        }
    }
    Ok(body)
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
    async fn deepseek_request_uses_keyring_auth_and_preserves_reasoning() {
        let mut config =
            ProviderConfig::deepseek_with_keyring("deepseek-flash", "test/deepseek").unwrap();
        assert_eq!(config.kind, ProviderKind::DeepSeek);
        assert_eq!(
            config.chat_completions_url.as_str(),
            "https://api.deepseek.com/chat/completions"
        );
        let server = MockServer::start();
        config.chat_completions_url = server.url("/chat/completions").parse().unwrap();
        config.reasoning = Some(ReasoningConfig {
            enabled: Some(true),
            effort: Some("high".into()),
            ..Default::default()
        });
        let keyring: Arc<CredentialStore> = keyring_core::mock::Store::new().unwrap();
        let secrets = Arc::new(KeyringCoreSecretStore::with_store("smolgent-test", keyring));
        secrets.set_api_key("test/deepseek", "sk-test").unwrap();
        let provider = ChatProvider::new(config).with_secrets(secrets);
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/chat/completions")
                .header("authorization", "Bearer sk-test")
                .json_body(json!({
                    "model": "deepseek-flash",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "thinking": {"type": "enabled"}, "reasoning_effort": "high"
                }));
            then.status(200).json_body(json!({"choices": [{"message": {
                "role": "assistant", "content": "Hi", "reasoning_content": "Thought",
                "tool_calls": null
            }}]}));
        });
        let response = provider
            .send_messages(&[ChatMessage::user("Hello")])
            .await
            .unwrap();
        assert_eq!(response.message.content, "Hi");
        assert_eq!(
            response.reasoning.unwrap().reasoning_content.as_deref(),
            Some("Thought")
        );
        assert!(provider.model_capabilities().await.unwrap().is_none());
        mock.assert();
    }

    #[test]
    fn deepseek_thinking_controls_and_unsupported_options() {
        let provider = ChatProvider::new(ProviderConfig::deepseek("any-model").unwrap());
        let mut request = provider.build_request(&[ChatMessage::user("Hi")], vec![]);
        let body = deepseek_request_body(&request).unwrap();
        assert!(body.get("reasoning").is_none());
        assert!(body.get("thinking").is_none());
        assert!(body.get("reasoning_effort").is_none());
        request.reasoning = Some(ReasoningConfig {
            enabled: Some(false),
            ..Default::default()
        });
        assert_eq!(
            deepseek_request_body(&request).unwrap()["thinking"],
            json!({"type": "disabled"})
        );
        request.reasoning = Some(ReasoningConfig {
            effort: Some("none".into()),
            ..Default::default()
        });
        assert_eq!(
            deepseek_request_body(&request).unwrap()["reasoning_effort"],
            "none"
        );
        for reasoning in [
            ReasoningConfig {
                exclude: Some(true),
                ..Default::default()
            },
            ReasoningConfig {
                max_tokens: Some(100),
                ..Default::default()
            },
            ReasoningConfig {
                enabled: Some(true),
                effort: Some("none".into()),
                ..Default::default()
            },
            ReasoningConfig {
                enabled: Some(false),
                effort: Some("high".into()),
                ..Default::default()
            },
        ] {
            request.reasoning = Some(reasoning);
            assert!(matches!(
                deepseek_request_body(&request),
                Err(Error::InvalidProviderConfig(_))
            ));
        }
    }

    #[test]
    fn literal_api_keys_are_redacted_from_debug_output() {
        let api_key = ApiKeyRef::Literal("sk-secret".to_string());

        let debug = format!("{api_key:?}");

        assert_eq!(debug, "Literal(\"[REDACTED]\")");
        assert!(!debug.contains("sk-secret"));
    }

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
    async fn provider_errors_preserve_status_and_json_or_plain_text_bodies() {
        for (status, body) in [
            (400, r#"{"error":{"message":"Invalid model","code":400}}"#),
            (429, r#"{"error":{"message":"Rate limited"}}"#),
            (502, "upstream unavailable"),
            (503, ""),
        ] {
            let server = MockServer::start();
            let mock = server.mock(|when, then| {
                when.method(POST).path("/v1/chat/completions");
                then.status(status).body(body);
            });
            let provider =
                ChatProvider::new(ProviderConfig::llama_cpp(server.base_url(), "model").unwrap());
            let error = provider.send_messages(&[]).await.unwrap_err();
            assert!(error.to_string().contains(&status.to_string()));
            assert!(
                matches!(error, Error::Provider { status: actual, body: actual_body }
                if actual == status && actual_body == body)
            );
            mock.assert();
        }
    }

    #[tokio::test]
    async fn provider_error_bodies_are_bounded_without_splitting_utf8() {
        for body in [
            "x".repeat(MAX_ERROR_BODY_BYTES),
            format!(
                "{}é{}",
                "x".repeat(MAX_ERROR_BODY_BYTES - 1),
                "z".repeat(100_000)
            ),
        ] {
            let server = MockServer::start();
            let mock = server.mock(|when, then| {
                when.method(POST);
                then.status(500).body(&body);
            });
            let provider =
                ChatProvider::new(ProviderConfig::llama_cpp(server.base_url(), "model").unwrap());
            let Error::Provider {
                status,
                body: actual,
            } = provider.send_messages(&[]).await.unwrap_err()
            else {
                panic!("expected provider error");
            };
            assert_eq!(status, 500);
            if body.len() == MAX_ERROR_BODY_BYTES {
                assert_eq!(actual, body);
            } else {
                assert_eq!(
                    actual,
                    format!(
                        "{}\n[Provider error body truncated at 16 KiB]",
                        "x".repeat(MAX_ERROR_BODY_BYTES - 1)
                    )
                );
            }
            mock.assert();
        }
    }

    #[tokio::test]
    async fn provider_rejects_oversized_responses_before_buffering_them() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(POST).path("/v1/chat/completions");
            then.status(200).body("x".repeat(MAX_RESPONSE_BYTES + 1));
        });
        let provider =
            ChatProvider::new(ProviderConfig::llama_cpp(server.base_url(), "local-model").unwrap());

        let error = provider.send_messages(&[]).await.unwrap_err();

        mock.assert();
        assert!(matches!(
            error,
            Error::ProviderResponseTooLarge {
                limit: MAX_RESPONSE_BYTES
            }
        ));
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
