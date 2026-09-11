use httpmock::{
    Method::{GET, POST},
    MockServer,
};
use serde_json::json;
use smolgent::{
    AgentState, ApiKeyRef, ChatMessage, ChatProvider, ChatSession, CompactionConfig, ContentPart,
    Error, MessageContent, ModelCapabilities, ProviderConfig, READ_MAX_MEDIA_BYTES, SessionConfig,
    Tool, ToolCall, ToolCallFunction, ToolDefinition, ToolRegistry, builtin_registry_for_provider,
    read_tool_with_capabilities,
};

fn openrouter(server: &MockServer) -> ChatProvider {
    let mut config = ProviderConfig::openrouter("z-ai/glm-5.3-flash").unwrap();
    config.chat_completions_url = server.url("/api/v1/chat/completions").parse().unwrap();
    config.api_key = ApiKeyRef::Literal("test-key".into());
    ChatProvider::new(config)
}

fn capabilities(modalities: &[&str]) -> ModelCapabilities {
    ModelCapabilities {
        model: "test-model".into(),
        input_modalities: modalities.iter().map(|s| (*s).into()).collect(),
    }
}

#[tokio::test]
async fn discovery_uses_input_modalities_and_only_augments_supported_formats() {
    let server = MockServer::start();
    let metadata = server.mock(|when, then| {
        when.method(GET)
            .path("/api/v1/model/z-ai/glm-5.3-flash")
            .header("authorization", "Bearer test-key");
        then.status(200).json_body(json!({"data": {"architecture": {
            "input_modalities": ["text", "image", "video"], "output_modalities": ["text", "audio"]
        }}}));
    });
    let registry = builtin_registry_for_provider(AgentState::default(), &openrouter(&server))
        .await
        .unwrap();
    let description = &registry
        .get("read")
        .unwrap()
        .definition()
        .function
        .description;
    assert!(description.contains("images:"));
    assert!(description.contains("videos:"));
    assert!(!description.contains("audio:"));
    assert!(!description.contains("PDF documents:"));
    metadata.assert();
}

#[tokio::test]
async fn unknown_or_missing_metadata_and_other_providers_keep_text_reader() {
    for body in [
        json!({"data": {}}),
        json!({"data": {"architecture": {"input_modalities": ["text"]}}}),
    ] {
        let server = MockServer::start();
        let metadata = server.mock(|when, then| {
            when.method(GET);
            then.status(200).json_body(body);
        });
        let registry = builtin_registry_for_provider(AgentState::default(), &openrouter(&server))
            .await
            .unwrap();
        assert_eq!(
            registry.get("read").unwrap().definition(),
            smolgent::read_tool(AgentState::default()).definition()
        );
        metadata.assert();
    }
    let server = MockServer::start();
    let not_found = server.mock(|when, then| {
        when.method(GET);
        then.status(404);
    });
    assert!(
        openrouter(&server)
            .model_capabilities()
            .await
            .unwrap()
            .is_none()
    );
    not_found.assert();
    let local = ChatProvider::new(ProviderConfig::llama_cpp(server.base_url(), "model").unwrap());
    assert!(local.model_capabilities().await.unwrap().is_none());
    let mut config = openrouter(&server).config().clone();
    config.default_model = "openrouter/auto".into();
    assert!(
        ChatProvider::new(config)
            .model_capabilities()
            .await
            .unwrap()
            .is_none()
    );
    not_found.assert_hits(1);
}

#[tokio::test]
async fn metadata_http_failures_remain_visible_to_the_caller() {
    let server = MockServer::start();
    let metadata = server.mock(|when, then| {
        when.method(GET);
        then.status(401).body("invalid API key");
    });
    let error =
        match builtin_registry_for_provider(AgentState::default(), &openrouter(&server)).await {
            Ok(_) => panic!("expected an authentication error"),
            Err(error) => error,
        };
    assert!(matches!(error, Error::Provider { status: 401, body } if body == "invalid API key"));
    metadata.assert();
}

#[tokio::test]
async fn media_reader_preserves_parts_for_each_supported_modality() {
    let temp = tempfile::tempdir().unwrap();
    let reader = read_tool_with_capabilities(
        AgentState::new([temp.path().into()], []),
        capabilities(&["image", "video", "audio", "file"]),
    );
    for (name, expected) in [
        ("image.PNG", ContentPart::image_bytes("image/png", b"hello")),
        ("clip.mp4", ContentPart::video_bytes("video/mp4", b"hello")),
        ("sound.wav", ContentPart::audio_bytes("wav", b"hello")),
        (
            "document.pdf",
            ContentPart::pdf_bytes("document.pdf", b"hello"),
        ),
    ] {
        let path = temp.path().join(name);
        std::fs::write(&path, b"hello").unwrap();
        let MessageContent::Parts(parts) =
            reader.call_content(json!({"path": path})).await.unwrap()
        else {
            panic!("expected content parts");
        };
        assert!(matches!(&parts[0], ContentPart::Text { text } if text.contains(name)));
        assert_eq!(parts[1], expected);
    }
    let text = temp.path().join("notes.txt");
    std::fs::write(&text, "first\nsecond\nthird\n").unwrap();
    let output = reader
        .call_content(json!({"path": text, "line-offset": 1, "line-count": 1}))
        .await
        .unwrap();
    assert!(matches!(&output, MessageContent::Text(_)));
    assert!(output.text().starts_with("second\n"));
    assert!(output.text().contains("Read truncated"));
}

#[tokio::test]
async fn media_reader_enforces_roots_capabilities_pagination_and_size() {
    let temp = tempfile::tempdir().unwrap();
    let denied = tempfile::tempdir().unwrap();
    let allowed_image = temp.path().join("image.png");
    let denied_image = denied.path().join("image.png");
    let video = temp.path().join("clip.mp4");
    for path in [&allowed_image, &denied_image, &video] {
        std::fs::write(path, b"hello").unwrap();
    }
    let reader = read_tool_with_capabilities(
        AgentState::new([temp.path().into()], []),
        capabilities(&["image"]),
    );
    assert!(matches!(
        reader
            .call_content(json!({"path": denied_image}))
            .await
            .unwrap_err(),
        Error::PathNotAllowed { .. }
    ));
    assert!(
        reader
            .call_content(json!({"path": video}))
            .await
            .unwrap_err()
            .to_string()
            .contains("does not advertise video")
    );
    assert!(
        reader
            .call_content(json!({"path": allowed_image, "line-offset": 0}))
            .await
            .unwrap_err()
            .to_string()
            .contains("text offsets/counts")
    );
    std::fs::File::create(&allowed_image)
        .unwrap()
        .set_len(READ_MAX_MEDIA_BYTES as u64 + 1)
        .unwrap();
    assert!(
        reader
            .call_content(json!({"path": allowed_image}))
            .await
            .unwrap_err()
            .to_string()
            .contains("20 MiB")
    );
}

#[tokio::test]
async fn ordinary_json_array_tool_results_are_not_interpreted_as_media() {
    let registry = ToolRegistry::new().with_tool(Tool::new(
        ToolDefinition::new("json", "Return ordinary JSON", json!({"type": "object"})),
        |_| {
            Box::pin(async {
                Ok(json!([{"type": "image_url", "image_url": {"url": "text-only"}}]))
            })
        },
    ));
    let result = registry
        .execute_call(&ToolCall {
            id: "call".into(),
            kind: "function".into(),
            function: ToolCallFunction {
                name: "json".into(),
                arguments: "{}".into(),
            },
        })
        .await
        .unwrap();
    assert!(matches!(result.content, MessageContent::Text(_)));
}

#[tokio::test]
async fn discovered_reader_attaches_images_and_video_in_the_managed_tool_loop() {
    let temp = tempfile::tempdir().unwrap();
    let image = temp.path().join("image.png");
    let video = temp.path().join("video.mp4");
    std::fs::write(&image, b"hello").unwrap();
    std::fs::write(&video, b"hello").unwrap();
    let server = MockServer::start();
    let metadata = server.mock(|when, then| {
        when.method(GET);
        then.status(200).json_body(
            json!({"data": {"architecture": {"input_modalities": ["text", "image", "video"]}}}),
        );
    });
    let first = server.mock(|when, then| {
        when.method(POST).body_contains("images:").body_contains("videos:")
            .matches(|request| {
                let body: serde_json::Value = serde_json::from_slice(request.body.as_deref().unwrap()).unwrap();
                body["messages"].as_array().unwrap().len() == 1
            });
        then.status(200).json_body(json!({"choices": [{"message": {"content": null, "tool_calls": [
            {"id": "read_image", "type": "function", "function": {"name": "read", "arguments": json!({"path": image}).to_string()}},
            {"id": "read_video", "type": "function", "function": {"name": "read", "arguments": json!({"path": video}).to_string()}}
        ]}}]}));
    });
    let final_reply = server.mock(|when, then| {
        when.method(POST).matches(|request| {
            let body: serde_json::Value = serde_json::from_slice(request.body.as_deref().unwrap()).unwrap();
            let messages = &body["messages"];
            messages[2]["role"] == "tool" && messages[2]["tool_call_id"] == "read_image"
                && messages[2]["content"][1] == json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}})
                && messages[3]["role"] == "tool" && messages[3]["tool_call_id"] == "read_video"
                && messages[3]["content"][1] == json!({"type": "video_url", "video_url": {"url": "data:video/mp4;base64,aGVsbG8="}})
        });
        then.status(200).json_body(json!({"choices": [{"message": {"content": "Comparison"}}]}));
    });
    let provider = openrouter(&server);
    let registry =
        builtin_registry_for_provider(AgentState::new([temp.path().into()], []), &provider)
            .await
            .unwrap();
    let (mut session, _) = ChatSession::with_config(SessionConfig {
        compaction: CompactionConfig::disabled(),
        ..SessionConfig::default()
    });
    let response = session
        .run_user_message_with_tools(&provider, &registry, "Compare the image and video")
        .await
        .unwrap();
    assert_eq!(response.message, ChatMessage::assistant("Comparison"));
    assert!(session.messages()[2].content.has_media());
    assert!(session.messages()[3].content.has_media());
    first.assert();
    final_reply.assert();
    metadata.assert_hits(1);
}
