use httpmock::{
    Method::{GET, POST},
    MockServer,
};
use serde_json::{Value, json};
use smolgent::{
    AgentState, ApiKeyRef, ChatMessage, ChatProvider, ChatSession, CompactionConfig, ContentPart,
    Error, ProviderConfig, SessionConfig, builtin_registry_for_provider,
};

fn provider(server: &MockServer) -> ChatProvider {
    let mut config = ProviderConfig::llama_cpp(server.url("/proxy/"), "org/model:variant").unwrap();
    config.api_key = ApiKeyRef::Literal("local-key".into());
    config.headers.push(("x-test".into(), "custom".into()));
    ChatProvider::new(config)
}

#[tokio::test]
async fn discovery_targets_selected_model_and_maps_only_explicit_modalities() {
    for (modalities, expected) in [
        (json!({"vision": true}), vec!["text", "image"]),
        (
            json!({"vision": false, "audio": true, "video": false}),
            vec!["text", "audio"],
        ),
        (
            json!({"vision": true, "audio": true, "video": true}),
            vec!["text", "image", "audio", "video"],
        ),
        (json!({"vision": false}), vec!["text"]),
    ] {
        let server = MockServer::start();
        let metadata = server.mock(|when, then| {
            when.method(GET)
                .path("/proxy/props")
                .query_param("model", "org/model:variant")
                .header("authorization", "Bearer local-key")
                .header("x-test", "custom");
            then.status(200)
                .json_body(json!({"modalities": modalities}));
        });
        let caps = provider(&server)
            .model_capabilities()
            .await
            .unwrap()
            .unwrap();
        assert_eq!(caps.model, "org/model:variant");
        assert_eq!(caps.input_modalities, expected);
        metadata.assert();
    }
    for base in [
        "http://localhost:8080",
        "http://localhost:8080/",
        "http://localhost:8080/proxy",
        "http://localhost:8080/proxy/",
    ] {
        let config = ProviderConfig::llama_cpp(base, "local").unwrap();
        let prefix = if base.contains("proxy") { "/proxy" } else { "" };
        assert_eq!(
            config.chat_completions_url.path(),
            format!("{prefix}/v1/chat/completions")
        );
    }
}

#[tokio::test]
async fn discovery_falls_back_for_old_servers_and_surfaces_bad_metadata_and_http_errors() {
    for (status, body, error) in [
        (404, json!({}), None),
        (200, json!({}), None),
        (200, json!({"modalities": null}), None),
        (200, json!({"modalities": {"vision": "yes"}}), Some("json")),
        (401, json!({"error": "key required"}), Some("http")),
    ] {
        let server = MockServer::start();
        let metadata = server.mock(|when, then| {
            when.method(GET).path("/proxy/props");
            then.status(status).json_body(body);
        });
        let result = provider(&server).model_capabilities().await;
        match error {
            None => assert!(result.unwrap().is_none()),
            Some("json") => assert!(matches!(result, Err(Error::Json(_)))),
            _ => assert!(
                matches!(result, Err(Error::Provider { status: 401, body }) if body.contains("key required"))
            ),
        }
        metadata.assert();
    }
}

#[tokio::test]
async fn discovered_reader_uses_llama_cpp_decoders_and_retains_text_pagination() {
    let server = MockServer::start();
    let metadata = server.mock(|when, then| {
        when.method(GET);
        then.status(200)
            .json_body(json!({"modalities": {"vision": true, "audio": true}}));
    });
    let temp = tempfile::tempdir().unwrap();
    let registry = builtin_registry_for_provider(
        AgentState::new([temp.path().into()], []),
        &provider(&server),
    )
    .await
    .unwrap();
    let reader = registry.get("read").unwrap();
    let description = &reader.definition().function.description;
    assert!(description.contains(".bmp"));
    assert!(description.contains(".flac"));
    for absent in [".webp", ".aac", "videos:", "PDF documents:"] {
        assert!(!description.contains(absent));
    }
    for (name, part) in [
        ("image.bmp", ContentPart::image_bytes("image/bmp", b"hello")),
        ("image.PNG", ContentPart::image_bytes("image/png", b"hello")),
        ("audio.flac", ContentPart::audio_bytes("flac", b"hello")),
    ] {
        let path = temp.path().join(name);
        std::fs::write(&path, b"hello").unwrap();
        let result = reader.call_content(json!({"path": path})).await.unwrap();
        assert_eq!(
            serde_json::to_value(result).unwrap()[1],
            serde_json::to_value(part).unwrap()
        );
    }
    for (name, expected) in [
        ("image.webp", "decoders do not support"),
        ("audio.aac", "decoders do not support"),
        ("video.mp4", "does not advertise video"),
        ("file.pdf", "does not advertise file"),
    ] {
        let path = temp.path().join(name);
        std::fs::write(&path, b"hello").unwrap();
        let error = reader
            .call_content(json!({"path": path}))
            .await
            .unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
    let path = temp.path().join("notes.txt");
    std::fs::write(&path, "first\nsecond\nthird\n").unwrap();
    let result = reader
        .call_content(json!({"path": path, "line-offset": 1, "line-count": 1}))
        .await
        .unwrap();
    assert!(result.text().starts_with("second\n"));
    metadata.assert_hits(1);
}

#[tokio::test]
async fn direct_requests_translate_video_and_preserve_image_audio_and_order() {
    let server = MockServer::start();
    let message = ChatMessage::user(vec![
        ContentPart::text("Describe these"),
        ContentPart::image_bytes("image/png", b"hello"),
        ContentPart::audio_bytes("wav", b"hello"),
        ContentPart::video_bytes("video/mp4", b"hello"),
        ContentPart::video_url("https://example.com/clip.mp4"),
    ]);
    let completion = server.mock(|when, then| {
        when.method(POST).path("/proxy/v1/chat/completions")
            .header("authorization", "Bearer local-key")
            .json_body_partial(json!({"model": "org/model:variant", "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Describe these"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
                {"type": "input_audio", "input_audio": {"data": "aGVsbG8=", "format": "wav"}},
                {"type": "input_video", "input_video": {"url": "data:video/mp4;base64,aGVsbG8="}},
                {"type": "input_video", "input_video": {"url": "https://example.com/clip.mp4"}}
            ]}]}).to_string());
        then.status(200).json_body(json!({"choices": [{"message": {"content": "Description"}}]}));
    });
    let provider = provider(&server);
    // Cover pre-built requests as well as the higher-level send_messages path.
    let request = provider.build_request(&[message.clone()], Vec::new());
    assert_eq!(request.messages[0], message);
    let reply = provider.send_request(request).await.unwrap();
    assert_eq!(reply.message.content, "Description");
    completion.assert();
}

#[tokio::test]
async fn managed_tool_loop_translates_media_results_without_changing_history() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("video.mp4");
    std::fs::write(&path, b"hello").unwrap();
    let server = MockServer::start();
    let metadata = server.mock(|when, then| {
        when.method(GET);
        then.status(200)
            .json_body(json!({"modalities": {"vision": true, "video": true}}));
    });
    let first = server.mock(|when, then| {
        when.method(POST).body_contains("videos:").matches(|request| {
            let body: Value = serde_json::from_slice(request.body.as_deref().unwrap()).unwrap();
            body["messages"].as_array().unwrap().len() == 1
        });
        then.status(200).json_body(json!({"choices": [{"message": {"content": null, "tool_calls": [
            {"id": "read_video", "type": "function", "function": {"name": "read", "arguments": json!({"path": path}).to_string()}}
        ]}}]}));
    });
    let last = server.mock(|when, then| {
        when.method(POST).matches(|request| {
            let body: Value = serde_json::from_slice(request.body.as_deref().unwrap()).unwrap();
            let result = &body["messages"][2];
            result["role"] == "tool" && result["tool_call_id"] == "read_video"
                && result["content"][1] == json!({"type": "input_video", "input_video": {"url": "data:video/mp4;base64,aGVsbG8="}})
        });
        then.status(200).json_body(json!({"choices": [{"message": {"content": "Video summary"}}]}));
    });
    let provider = provider(&server);
    let registry =
        builtin_registry_for_provider(AgentState::new([temp.path().into()], []), &provider)
            .await
            .unwrap();
    let (mut session, _) = ChatSession::with_config(SessionConfig {
        compaction: CompactionConfig::disabled(),
        ..SessionConfig::default()
    });
    session
        .run_user_message_with_tools(&provider, &registry, "Read the video")
        .await
        .unwrap();
    let history = serde_json::to_value(session.messages()).unwrap();
    assert_eq!(history[2]["content"][1]["type"], "video_url");
    assert_eq!(history[2]["tool_call_id"], "read_video");
    first.assert();
    last.assert();
    metadata.assert_hits(1);
}
