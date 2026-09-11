use httpmock::{Method::POST, MockServer};
use serde_json::json;
use smolgent::{
    ApiKeyRef, ChatMessage, ChatProvider, ChatSession, CompactionConfig, ContentPart, ImageDetail,
    ImageUrl, MessageContent, ProviderConfig, SessionConfig, TelemetryConfig, TelemetryEvent,
    ToolRegistry,
};

#[test]
fn content_serializes_all_modalities_in_order_and_round_trips() {
    let message = ChatMessage::user(vec![
        ContentPart::text("Compare these"),
        ContentPart::image_url("https://example.com/image.png"),
        ContentPart::text("and these"),
        ContentPart::image_bytes("image/png", b"hello"),
        ContentPart::file("remote.pdf", "https://example.com/document.pdf"),
        ContentPart::pdf_bytes("local.pdf", b"hello"),
        ContentPart::audio_bytes("wav", b"hello"),
        ContentPart::video_url("https://example.com/video.mp4"),
        ContentPart::video_bytes("video/mp4", b"hello"),
    ]);
    let expected = json!({"role": "user", "content": [
        {"type": "text", "text": "Compare these"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        {"type": "text", "text": "and these"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
        {"type": "file", "file": {"filename": "remote.pdf", "file_data": "https://example.com/document.pdf"}},
        {"type": "file", "file": {"filename": "local.pdf", "file_data": "data:application/pdf;base64,aGVsbG8="}},
        {"type": "input_audio", "input_audio": {"data": "aGVsbG8=", "format": "wav"}},
        {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}},
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,aGVsbG8="}}
    ]});
    assert_eq!(serde_json::to_value(&message).unwrap(), expected);
    assert_eq!(
        serde_json::from_value::<ChatMessage>(expected).unwrap(),
        message
    );
    assert_eq!(message.content.text(), "Compare theseand these");
}

#[test]
fn plain_text_retains_its_wire_format_and_empty_media_is_not_empty_text() {
    let message = ChatMessage::user("hello");
    let value = json!({"role": "user", "content": "hello"});
    assert_eq!(serde_json::to_value(&message).unwrap(), value);
    assert_eq!(
        serde_json::from_value::<ChatMessage>(value).unwrap(),
        message
    );
    assert!(MessageContent::default().is_empty());
    assert!(MessageContent::from(vec![]).is_empty());
    assert!(!MessageContent::from(vec![ContentPart::image_url("image.png")]).is_empty());
    // Comparing to a string must never hide attached media.
    assert_ne!(
        ChatMessage::user(vec![ContentPart::text("hello")]).content,
        "hello"
    );
}

#[test]
fn image_detail_and_preencoded_audio_are_supported() {
    let parts = vec![
        ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: "https://example.com/image.png".into(),
                detail: Some(ImageDetail::High),
            },
        },
        ContentPart::input_audio("aGVsbG8=", "mp3"),
    ];
    assert_eq!(
        serde_json::to_value(parts).unwrap(),
        json!([
            {"type": "image_url", "image_url": {"url": "https://example.com/image.png", "detail": "high"}},
            {"type": "input_audio", "input_audio": {"data": "aGVsbG8=", "format": "mp3"}}
        ])
    );
}

#[tokio::test]
async fn openrouter_sends_mixed_inputs_and_replays_pdf_annotations_and_reasoning() {
    let server = MockServer::start();
    let inputs = json!([
        {"type": "text", "text": "Describe these"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        {"type": "file", "file": {"filename": "doc.pdf", "file_data": "data:application/pdf;base64,aGVsbG8="}},
        {"type": "input_audio", "input_audio": {"data": "aGVsbG8=", "format": "wav"}},
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,aGVsbG8="}}
    ]);
    let annotations = json!([{"type": "file", "file": {
        "hash": "parsed-pdf-hash", "name": "doc.pdf",
        "content": [{"type": "text", "text": "Parsed document"}]
    }}]);
    let assistant = json!({
        "role": "assistant", "content": [{"type": "text", "text": "Description"}],
        "annotations": annotations,
        "reasoning_details": [{"type": "reasoning.encrypted", "data": "opaque-state"}]
    });
    let first = server.mock(|when, then| {
        when.method(POST)
            .path("/api/v1/chat/completions")
            .header("authorization", "Bearer sk-test")
            .json_body(json!({
                "model": "multimodal-model", "messages": [{"role": "user", "content": inputs}],
                "reasoning": {"enabled": true, "exclude": false}
            }));
        then.status(200)
            .json_body(json!({"choices": [{"message": assistant}]}));
    });
    let followup = server.mock(|when, then| {
        when.method(POST)
            .path("/api/v1/chat/completions")
            .json_body(json!({
                "model": "multimodal-model", "messages": [
                    {"role": "user", "content": inputs}, assistant,
                    {"role": "user", "content": "Explain more"}
                ], "reasoning": {"enabled": true, "exclude": false}
            }));
        then.status(200)
            .json_body(json!({"choices": [{"message": {"content": "More detail"}}]}));
    });
    let mut config = ProviderConfig::openrouter("multimodal-model").unwrap();
    config.chat_completions_url = format!("{}/api/v1/chat/completions", server.base_url())
        .parse()
        .unwrap();
    config.api_key = ApiKeyRef::Literal("sk-test".into());
    let provider = ChatProvider::new(config);
    let mut session = ChatSession::new();
    let response = session
        .send_user_message(
            &provider,
            vec![
                ContentPart::text("Describe these"),
                ContentPart::image_url("https://example.com/image.png"),
                ContentPart::pdf_bytes("doc.pdf", b"hello"),
                ContentPart::audio_bytes("wav", b"hello"),
                ContentPart::video_bytes("video/mp4", b"hello"),
            ],
        )
        .await
        .unwrap();
    assert_eq!(response.message.content.text(), "Description");
    assert_eq!(serde_json::to_value(&response.message).unwrap(), assistant);
    assert!(response.reasoning.is_some());
    session
        .send_user_message(&provider, "Explain more")
        .await
        .unwrap();
    first.assert();
    followup.assert();
}

#[tokio::test]
async fn managed_tool_loop_preserves_media_across_tool_continuation() {
    let server = MockServer::start();
    let user = ChatMessage::user(vec![ContentPart::image_url(
        "https://example.com/image.png",
    )]);
    let call = json!({"id": "call_1", "type": "function", "function": {"name": "missing", "arguments": "{}"}});
    let first = server.mock(|when, then| {
        when.method(POST).json_body(json!({
            "model": "multimodal-model", "messages": [user],
            "reasoning": {"enabled": true, "exclude": false}
        }));
        then.status(200)
            .json_body(json!({"choices": [{"message": {"content": null, "tool_calls": [call]}}]}));
    });
    let second = server.mock(|when, then| {
        when.method(POST).json_body(json!({
            "model": "multimodal-model", "messages": [user,
                {"role": "assistant", "content": "", "tool_calls": [call]},
                {"role": "tool", "tool_call_id": "call_1", "name": "missing",
                 "content": "Tool error: unknown tool 'missing'\nPlease correct the tool call arguments and try again."}
            ], "reasoning": {"enabled": true, "exclude": false}
        }));
        then.status(200).json_body(json!({"choices": [{"message": {"content": "Done"}}]}));
    });
    let mut config = ProviderConfig::openrouter("multimodal-model").unwrap();
    config.chat_completions_url = server.url("/api/v1/chat/completions").parse().unwrap();
    config.api_key = ApiKeyRef::None;
    let provider = ChatProvider::new(config);
    let (mut session, _) = ChatSession::with_config(SessionConfig {
        compaction: CompactionConfig::disabled(),
        ..SessionConfig::default()
    });
    session
        .run_user_message_with_tools(&provider, &ToolRegistry::new(), user.content.clone())
        .await
        .unwrap();
    assert_eq!(session.messages()[0], user);
    first.assert();
    second.assert();
}

#[test]
fn media_does_not_leak_into_text_telemetry_or_count_as_text_tokens() {
    let (mut session, _, telemetry) = ChatSession::with_config_and_telemetry(SessionConfig {
        telemetry: TelemetryConfig::messages(),
        ..SessionConfig::default()
    });
    session.push_user(vec![
        ContentPart::text("Look"),
        ContentPart::image_bytes("image/png", &vec![42; 100_000]),
    ]);
    let usage = session.context_usage_breakdown();
    assert!(usage.total_bytes > 100_000);
    assert_eq!(usage.total_estimated_tokens, 1);
    assert_eq!(usage.largest_turns[0].summary, "Look");
    assert!(matches!(telemetry.unwrap().try_recv().unwrap(),
        TelemetryEvent::UserMessage { content: Some(text), content_len, .. }
        if text == "Look" && content_len > 100_000));
}
