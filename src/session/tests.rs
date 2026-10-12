use std::sync::{Arc, Mutex};
use std::time::Duration;

use httpmock::Method::POST;
use httpmock::MockServer;
use serde_json::json;

use super::*;
use crate::chat::ToolCallFunction;
use crate::{ChatProvider, ProviderConfig, Tool, ToolDefinition};

#[test]
fn compaction_keeps_unselected_media_and_annotations_when_removing_tool_calls() {
    let (mut session, _) = ChatSession::with_config(SessionConfig {
        compaction: CompactionConfig {
            protect_recent_turns: 0,
            ..CompactionConfig::default()
        },
        ..SessionConfig::default()
    });
    session.push_system("system");
    let media = ChatMessage::assistant(vec![crate::ContentPart::image_url(
        "https://example.com/image.png",
    )]);
    session.push(media.clone());
    let mut annotated = ChatMessage::assistant("");
    annotated.annotations =
        vec![json!({"type": "file", "file": {"hash": "parsed", "content": []}})];
    session.push(annotated.clone());
    session.push(ChatMessage::assistant("").with_tool_calls(vec![ToolCall {
        id: "call_1".into(),
        kind: "function".into(),
        function: ToolCallFunction {
            name: "read".into(),
            arguments: "{}".into(),
        },
    }]));
    session.push(ChatMessage::tool_result("call_1", "read", "stale contents"));
    session.compact_remove_messages(&[5], "stale read").unwrap();
    assert_eq!(
        session.messages(),
        vec![ChatMessage::system("system"), media, annotated]
    );
}

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

#[test]
fn context_usage_estimates_turns_and_tool_calls() {
    let mut session = ChatSession::new();
    session.push_system("system");
    session.push(ChatMessage::assistant("").with_tool_calls(vec![ToolCall {
        id: "call_1".to_string(),
        kind: "function".to_string(),
        function: ToolCallFunction {
            name: "read".to_string(),
            arguments: r#"{"path":"src/lib.rs"}"#.to_string(),
        },
    }]));
    session.push(ChatMessage::tool_result(
        "call_1",
        "read",
        "large file contents",
    ));

    let breakdown = session.context_usage_breakdown();

    assert!(breakdown.total_estimated_tokens > 0);
    assert_eq!(breakdown.largest_tool_calls.len(), 1);
    assert_eq!(breakdown.largest_tool_calls[0].tool_call_id, "call_1");
    assert_eq!(breakdown.largest_tool_calls[0].result_turn_ids, vec![3]);
}

#[test]
fn compaction_removes_tool_result_and_matching_request() {
    let (mut session, _receiver) = ChatSession::with_config(SessionConfig {
        compaction: CompactionConfig {
            protect_recent_turns: 0,
            ..CompactionConfig::default()
        },
        ..SessionConfig::default()
    });
    session.push_system("system");
    session.push(ChatMessage::assistant("").with_tool_calls(vec![ToolCall {
        id: "call_1".to_string(),
        kind: "function".to_string(),
        function: ToolCallFunction {
            name: "read".to_string(),
            arguments: r#"{"path":"src/lib.rs"}"#.to_string(),
        },
    }]));
    session.push(ChatMessage::tool_result("call_1", "read", "contents"));

    let tool_turn_id = session.turns()[2].id;
    session
        .compact_remove_messages(&[tool_turn_id], "stale read")
        .unwrap();

    let messages = session.messages();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].role, MessageRole::System);
}

#[test]
fn compaction_summarizes_selected_messages() {
    let (mut session, _receiver) = ChatSession::with_config(SessionConfig {
        compaction: CompactionConfig {
            protect_recent_turns: 0,
            ..CompactionConfig::default()
        },
        ..SessionConfig::default()
    });
    session.push_system("system");
    session.push_user("old question");
    session.push_assistant("old answer");

    let old_ids = vec![session.turns()[1].id, session.turns()[2].id];
    session
        .compact_summarize_messages(&old_ids, "The old answer was useful.", "old topic")
        .unwrap();

    assert_eq!(session.turns().len(), 2);
    assert_eq!(session.turns()[0].content, "system");
    assert!(
        session.turns()[1]
            .content
            .text()
            .contains("The old answer was useful.")
    );
}

#[test]
fn explicit_compaction_cleanup_removes_prior_compaction_tool_messages() {
    let mut session = ChatSession::new();
    session.push_system("system");
    session.push(ChatMessage::assistant("").with_tool_calls(vec![ToolCall {
        id: "compact_old".to_string(),
        kind: "function".to_string(),
        function: ToolCallFunction {
            name: COMPACT_REMOVE_MESSAGES.to_string(),
            arguments: r#"{"turn_ids":[2],"reason":"old"}"#.to_string(),
        },
    }]));
    session.push(ChatMessage::tool_result(
        "compact_old",
        COMPACT_REMOVE_MESSAGES,
        "Compaction acknowledged: removed selected context.",
    ));
    session.push_system("Compacted context summary: keep this useful summary");

    let removed = session.remove_previous_compaction_tool_messages();

    assert_eq!(removed, 2);
    assert_eq!(session.turns().len(), 2);
    assert!(
        session
            .turns()
            .iter()
            .any(|turn| turn.content.text().contains("keep this useful summary"))
    );
    assert!(
        session
            .turns()
            .iter()
            .all(|turn| turn.name.as_deref() != Some(COMPACT_REMOVE_MESSAGES))
    );
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
            .text()
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
        stop_after_tools: Vec::new(),
        event_channel_capacity: Some(2),
        notifications: NotificationConfig::tools(),
        compaction: CompactionConfig::default(),
        telemetry: TelemetryConfig::default(),
        telemetry_channel_capacity: None,
    };
    assert_eq!(config.event_channel_capacity(), 2);

    let (_session, receiver) = ChatSession::with_config(SessionConfig {
        notifications: NotificationConfig::none(),
        ..SessionConfig::default()
    });

    assert!(receiver.is_none());
}

#[test]
fn configured_sessions_create_bounded_telemetry_streams() {
    let config = SessionConfig {
        telemetry: TelemetryConfig::messages(),
        telemetry_channel_capacity: Some(3),
        ..SessionConfig::default()
    };
    assert_eq!(config.telemetry_channel_capacity(), 3);

    let (mut session, events, telemetry) = ChatSession::with_config_and_telemetry(config);
    assert!(events.is_none());
    let telemetry = telemetry.unwrap();

    session.push_user("hello");
    session.push_assistant("hi");

    assert_eq!(
        telemetry.recv().unwrap(),
        TelemetryEvent::UserMessage {
            turn_id: 1,
            content_len: 5,
            content: Some("hello".to_string()),
        }
    );
    assert_eq!(
        telemetry.recv().unwrap(),
        TelemetryEvent::AssistantMessage {
            turn_id: 2,
            content_len: 2,
            tool_calls: 0,
            content: Some("hi".to_string()),
        }
    );
}

#[test]
fn tool_telemetry_redacts_payloads_unless_enabled() {
    let (session, _events, telemetry) = ChatSession::with_config_and_telemetry(SessionConfig {
        telemetry: TelemetryConfig::tools(),
        ..SessionConfig::default()
    });
    let telemetry = telemetry.unwrap();
    let call = ToolCall {
        id: "call_1".to_string(),
        kind: "function".to_string(),
        function: ToolCallFunction {
            name: "read".to_string(),
            arguments: r#"{"path":"secret.txt"}"#.to_string(),
        },
    };
    session.emit_tool_call_started_telemetry(&call);

    assert_eq!(
        telemetry.recv().unwrap(),
        TelemetryEvent::ToolCallStarted {
            tool_call_id: "call_1".to_string(),
            name: "read".to_string(),
            arguments_len: r#"{"path":"secret.txt"}"#.len(),
            arguments: None,
        }
    );

    let (session, _events, telemetry) = ChatSession::with_config_and_telemetry(SessionConfig {
        telemetry: TelemetryConfig::all(),
        ..SessionConfig::default()
    });
    let telemetry = telemetry.unwrap();
    session.emit_tool_call_started_telemetry(&call);

    assert_eq!(
        telemetry.recv().unwrap(),
        TelemetryEvent::ToolCallStarted {
            tool_call_id: "call_1".to_string(),
            name: "read".to_string(),
            arguments_len: r#"{"path":"secret.txt"}"#.len(),
            arguments: Some(r#"{"path":"secret.txt"}"#.to_string()),
        }
    );
}

#[tokio::test]
async fn compaction_rounds_rebuild_history_and_instructions_after_edits() {
    for tool_name in [COMPACT_REMOVE_MESSAGES, COMPACT_SUMMARIZE_MESSAGES] {
        let server = MockServer::start();
        let arguments = if tool_name == COMPACT_REMOVE_MESSAGES {
            json!({"turn_ids": [2], "reason": "obsolete"})
        } else {
            json!({"turn_ids": [2], "summary": "Keep this conclusion", "reason": "condensed"})
        };
        let first = server.mock(|when, then| {
            when.method(POST)
                .body_contains("Context compaction is required")
                .body_contains("OLD_PAYLOAD_TO_REMOVE");
            then.status(200).json_body(json!({"choices": [{"message": {
                "content": null, "tool_calls": [{
                    "id": "compact_1", "type": "function",
                    "function": {"name": tool_name, "arguments": arguments.to_string()}
                }]
            }}]}));
        });
        let second = server.mock(|when, then| {
            when.method(POST)
                .body_contains("Context compaction is required")
                .body_contains("Compaction acknowledged:");
            then.status(200)
                .json_body(json!({"choices": [{"message": {"content": "Done"}}]}));
        });
        let provider =
            ChatProvider::new(ProviderConfig::llama_cpp(server.base_url(), "model").unwrap());
        let (mut session, _, telemetry) = ChatSession::with_config_and_telemetry(SessionConfig {
            compaction: CompactionConfig {
                trigger_estimated_tokens: 1,
                target_estimated_tokens: 0,
                max_compaction_rounds: 2,
                protect_recent_turns: 0,
                ..CompactionConfig::default()
            },
            telemetry: TelemetryConfig {
                model_payloads: true,
                ..TelemetryConfig::default()
            },
            ..SessionConfig::default()
        });
        session.push_system("system");
        session.push_user("OLD_PAYLOAD_TO_REMOVE");
        session.push_user("current question");
        session.maybe_compact(&provider).await.unwrap();
        first.assert();
        second.assert();

        let requests = telemetry
            .unwrap()
            .try_iter()
            .filter_map(|event| match event {
                TelemetryEvent::ModelRequest { request, .. } => request,
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(requests.len(), 2);
        let mut messages = requests[1].messages.clone();
        let instruction = messages.pop().unwrap();
        assert_eq!(messages, session.messages_with_turn_ids());
        assert!(messages[0].content.text().starts_with("[turn_id=1] "));
        assert!(
            !serde_json::to_string(&requests[1])
                .unwrap()
                .contains("OLD_PAYLOAD_TO_REMOVE")
        );
        assert_eq!(
            instruction,
            ChatMessage::system(compaction::instruction(
                &session.config.compaction,
                &session
                    .context_usage_breakdown_with_limit(session.config.compaction.top_consumers),
                &compaction::protected_ids(&session.turns, &session.config.compaction),
            ))
        );
        assert_eq!(messages[messages.len() - 2].tool_calls[0].id, "compact_1");
        assert_eq!(
            messages.last().unwrap().tool_call_id.as_deref(),
            Some("compact_1")
        );
        if tool_name == COMPACT_SUMMARIZE_MESSAGES {
            assert!(
                messages
                    .iter()
                    .any(|message| message.content.text().contains("Keep this conclusion"))
            );
        }
    }
}

#[tokio::test]
async fn unsuccessful_compaction_retries_only_on_the_next_agent_run() {
    // Cover both a declined attempt and an attempt that exhausts its tool-round budget.
    for compaction_message in [
        json!({"content": "Nothing to compact"}),
        json!({"content": null, "tool_calls": [{
            "id": "invalid_compaction", "type": "function",
            "function": {"name": COMPACT_REMOVE_MESSAGES, "arguments": "{\"turn_ids\":[999]}"}
        }]}),
    ] {
        let server = MockServer::start();
        let compact = server.mock(|when, then| {
            when.method(POST)
                .body_contains("Context compaction is required");
            then.status(200)
                .json_body(json!({"choices": [{"message": compaction_message}]}));
        });
        let tool = server.mock(|when, then| {
            when.method(POST).matches(|request| {
                let body: serde_json::Value =
                    serde_json::from_slice(request.body.as_deref().unwrap()).unwrap();
                let last = body["messages"].as_array().unwrap().last().unwrap();
                last["role"] == "user"
                    || (last["role"] == "tool" && last["tool_call_id"] == "invalid_compaction")
            });
            then.status(200).json_body(json!({"choices": [{"message": {
                "content": null, "tool_calls": [{
                    "id": "normal_tool", "type": "function",
                    "function": {"name": "missing_tool", "arguments": "{}"}
                }]
            }}]}));
        });
        let answer = server.mock(|when, then| {
            when.method(POST).matches(|request| {
                let body: serde_json::Value =
                    serde_json::from_slice(request.body.as_deref().unwrap()).unwrap();
                let last = body["messages"].as_array().unwrap().last().unwrap();
                last["role"] == "tool" && last["tool_call_id"] == "normal_tool"
            });
            then.status(200)
                .json_body(json!({"choices": [{"message": {"content": "answer"}}]}));
        });
        let provider =
            ChatProvider::new(ProviderConfig::llama_cpp(server.base_url(), "model").unwrap());
        let (mut session, _) = ChatSession::with_config(SessionConfig {
            compaction: CompactionConfig {
                trigger_estimated_tokens: 1,
                target_estimated_tokens: 0,
                max_compaction_rounds: 1,
                ..CompactionConfig::default()
            },
            ..SessionConfig::default()
        });
        for run in 1..=2 {
            let response = session
                .run_user_message_with_tools(&provider, &ToolRegistry::new(), "Please answer")
                .await
                .unwrap();
            assert_eq!(response.message.content, "answer");
            compact.assert_hits(run);
        }
        tool.assert_hits(2);
        answer.assert_hits(2);
    }
}

#[tokio::test]
async fn managed_loop_compacts_before_normal_request() {
    let server = MockServer::start();
    let compaction_mock = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .body_contains("Context compaction is required")
            .body_contains("old huge content")
            .body_contains(COMPACT_REMOVE_MESSAGES);
        then.status(200).json_body(json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "compact_1",
                        "type": "function",
                        "function": {
                            "name": COMPACT_REMOVE_MESSAGES,
                            "arguments": "{\"turn_ids\":[2],\"reason\":\"obsolete\"}"
                        }
                    }]
                }
            }]
        }));
    });
    let answer_mock = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .body_contains("fresh question");
        then.status(200).json_body(json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "fresh answer"
                }
            }]
        }));
    });

    let config = ProviderConfig::llama_cpp(server.base_url(), "local-model").unwrap();
    let provider = ChatProvider::new(config);
    let (mut session, _receiver) = ChatSession::with_system_prompt_and_config(
        "system",
        SessionConfig {
            compaction: CompactionConfig {
                trigger_estimated_tokens: 10,
                target_estimated_tokens: 1_000,
                protect_recent_turns: 0,
                ..CompactionConfig::default()
            },
            ..SessionConfig::default()
        },
    );
    session.push_user("old huge content old huge content old huge content");

    let response = session
        .run_user_message_with_tools(&provider, &ToolRegistry::new(), "fresh question")
        .await
        .unwrap();

    compaction_mock.assert();
    answer_mock.assert();
    assert_eq!(response.message.content, "fresh answer");
    assert!(
        session
            .turns()
            .iter()
            .all(|turn| !turn.content.text().contains("old huge content"))
    );
    assert!(
        session
            .turns()
            .iter()
            .any(|turn| turn.name.as_deref() == Some(COMPACT_REMOVE_MESSAGES)
                && turn.content.text().starts_with("Compaction acknowledged:"))
    );
}

#[test]
fn proactive_compaction_calls_are_persisted_as_acknowledgements() {
    let (mut session, _receiver) = ChatSession::with_system_prompt_and_config(
        "system",
        SessionConfig {
            compaction: CompactionConfig {
                trigger_estimated_tokens: usize::MAX,
                protect_recent_turns: 0,
                ..CompactionConfig::default()
            },
            ..SessionConfig::default()
        },
    );
    session.push_user("old topic");
    session.push_user("new topic");

    let message = ChatMessage::assistant("").with_tool_calls(vec![ToolCall {
        id: "compact_1".to_string(),
        kind: "function".to_string(),
        function: ToolCallFunction {
            name: COMPACT_REMOVE_MESSAGES.to_string(),
            arguments: r#"{"turn_ids":[2],"reason":"stale"}"#.to_string(),
        },
    }]);
    session.execute_compaction_response(message).unwrap();

    let acknowledgement = session
        .turns()
        .iter()
        .find(|turn| turn.name.as_deref() == Some(COMPACT_REMOVE_MESSAGES))
        .unwrap();

    assert!(session.turns().iter().any(|turn| {
        turn.tool_calls
            .iter()
            .any(|call| call.function.name == COMPACT_REMOVE_MESSAGES)
    }));
    assert!(
        acknowledgement
            .content
            .text()
            .starts_with("Compaction acknowledged:")
    );
    assert!(!acknowledgement.content.text().contains("old topic"));
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
    assert!(
        events.iter().any(
            |event| matches!(event, AgentEvent::ToolCallStarted { name, arguments: None, .. } if name == "add")
        )
    );
    assert!(events
        .iter()
        .any(|event| matches!(event, AgentEvent::ToolCallFinished { name, content_len, .. } if name == "add" && *content_len == 1)));
}

#[tokio::test]
async fn managed_tool_loop_runs_mixed_compaction_and_registry_tools() {
    let server = MockServer::start();
    let tool_call_mock = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .body_contains("\"messages\":[{\"role\":\"user\",\"content\":\"old context\"},{\"role\":\"user\",\"content\":\"add 2 and 3\"}]")
            .body_contains(COMPACT_SUMMARIZE_MESSAGES)
            .body_contains("\"name\":\"add\"");
        then.status(200).json_body(json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        {
                            "id": "compact_1",
                            "type": "function",
                            "function": {
                                "name": COMPACT_SUMMARIZE_MESSAGES,
                                "arguments": "{\"turn_ids\":[1],\"summary\":\"The user mentioned old context.\",\"reason\":\"keep context short\"}"
                            }
                        },
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "add",
                                "arguments": "{\"a\":2,\"b\":3}"
                            }
                        }
                    ]
                }
            }]
        }));
    });
    let final_mock = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .body_contains("\"role\":\"tool\"")
            .body_contains("Compaction acknowledged:")
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

    let provider =
        ChatProvider::new(ProviderConfig::llama_cpp(server.base_url(), "local-model").unwrap());
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
                Ok(json!(
                    arguments["a"].as_i64().unwrap() + arguments["b"].as_i64().unwrap()
                ))
            })
        },
    ));
    let (mut session, _receiver) = ChatSession::with_config(SessionConfig {
        compaction: CompactionConfig {
            protect_recent_turns: 0,
            ..CompactionConfig::default()
        },
        ..SessionConfig::default()
    });
    session.push_user("old context");

    let response = session
        .run_user_message_with_tools(&provider, &registry, "add 2 and 3")
        .await
        .unwrap();

    tool_call_mock.assert();
    final_mock.assert();
    assert_eq!(response.message.content, "2 + 3 = 5");
    assert!(session.turns().iter().any(|turn| turn.name.as_deref()
        == Some(COMPACT_SUMMARIZE_MESSAGES)
        && turn.content.text().starts_with("Compaction acknowledged:")));
    assert!(
        session
            .turns()
            .iter()
            .any(|turn| turn.name.as_deref() == Some("add") && turn.content == "5")
    );
}

#[tokio::test]
async fn model_payload_telemetry_captures_request_and_response() {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(POST).path("/v1/chat/completions");
        then.status(200).json_body(json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "pong"
                }
            }]
        }));
    });

    let provider =
        ChatProvider::new(ProviderConfig::llama_cpp(server.base_url(), "local-model").unwrap());
    let (mut session, _events, telemetry) = ChatSession::with_config_and_telemetry(SessionConfig {
        telemetry: TelemetryConfig::all(),
        ..SessionConfig::default()
    });
    let telemetry = telemetry.unwrap();

    let response = session.send_user_message(&provider, "ping").await.unwrap();

    mock.assert();
    assert_eq!(response.message.content, "pong");
    let events = (0..4)
        .map(|_| telemetry.recv().unwrap())
        .collect::<Vec<_>>();
    assert!(events.iter().any(|event| matches!(
        event,
        TelemetryEvent::ModelRequest {
            request: Some(request),
            message_count: 1,
            ..
        } if request.messages[0].content == "ping"
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        TelemetryEvent::ModelResponse {
            response: Some(response),
            content_len: 4,
            ..
        } if response.message.content == "pong"
    )));
}

/// A session with a system prompt and `n` exchanges, each a user turn, a tool call, its result
/// and an answer.
fn session_with_exchanges(
    config: SessionConfig,
    n: usize,
) -> (ChatSession, Option<AgentEventReceiver>) {
    let (mut session, receiver) = ChatSession::with_system_prompt_and_config("system", config);
    for i in 0..n {
        session.push_user(format!("question {i} {}", "x".repeat(400)));
        session.push(ChatMessage {
            tool_calls: vec![ToolCall {
                id: format!("call_{i}"),
                kind: "function".to_string(),
                function: ToolCallFunction {
                    name: "lookup".to_string(),
                    arguments: "{}".to_string(),
                },
            }],
            ..ChatMessage::assistant("")
        });
        session.push(ChatMessage::tool_result(
            format!("call_{i}"),
            "lookup",
            "y".repeat(400),
        ));
        session.push_assistant(format!("answer {i}"));
    }
    (session, receiver)
}

#[tokio::test]
async fn trim_only_compaction_drops_oldest_exchanges_without_asking_the_model() {
    let server = MockServer::start();
    let any = server.mock(|when, then| {
        when.method(POST);
        then.status(500);
    });
    let provider =
        ChatProvider::new(ProviderConfig::llama_cpp(server.base_url(), "model").unwrap());
    let (mut session, receiver) = session_with_exchanges(
        SessionConfig {
            compaction: CompactionConfig::trim_only(500, 300),
            notifications: NotificationConfig {
                compaction: true,
                ..NotificationConfig::default()
            },
            ..SessionConfig::default()
        },
        4,
    );
    let before = session.estimated_context_tokens();
    session.maybe_compact(&provider).await.unwrap();
    any.assert_hits(0);

    let after = session.estimated_context_tokens();
    assert!(after <= 300, "{before} -> {after}");
    let texts = session
        .turns()
        .iter()
        .map(|t| t.content.text().into_owned())
        .collect::<Vec<_>>();
    assert_eq!(texts[0], "system");
    assert_eq!(
        session.turns()[1].name.as_deref(),
        Some(compaction::TRIMMED_NOTE_NAME)
    );
    // Whole exchanges went, oldest first; the latest stays.
    assert!(
        !texts
            .iter()
            .any(|t| t.starts_with("question 0") || t.starts_with("question 2"))
    );
    assert!(texts.iter().any(|t| t.starts_with("question 3")));
    assert_eq!(session.turns()[2].role, MessageRole::User);
    // Every remaining tool result still has its call.
    for turn in session
        .turns()
        .iter()
        .filter(|t| t.role == MessageRole::Tool)
    {
        let id = turn.tool_call_id.as_deref().unwrap();
        assert!(
            session
                .turns()
                .iter()
                .any(|t| t.tool_calls.iter().any(|c| c.id == id))
        );
    }
    let events = receiver.unwrap().try_iter().collect::<Vec<_>>();
    assert!(events.iter().any(|e| matches!(
        e,
        AgentEvent::CompactionTrimmed {
            removed_turns: 12,
            ..
        }
    )));

    // A second trim keeps a single note.
    for i in 4..8 {
        session.push_user(format!("question {i} {}", "x".repeat(400)));
        session.push_assistant(format!("answer {i}"));
    }
    session.maybe_compact(&provider).await.unwrap();
    let notes = session
        .turns()
        .iter()
        .filter(|t| t.name.as_deref() == Some(compaction::TRIMMED_NOTE_NAME))
        .count();
    assert_eq!(notes, 1);
}

#[test]
fn trimming_keeps_the_latest_exchange_even_above_the_target() {
    let (mut session, _) = session_with_exchanges(
        SessionConfig {
            compaction: CompactionConfig {
                protect_recent_turns: 0,
                ..CompactionConfig::trim_only(1, 0)
            },
            ..SessionConfig::default()
        },
        1,
    );
    let removed = compaction::trim_oldest(
        &mut session.turns,
        &mut session.next_turn_id,
        &session.config.compaction,
    );
    assert_eq!(removed, 0);
    assert_eq!(session.turns().len(), 5);
}

#[test]
fn trim_only_never_offers_compaction_tools() {
    let (session, _) = ChatSession::with_config(SessionConfig {
        compaction: CompactionConfig {
            always_offer_tools: true,
            ..CompactionConfig::trim_only(10, 5)
        },
        ..SessionConfig::default()
    });
    assert!(
        session
            .definitions_with_compaction(&ToolRegistry::new())
            .is_empty()
    );
}

#[test]
fn compaction_skips_protected_and_unknown_ids_instead_of_failing() {
    let (mut session, _) = session_with_exchanges(SessionConfig::default(), 3);
    let last = session.turns().last().unwrap().id;
    let result = session
        .compact_remove_messages(&[1, 2, 3, 4, 999, last], "old")
        .unwrap();
    assert!(
        result.contains("Skipped turn_ids: 1 (protected), 999 (unknown)"),
        "{result}"
    );
    assert!(session.turns().iter().all(|t| ![2, 3, 4].contains(&t.id)));
    assert_eq!(session.turns()[0].id, 1);
    assert_eq!(session.turns().last().unwrap().id, last);

    let error = session
        .compact_remove_messages(&[1, last], "nothing")
        .unwrap_err()
        .to_string();
    assert!(error.contains("no turn_id can be compacted"), "{error}");
    assert!(error.contains("protected turn_ids: [1, "), "{error}");
}

#[tokio::test]
async fn stop_after_tools_ends_the_loop_on_a_successful_call() {
    let server = MockServer::start();
    let submit = |id: &str, value: i64| {
        json!({"choices": [{"message": {"content": null, "tool_calls": [{
            "id": id, "type": "function",
            "function": {"name": "submit", "arguments": json!({"value": value}).to_string()}
        }]}}]})
    };
    // The first call is rejected by the tool; the second succeeds and ends the loop.
    let first = server.mock(|when, then| {
        when.method(POST).matches(|request| {
            let body: serde_json::Value =
                serde_json::from_slice(request.body.as_deref().unwrap()).unwrap();
            body["messages"].as_array().unwrap().last().unwrap()["role"] == "user"
        });
        then.status(200).json_body(submit("bad", -1));
    });
    let second = server.mock(|when, then| {
        when.method(POST).body_contains("must be positive");
        then.status(200).json_body(submit("good", 3));
    });
    let provider =
        ChatProvider::new(ProviderConfig::llama_cpp(server.base_url(), "model").unwrap());
    let registry = ToolRegistry::new().with_tool(Tool::new(
        ToolDefinition::new("submit", "Submit the answer.", json!({"type": "object"})),
        |arguments| {
            Box::pin(async move {
                if arguments["value"].as_i64().unwrap() < 0 {
                    return Err(crate::Error::Tool("must be positive".into()));
                }
                Ok(json!("ok"))
            })
        },
    ));
    let (mut session, _) = ChatSession::with_config(SessionConfig {
        stop_after_tools: vec!["submit".to_string()],
        ..SessionConfig::default()
    });
    let response = session
        .run_user_message_with_tools(&provider, &registry, "go")
        .await
        .unwrap();
    first.assert_hits(1);
    second.assert_hits(1);
    assert_eq!(response.message.tool_calls[0].id, "good");
    assert_eq!(session.turns().last().unwrap().role, MessageRole::Tool);
    assert_eq!(
        session.turns().last().unwrap().tool_call_id.as_deref(),
        Some("good")
    );
}
