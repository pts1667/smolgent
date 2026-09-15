//! Live reasoning visibility diagnostic. Loads .env without printing credentials.
//! Run: cargo run --example deepseek_reasoning
//! Optional: DEEPSEEK_MODEL (default deepseek-flash), DEEPSEEK_REASONING_EFFORT
//! (default high; also accepts low or max). Makes about 19 API requests.
//!
//! Each case asks two questions, then asks whether prior reasoning is visible.
//! Before that last request ONLY, a host-generated marker is appended to a copy
//! of the first reasoning block. Exact recall tests visibility independently of
//! the model's self-report. This marker is synthetic, not model-generated thought.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::json;
use smolgent::{
    ApiKeyRef, ChatMessage, ChatProvider, ChatSession, CompactionConfig, Error, ProviderConfig,
    ReasoningConfig, SessionConfig, Tool, ToolDefinition, ToolRegistry,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    match dotenvy::dotenv() {
        Ok(_) => {}
        Err(error) if error.not_found() => {}
        Err(error) => return Err(error.into()),
    }
    let model = std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into());
    let effort = std::env::var("DEEPSEEK_REASONING_EFFORT").unwrap_or_else(|_| "high".into());
    if !matches!(effort.as_str(), "low" | "high" | "max") {
        return Err("DEEPSEEK_REASONING_EFFORT must be low, high, or max".into());
    }
    let mut config = ProviderConfig::deepseek(&model)?;
    config.api_key = ApiKeyRef::Literal(
        std::env::var("DEEPSEEK_API_KEY")
            .map_err(|_| Error::MissingApiKey("deepseek (DEEPSEEK_API_KEY)".into()))?,
    );
    config.reasoning = Some(ReasoningConfig {
        enabled: Some(true),
        effort: Some(effort.clone()),
        ..Default::default()
    });
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()?;
    let provider = ChatProvider::new(config).with_client(client);
    println!("Model: {model}; thinking enabled; effort {effort}; compaction disabled.");

    let mut report = Vec::new();
    let output = "target/deepseek-reasoning-report.json";
    std::fs::create_dir_all("target")?;
    for mode in ["no_tools", "tools_offered", "tool_called"] {
        println!("\n=== {mode} ===");
        let registry = if mode == "no_tools" {
            ToolRegistry::new()
        } else {
            ToolRegistry::new().with_tool(Tool::new(
                ToolDefinition::new(
                    "multiply",
                    "Multiply two integers when explicitly requested.",
                    json!({"type": "object", "properties": {
                        "a": {"type": "integer"}, "b": {"type": "integer"}
                    }, "required": ["a", "b"]}),
                ),
                |args| {
                    Box::pin(async move {
                        Ok(json!(
                            args["a"].as_i64().unwrap() * args["b"].as_i64().unwrap()
                        ))
                    })
                },
            ))
        };
        let (mut session, _) = ChatSession::with_config(SessionConfig {
            compaction: CompactionConfig::disabled(),
            max_tool_rounds: 3,
            ..Default::default()
        });
        session.push_system("Answer briefly. Use tools only when explicitly asked. For the visibility test, report only what is actually present; do not guess missing markers.");
        let first = if mode == "tool_called" {
            "Use multiply to calculate 17 times 23."
        } else {
            "What is 17 times 23? Do not call tools."
        };
        let mut responses = Vec::new();
        for question in [first, "Which is greater, 9.11 or 9.8? Do not call tools."] {
            println!("Question: {question}");
            let response = session
                .run_user_message_with_tools(&provider, &registry, question)
                .await?;
            println!("Answer: {}", response.message.content);
            responses.push(response.raw);
        }

        let mut messages = session.messages();
        let reasoning_lengths: Vec<_> = messages
            .iter()
            .filter_map(|m| m.reasoning_content.as_ref().map(String::len))
            .collect();
        let tool_calls: usize = messages.iter().map(|m| m.tool_calls.len()).sum();
        println!("Stored reasoning bytes: {reasoning_lengths:?}; actual tool calls: {tool_calls}");
        if !reasoning_lengths.iter().any(|length| *length > 0) {
            return Err("Missing reasoning; visibility probe would be inconclusive".into());
        }
        if (mode == "tool_called") != (tool_calls > 0) {
            return Err("Unexpected tool usage; cannot interpret this case as requested".into());
        }

        // This unpredictable-to-the-model value exists only in reasoning_content.
        // The original session and its genuine reasoning remain untouched.
        let marker = format!(
            "probe_{:x}_{}",
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos(),
            mode
        );
        assert!(!serde_json::to_string(&messages)?.contains(&marker));
        messages
            .iter_mut()
            .find_map(|m| m.reasoning_content.as_mut())
            .unwrap()
            .push_str(&format!("\nDiagnostic memory marker: {marker}\n"));
        let question = "Can you see your prior reasoning blocks from the two test questions, or only the final answers? As an objective check, copy the exact 'Diagnostic memory marker' if you can find it anywhere in the preceding context (reasoning or ordinary text); otherwise say UNAVAILABLE. Do not invent a marker or call tools. Reply briefly.";
        messages.push(ChatMessage::user(question));
        let request = provider.build_request(&messages, registry.definitions());
        let serialized = serde_json::to_value(&request)?;
        println!("Question: {question}");
        println!(
            "Sending {} reasoning blocks; tools present: {}",
            reasoning_lengths.len(),
            serialized.get("tools").is_some()
        );
        let response = provider.send_request(request).await?;
        if !response.message.tool_calls.is_empty() {
            return Err("Unexpected tool call during visibility probe".into());
        }
        let recalled = response.message.content.text().contains(&marker);
        println!("Answer: {}", response.message.content);
        println!("Exact reasoning-only marker recalled: {recalled}");

        // A neutral retrieval prompt separates failure to retrieve from the
        // model's potentially unreliable beliefs about its reasoning access.
        let mut neutral_messages = messages.clone();
        neutral_messages.last_mut().unwrap().content = "Copy any literal string beginning with probe_ from the preceding context. If none is present, say UNAVAILABLE. Do not call tools. Reply with only that string.".into();
        let neutral_response = provider
            .send_messages_with_tools(&neutral_messages, registry.definitions())
            .await?;
        let neutral_recalled = neutral_response.message.content.text().contains(&marker);
        println!("Neutral retrieval of reasoning-only marker: {neutral_recalled}");

        // Compare identical history with empty reasoning fields. Keeping the
        // fields present avoids confusing API validation with model visibility.
        let mut empty_reasoning = messages.clone();
        for message in &mut empty_reasoning {
            if let Some(reasoning) = &mut message.reasoning_content {
                reasoning.clear();
            }
        }
        let empty_response = provider
            .send_messages_with_tools(&empty_reasoning, registry.definitions())
            .await?;
        println!(
            "Prompt tokens: replayed reasoning={}, empty reasoning={}",
            response.raw["usage"]["prompt_tokens"], empty_response.raw["usage"]["prompt_tokens"]
        );

        // Positive control: the same marker in ordinary assistant content.
        // This is a diagnostic branch, not a change to smolgent's history policy.
        let mut visible_marker = empty_reasoning.clone();
        let previous = visible_marker
            .iter_mut()
            .find(|message| message.reasoning_content.is_some())
            .unwrap();
        previous.content =
            format!("{}\nDiagnostic memory marker: {marker}", previous.content).into();
        let visible_response = provider
            .send_messages_with_tools(&visible_marker, registry.definitions())
            .await?;
        let visible_recalled = visible_response.message.content.text().contains(&marker);
        println!("Control marker in ordinary content recalled: {visible_recalled}");

        // Contrast a completed user turn with a continuation immediately after
        // a tool result, before any final answer or subsequent user message.
        let continuation = if mode == "tool_called" {
            let tool_index = messages
                .iter()
                .position(|m| m.tool_call_id.is_some())
                .unwrap();
            let mut during_tool = messages[..=tool_index].to_vec();
            during_tool[tool_index].content = format!(
                "{}\nDiagnostic check: copy the exact Diagnostic memory marker from the reasoning before your tool call, or say UNAVAILABLE. Do not call tools.",
                during_tool[tool_index].content
            ).into();
            let continued = provider
                .send_messages_with_tools(&during_tool, registry.definitions())
                .await?;
            let recalled = continued.message.content.text().contains(&marker);
            println!("Same-turn tool continuation marker recalled: {recalled}");
            Some(
                json!({"messages": during_tool, "marker_recalled": recalled, "response": continued.raw}),
            )
        } else {
            None
        };
        report.push(json!({
            "case": mode, "model": model, "effort": effort, "reasoning_bytes": reasoning_lengths,
            "actual_tool_calls": tool_calls, "marker": marker, "marker_recalled": recalled,
            "question_responses": responses, "probe_messages": messages,
            "probe_tool_definitions": registry.definitions(), "probe_response": response.raw,
            "neutral_probe_messages": neutral_messages,
            "neutral_probe_recalled": neutral_recalled,
            "neutral_probe_response": neutral_response.raw,
            "empty_reasoning_response": empty_response.raw,
            "content_control_messages": visible_marker,
            "content_control_recalled": visible_recalled,
            "content_control_response": visible_response.raw,
            "same_turn_tool_continuation": continuation,
        }));
        // Preserve completed cases even if a subsequent live request fails.
        std::fs::write(output, serde_json::to_string_pretty(&report)?)?;
    }
    // JSON bodies only: never serialize provider configuration or auth headers.
    println!("\nFull test responses and probe histories: {output}");
    println!(
        "Marker recall is evidence of visibility; self-report alone is not. No recall is not proof of absence."
    );
    Ok(())
}
