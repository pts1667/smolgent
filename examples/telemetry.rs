use httpmock::Method::POST;
use httpmock::MockServer;
use serde_json::json;
use smolgent::{
    ChatProvider, ChatSession, ProviderConfig, SessionConfig, TelemetryConfig, TelemetryEvent,
};

#[tokio::main]
async fn main() -> smolgent::Result<()> {
    let server = MockServer::start();
    let _mock = server.mock(|when, then| {
        when.method(POST).path("/v1/chat/completions");
        then.status(200).json_body(json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Telemetry is flowing."
                }
            }]
        }));
    });

    let provider = ChatProvider::new(ProviderConfig::llama_cpp(
        server.base_url(),
        "telemetry-demo-model",
    )?);
    let (mut session, _events, telemetry) = ChatSession::with_system_prompt_config_and_telemetry(
        "You are concise and helpful.",
        SessionConfig {
            telemetry: TelemetryConfig::all(),
            telemetry_channel_capacity: Some(16),
            ..SessionConfig::default()
        },
    );
    let telemetry = telemetry.expect("TelemetryConfig::all enables a telemetry stream");

    let response = session
        .send_user_message(&provider, "Reply in one short sentence.")
        .await?;
    println!("assistant: {}", response.message.content);
    println!();
    println!("telemetry:");

    while let Ok(event) = telemetry.try_recv() {
        print_event(event);
    }

    Ok(())
}

fn print_event(event: TelemetryEvent) {
    match event {
        TelemetryEvent::UserMessage {
            turn_id,
            content_len,
            content,
        } => {
            println!(
                "- user message turn={turn_id} len={content_len} content={:?}",
                content
            );
        }
        TelemetryEvent::AssistantMessage {
            turn_id,
            content_len,
            tool_calls,
            content,
        } => {
            println!(
                "- assistant message turn={turn_id} len={content_len} tool_calls={tool_calls} content={:?}",
                content
            );
        }
        TelemetryEvent::ToolMessage {
            turn_id,
            tool_call_id,
            name,
            content_len,
            content,
        } => {
            println!(
                "- tool message turn={turn_id} call={tool_call_id} name={name} len={content_len} content={:?}",
                content
            );
        }
        TelemetryEvent::ModelRequest {
            provider,
            model,
            message_count,
            tool_count,
            request,
        } => {
            println!(
                "- model request provider={provider} model={model} messages={message_count} tools={tool_count} payload_captured={}",
                request.is_some()
            );
        }
        TelemetryEvent::ModelResponse {
            provider,
            model,
            content_len,
            tool_calls,
            response,
        } => {
            println!(
                "- model response provider={provider} model={model} len={content_len} tool_calls={tool_calls} payload_captured={}",
                response.is_some()
            );
        }
        TelemetryEvent::ToolCallStarted {
            tool_call_id,
            name,
            arguments_len,
            arguments,
        } => {
            println!(
                "- tool call started call={tool_call_id} name={name} args_len={arguments_len} args={:?}",
                arguments
            );
        }
        TelemetryEvent::ToolCallFinished {
            tool_call_id,
            name,
            content_len,
            content,
        } => {
            println!(
                "- tool call finished call={tool_call_id} name={name} len={content_len} content={:?}",
                content
            );
        }
        TelemetryEvent::ToolCallFailed {
            tool_call_id,
            name,
            error,
        } => {
            println!("- tool call failed call={tool_call_id} name={name} error={error}");
        }
    }
}
