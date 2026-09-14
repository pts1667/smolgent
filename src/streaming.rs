//! Streaming Chat Completions transport shared by providers and managed sessions.
use std::{collections::BTreeMap, future::Future};

use futures_channel::mpsc;
use futures_util::{FutureExt, StreamExt, stream::BoxStream};
use serde::Serialize;
use serde_json::{Value, json};

use crate::{ChatResponse, Error, MessageContent, Result, ToolCall};

/// Events across all model requests and tool rounds in one run.
/// Text deltas include intermediate assistant messages. `Completed` is the final answer.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    ModelStarted,
    TextDelta {
        text: String,
    },
    /// Provider-supplied reasoning, kept separate from visible assistant text.
    ReasoningDelta {
        text: String,
    },
    ToolCallDelta {
        index: u64,
        delta: Value,
    },
    /// Complete, validated model response, before any tools execute.
    ModelCompleted {
        response: ChatResponse,
    },
    ToolStarted {
        call: ToolCall,
    },
    ToolResult {
        tool_call_id: String,
        name: String,
        content: MessageContent,
    },
    /// Emitted once, after successful completion and session history commit.
    Completed {
        response: ChatResponse,
    },
}

/// Dropping a stream cancels its in-flight request/run. Poll it to make progress.
pub type EventStream<'a> = BoxStream<'a, Result<StreamEvent>>;
pub(crate) type Sender = mpsc::Sender<StreamEvent>;

// Drive the producer in the consumer's task: no detached tasks or runtime requirement.
pub(crate) fn drive<'a, F, Fut>(run: F) -> EventStream<'a>
where
    F: FnOnce(Sender) -> Fut + Send + 'a,
    Fut: Future<Output = Result<ChatResponse>> + Send + 'a,
{
    Box::pin(async_stream::try_stream! {
        let (sender, mut receiver) = mpsc::channel(16);
        let future = run(sender).fuse();
        futures_util::pin_mut!(future);
        let result = loop {
            futures_util::select_biased! {
                event = receiver.next() => {
                    if let Some(event) = event { yield event; }
                    else { break future.await; }
                },
                result = future => break result,
            }
        };
        receiver.close();
        while let Some(event) = receiver.next().await { yield event; }
        yield StreamEvent::Completed { response: result? };
    })
}

const MAX_EVENT_BYTES: usize = 1024 * 1024;
const MAX_DATA_BYTES: usize = 10 * 1024 * 1024;

/// Incremental SSE framing, including split UTF-8, CR/LF/CRLF, comments and multiline data.
#[derive(Default)]
pub(crate) struct SseDecoder {
    line: Vec<u8>,
    data: String,
    data_seen: bool,
    skip_lf: bool,
    first_line: bool,
    started: bool,
    total: usize,
}

impl SseDecoder {
    pub(crate) fn feed(&mut self, bytes: &[u8]) -> Result<Vec<String>> {
        let mut events = Vec::new();
        for &byte in bytes {
            if self.skip_lf && byte == b'\n' {
                self.skip_lf = false;
                continue;
            }
            self.skip_lf = byte == b'\r';
            if byte == b'\r' || byte == b'\n' {
                if !self.started {
                    self.first_line = true;
                    self.started = true;
                }
                let line = std::str::from_utf8(&self.line)
                    .map_err(|_| Error::Stream("invalid UTF-8 in SSE frame".into()))?;
                let line = if self.first_line {
                    line.trim_start_matches('\u{feff}')
                } else {
                    line
                };
                self.first_line = false;
                if line.is_empty() {
                    if self.data_seen {
                        self.data.pop(); // trailing newline from the last data field
                        events.push(std::mem::take(&mut self.data));
                        self.data_seen = false;
                    }
                } else {
                    let (field, value) = line.split_once(':').unwrap_or((line, ""));
                    if field == "data" {
                        let value = value.strip_prefix(' ').unwrap_or(value);
                        self.total += value.len();
                        if self.total > MAX_DATA_BYTES
                            || self.data.len() + value.len() > MAX_EVENT_BYTES
                        {
                            return Err(Error::Stream(
                                "SSE data exceeded the response/event size limit".into(),
                            ));
                        }
                        self.data.push_str(value);
                        self.data.push('\n');
                        self.data_seen = true;
                    }
                }
                self.line.clear();
            } else {
                self.line.push(byte);
                if self.line.len() > MAX_EVENT_BYTES {
                    return Err(Error::Stream("SSE line exceeded the size limit".into()));
                }
            }
        }
        Ok(events)
    }
}

#[derive(Default)]
pub(crate) struct Accumulator {
    metadata: serde_json::Map<String, Value>,
    message: serde_json::Map<String, Value>,
    tools: BTreeMap<u64, Value>,
    details: BTreeMap<u64, Value>,
    finish: Option<String>,
    seen_choice: bool,
}

// Merge provider reasoning detail fragments by index, preserving opaque fields for replay.
fn merge_object(target: &mut Value, delta: &Value, fragments: &[&str]) -> Result<()> {
    let source = delta
        .as_object()
        .ok_or_else(|| Error::Stream("expected an object delta".into()))?;
    if !target.is_object() {
        *target = json!({});
    }
    for (key, value) in source {
        if key == "index" || value.is_null() {
            continue;
        }
        if fragments.contains(&key.as_str()) {
            let text = value
                .as_str()
                .ok_or_else(|| Error::Stream(format!("invalid {key} delta")))?;
            let combined = target
                .as_object_mut()
                .unwrap()
                .entry(key)
                .or_insert_with(|| json!(""));
            let Value::String(combined) = combined else {
                return Err(Error::Stream(format!("mixed {key} delta types")));
            };
            combined.push_str(text);
        } else {
            target[key] = value.clone();
        }
    }
    Ok(())
}

impl Accumulator {
    pub(crate) fn push(&mut self, value: Value) -> Result<Vec<StreamEvent>> {
        if let Some(error) = value.get("error").filter(|e| !e.is_null()) {
            return Err(Error::Stream(format!("provider error: {error}")));
        }
        let object = value
            .as_object()
            .ok_or_else(|| Error::Stream("expected a chunk object".into()))?;
        for (key, value) in object {
            if key != "choices" && key != "object" {
                self.metadata.insert(key.clone(), value.clone());
            }
        }
        let choices = value["choices"]
            .as_array()
            .ok_or_else(|| Error::Stream("chunk has no choices array".into()))?;
        let mut events = Vec::new();
        for choice in choices {
            if choice["index"].as_u64() != Some(0) {
                continue;
            }
            self.seen_choice = true;
            if let Some(reason) = choice["finish_reason"].as_str() {
                if reason == "error" {
                    return Err(Error::Stream("provider reported a generation error".into()));
                }
                self.finish = Some(reason.to_owned());
            }
            let delta = &choice["delta"];
            if delta.is_null() {
                continue;
            }
            if !delta.is_object() {
                return Err(Error::Stream("invalid choice delta".into()));
            }
            for field in ["content", "reasoning_content", "reasoning"] {
                if let Some(text) = delta[field].as_str() {
                    let combined = self.message.entry(field).or_insert_with(|| json!(""));
                    let Value::String(combined) = combined else {
                        return Err(Error::Stream("mixed content delta types".into()));
                    };
                    combined.push_str(text);
                    if !text.is_empty() {
                        events.push(if field == "content" {
                            StreamEvent::TextDelta { text: text.into() }
                        } else {
                            StreamEvent::ReasoningDelta { text: text.into() }
                        });
                    }
                } else if !delta[field].is_null() {
                    return Err(Error::Stream(format!("unsupported {field} delta type")));
                }
            }
            if let Some(details) = delta.get("reasoning_details").filter(|v| !v.is_null()) {
                for detail in details
                    .as_array()
                    .ok_or_else(|| Error::Stream("invalid reasoning_details".into()))?
                {
                    let index = detail["index"].as_u64().unwrap_or(0);
                    merge_object(
                        self.details.entry(index).or_insert_with(|| json!({})),
                        detail,
                        &["text", "summary", "data", "signature"],
                    )?;
                    if delta["reasoning"].as_str().unwrap_or_default().is_empty()
                        && delta["reasoning_content"]
                            .as_str()
                            .unwrap_or_default()
                            .is_empty()
                        && let Some(text) = detail["text"]
                            .as_str()
                            .or_else(|| detail["summary"].as_str())
                        && !text.is_empty()
                    {
                        events.push(StreamEvent::ReasoningDelta { text: text.into() });
                    }
                }
            }
            if let Some(annotations) = delta.get("annotations").filter(|v| !v.is_null()) {
                let items = annotations
                    .as_array()
                    .ok_or_else(|| Error::Stream("invalid annotations".into()))?;
                self.message
                    .entry("annotations")
                    .or_insert_with(|| json!([]))
                    .as_array_mut()
                    .unwrap()
                    .extend(items.iter().cloned());
            }
            if let Some(calls) = delta.get("tool_calls").filter(|v| !v.is_null()) {
                for call in calls
                    .as_array()
                    .ok_or_else(|| Error::Stream("invalid tool call deltas".into()))?
                {
                    let index = call["index"]
                        .as_u64()
                        .ok_or_else(|| Error::Stream("tool delta has no index".into()))?;
                    let target = self
                        .tools
                        .entry(index)
                        .or_insert_with(|| json!({"type":"function", "function":{}}));
                    if let Some(id) = call.get("id").filter(|v| !v.is_null()) {
                        let id = id
                            .as_str()
                            .ok_or_else(|| Error::Stream("invalid tool id".into()))?;
                        let previous = target["id"].as_str().unwrap_or_default();
                        target["id"] = json!(format!("{previous}{id}"));
                    }
                    if let Some(kind) = call.get("type").filter(|v| !v.is_null()) {
                        target["type"] = kind.clone();
                    }
                    if let Some(function) = call.get("function").filter(|v| !v.is_null()) {
                        merge_object(&mut target["function"], function, &["name", "arguments"])?;
                    }
                    events.push(StreamEvent::ToolCallDelta {
                        index,
                        delta: call.clone(),
                    });
                }
            }
        }
        Ok(events)
    }

    pub(crate) fn finish(mut self) -> Result<Value> {
        if !self.seen_choice || self.finish.is_none() {
            return Err(Error::Stream(
                "stream ended without a completed choice".into(),
            ));
        }
        if !matches!(
            self.finish.as_deref(),
            Some("stop" | "tool_calls" | "length" | "content_filter")
        ) {
            return Err(Error::Stream(format!(
                "generation stopped: {}",
                self.finish.as_deref().unwrap()
            )));
        }
        if self.finish.as_deref() == Some("tool_calls") && self.tools.is_empty() {
            return Err(Error::Stream(
                "stream finished with missing tool calls".into(),
            ));
        }
        if !self.tools.is_empty() {
            if !matches!(self.finish.as_deref(), Some("tool_calls" | "stop")) {
                return Err(Error::Stream("tool call generation was interrupted".into()));
            }
            let mut ids = std::collections::HashSet::new();
            for call in self.tools.values() {
                let call: ToolCall = serde_json::from_value(call.clone())?;
                if call.id.is_empty()
                    || call.function.name.is_empty()
                    || call.kind != "function"
                    || !ids.insert(call.id)
                {
                    return Err(Error::Stream("incomplete or duplicate tool call".into()));
                }
                let args: Value = serde_json::from_str(&call.function.arguments)?;
                if !args.is_object() {
                    return Err(Error::Stream("tool arguments must be a JSON object".into()));
                }
            }
            self.message
                .insert("tool_calls".into(), self.tools.into_values().collect());
        }
        if !self.details.is_empty() {
            // Keep indexes when replaying opaque reasoning metadata.
            let details: Vec<_> = self
                .details
                .into_iter()
                .map(|(index, mut detail)| {
                    detail["index"] = json!(index);
                    detail
                })
                .collect();
            self.message
                .insert("reasoning_details".into(), json!(details));
        }
        self.message.insert("role".into(), json!("assistant"));
        self.metadata
            .insert("object".into(), json!("chat.completion"));
        self.metadata.insert(
            "choices".into(),
            json!([{"index":0,"finish_reason":self.finish,"message":self.message}]),
        );
        Ok(Value::Object(self.metadata))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChatProvider, ChatSession, ProviderConfig, ToolRegistry};

    #[test]
    fn sse_handles_every_byte_boundary_and_line_ending() {
        let bytes = "\u{feff}: keep-alive\r\ndata: {\rdata: \"text\":\"你好🙂\"}\n\nid: ignored\n\ndata: [DONE]\r\n\r\n".as_bytes();
        for size in 1..=bytes.len() {
            let mut parser = SseDecoder::default();
            let events: Vec<_> = bytes
                .chunks(size)
                .flat_map(|chunk| parser.feed(chunk).unwrap())
                .collect();
            assert_eq!(events, vec!["{\n\"text\":\"你好🙂\"}", "[DONE]"]);
        }
    }

    fn chunk(delta: Value, finish: Value) -> Value {
        json!({"choices":[{"index":0,"delta":delta,"finish_reason":finish}]})
    }

    #[test]
    fn interleaved_tools_reasoning_and_usage_are_assembled() {
        let mut acc = Accumulator::default();
        acc.push(chunk(json!({"reasoning_content":"Think ", "reasoning_details":[{"index":2,"type":"reasoning.text","id":"r1","text":"First "}],
            "tool_calls":[{"index":1,"id":"b","function":{"name":"ad","arguments":"{\"b\":"}},
                          {"index":0,"id":"a","function":{"name":"add","arguments":"{\"a\":"}}]}), Value::Null)).unwrap();
        acc.push(chunk(json!({"reasoning_content":"more", "reasoning_details":[{"index":2,"text":"second","signature":"opaque"}],
            "tool_calls":[{"index":0,"function":{"arguments":"1}"}}, {"index":1,"function":{"name":"d","arguments":"2}"}}]}), json!("tool_calls"))).unwrap();
        // OpenRouter repeats finish_reason on the usage frame.
        acc.push(json!({"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"completion_tokens":7}})).unwrap();
        let raw = acc.finish().unwrap();
        let message = &raw["choices"][0]["message"];
        assert_eq!(message["reasoning_content"], "Think more");
        assert_eq!(message["reasoning_details"][0]["text"], "First second");
        assert_eq!(message["reasoning_details"][0]["signature"], "opaque");
        assert_eq!(message["reasoning_details"][0]["index"], 2);
        assert_eq!(message["tool_calls"][0]["id"], "a");
        assert_eq!(message["tool_calls"][1]["function"]["name"], "add");
        assert_eq!(
            message["tool_calls"][1]["function"]["arguments"],
            "{\"b\":2}"
        );
        assert_eq!(raw["usage"]["completion_tokens"], 7);
    }

    #[test]
    fn malformed_truncated_and_error_streams_fail() {
        let mut acc = Accumulator::default();
        assert!(acc.push(json!({"error":{"message":"failed"}})).is_err());
        assert!(Accumulator::default().finish().is_err());
        for reason in ["insufficient_system_resource", "tool_calls"] {
            let mut acc = Accumulator::default();
            acc.push(chunk(json!({"content":"partial"}), json!(reason)))
                .unwrap();
            assert!(acc.finish().is_err());
        }
        for (arguments, finish) in [("{", "tool_calls"), ("{}", "length"), ("[]", "tool_calls")] {
            let mut acc = Accumulator::default();
            acc.push(chunk(json!({"tool_calls":[{"index":0,"id":"a","function":{"name":"add","arguments":arguments}}]}), json!(finish))).unwrap();
            assert!(acc.finish().is_err());
        }
        assert!(
            SseDecoder::default()
                .feed(&vec![b'x'; MAX_EVENT_BYTES + 1])
                .is_err()
        );
    }

    #[tokio::test]
    async fn public_streams_commit_only_on_success() {
        let server = httpmock::MockServer::start_async().await;
        let body = format!(
            "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
            chunk(json!({"content":"Hello"}), Value::Null),
            chunk(json!({}), json!("stop"))
        );
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .body_contains("\"stream\":true");
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(body);
        });
        let provider =
            ChatProvider::new(ProviderConfig::llama_cpp(server.base_url(), "test").unwrap());
        let registry = ToolRegistry::new();
        let mut session = ChatSession::with_system_prompt("Test");
        session.config_mut().compaction.enabled = false;
        {
            let mut stream = session.stream_user_message_with_tools(&provider, &registry, "Hi");
            while let Some(event) = stream.next().await {
                if matches!(event.unwrap(), StreamEvent::TextDelta { .. }) {
                    break;
                }
            }
        }
        assert_eq!(session.messages().len(), 1);
        let events: Vec<_> = session
            .stream_user_message_with_tools(&provider, &registry, "Hi")
            .collect()
            .await;
        assert!(
            matches!(events.last().unwrap(), Ok(StreamEvent::Completed { response }) if response.message.content.text() == "Hello")
        );
        assert_eq!(session.messages().len(), 3);
        let events: Vec<_> = provider
            .send_request_stream(provider.build_request(&session.messages(), vec![]))
            .collect()
            .await;
        assert!(matches!(
            events.last().unwrap(),
            Ok(StreamEvent::Completed { .. })
        ));
        mock.assert_hits(3);
    }
}
