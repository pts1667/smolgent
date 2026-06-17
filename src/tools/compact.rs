use serde::Deserialize;
use serde_json::json;

use crate::tools::ToolDefinition;
use crate::{Error, Result};

pub const COMPACT_REMOVE_MESSAGES: &str = "compact_remove_messages";
pub const COMPACT_SUMMARIZE_MESSAGES: &str = "compact_summarize_messages";

#[derive(Deserialize)]
pub(crate) struct RemoveMessagesArgs {
    pub turn_ids: Vec<u64>,
    #[serde(default)]
    pub reason: String,
}

#[derive(Deserialize)]
pub(crate) struct SummarizeMessagesArgs {
    pub turn_ids: Vec<u64>,
    pub summary: String,
    #[serde(default)]
    pub reason: String,
}

pub(crate) fn parse_remove_args(arguments: &str) -> Result<RemoveMessagesArgs> {
    if arguments.trim().is_empty() {
        return Err(Error::Tool(
            "compact_remove_messages requires arguments".to_string(),
        ));
    }
    Ok(serde_json::from_str(arguments)?)
}

pub(crate) fn parse_summarize_args(arguments: &str) -> Result<SummarizeMessagesArgs> {
    if arguments.trim().is_empty() {
        return Err(Error::Tool(
            "compact_summarize_messages requires arguments".to_string(),
        ));
    }
    Ok(serde_json::from_str(arguments)?)
}

pub(crate) fn is_compaction_tool(name: &str) -> bool {
    matches!(name, COMPACT_REMOVE_MESSAGES | COMPACT_SUMMARIZE_MESSAGES)
}

pub(crate) fn acknowledgement(name: &str, detail: &str) -> String {
    let action = match name {
        COMPACT_REMOVE_MESSAGES => "removed selected context",
        COMPACT_SUMMARIZE_MESSAGES => "summarized selected context",
        _ => "updated context",
    };
    format!(
        "Compaction acknowledged: {action}. {}",
        preview(detail, 160)
    )
}

pub(crate) fn definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition::new(
            COMPACT_REMOVE_MESSAGES,
            "Remove old or irrelevant messages from the active context. Use this mainly for stale read/ripgrep tool results. When removing tool results, the matching assistant tool request is also removed or edited so the transcript stays valid. Do not remove the root system prompt or recent conversation.",
            json!({
                "type": "object",
                "properties": {
                    "turn_ids": {
                        "type": "array",
                        "items": { "type": "integer" },
                        "description": "Stable turn IDs to remove from the active context."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief reason these turns are safe to remove."
                    }
                },
                "required": ["turn_ids", "reason"]
            }),
        ),
        ToolDefinition::new(
            COMPACT_SUMMARIZE_MESSAGES,
            "Replace old messages with one concise context summary. Use this for older discussion whose conclusions still matter. When summarizing tool results, include the useful facts and paths, not raw file dumps.",
            json!({
                "type": "object",
                "properties": {
                    "turn_ids": {
                        "type": "array",
                        "items": { "type": "integer" },
                        "description": "Stable turn IDs to summarize and remove from the active context."
                    },
                    "summary": {
                        "type": "string",
                        "description": "Concise replacement summary preserving facts still needed later."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief reason summarization is appropriate."
                    }
                },
                "required": ["turn_ids", "summary", "reason"]
            }),
        ),
    ]
}

fn preview(text: &str, limit: usize) -> String {
    let mut preview = String::new();
    for ch in text.chars().take(limit) {
        if ch.is_control() && ch != '\n' && ch != '\t' {
            continue;
        }
        preview.push(ch);
    }
    if text.chars().count() > limit {
        preview.push_str("...");
    }
    preview.replace('\n', "\\n")
}
