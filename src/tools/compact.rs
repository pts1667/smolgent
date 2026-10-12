use schemars::JsonSchema;
use serde::{Deserialize, Deserializer};

use crate::tools::ToolDefinition;
use crate::{Error, Result};

pub const COMPACT_REMOVE_MESSAGES: &str = "compact_remove_messages";
pub const COMPACT_SUMMARIZE_MESSAGES: &str = "compact_summarize_messages";

#[smolgent::tool_definition(
    name = COMPACT_REMOVE_MESSAGES,
    description = "Remove old or irrelevant messages from the active context. Use this mainly for stale read/ripgrep tool results. When removing tool results, the matching assistant tool request is also removed or edited so the transcript stays valid. Do not remove the root system prompt or recent conversation.",
    function = "remove_messages_definition"
)]
#[derive(Deserialize, JsonSchema)]
pub(crate) struct RemoveMessagesArgs {
    /// Stable turn IDs to remove from the active context.
    #[serde(deserialize_with = "one_or_many_ids")]
    pub turn_ids: Vec<u64>,
    /// Brief reason these turns are safe to remove.
    #[serde(default)]
    pub reason: String,
}

#[smolgent::tool_definition(
    name = COMPACT_SUMMARIZE_MESSAGES,
    description = "Replace old messages with one concise context summary. Use this for older discussion whose conclusions still matter. When summarizing tool results, include the useful facts and paths, not raw file dumps.",
    function = "summarize_messages_definition"
)]
#[derive(Deserialize, JsonSchema)]
pub(crate) struct SummarizeMessagesArgs {
    /// Stable turn IDs to summarize and remove from the active context.
    #[serde(deserialize_with = "one_or_many_ids")]
    pub turn_ids: Vec<u64>,
    /// Concise replacement summary preserving facts still needed later.
    pub summary: String,
    /// Brief reason summarization is appropriate.
    #[serde(default)]
    pub reason: String,
}

/// Small models often send `"turn_ids": 5` or `["5"]` instead of `[5]`; accept those too.
fn one_or_many_ids<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Vec<u64>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Id {
        Number(u64),
        Text(String),
    }
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Ids {
        One(Id),
        Many(Vec<Id>),
    }
    let ids = match Ids::deserialize(deserializer)? {
        Ids::One(id) => vec![id],
        Ids::Many(ids) => ids,
    };
    ids.into_iter()
        .map(|id| match id {
            Id::Number(n) => Ok(n),
            Id::Text(text) => text
                .trim()
                .trim_start_matches("turn_id=")
                .parse()
                .map_err(|_| serde::de::Error::custom(format!("turn_id {text:?} is not a number"))),
        })
        .collect()
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
        remove_messages_definition(),
        summarize_messages_definition(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_definitions_include_argument_rustdoc() {
        let definitions = definitions();
        let remove = definitions
            .iter()
            .find(|definition| definition.name() == COMPACT_REMOVE_MESSAGES)
            .unwrap();

        assert_eq!(
            remove.function.parameters["properties"]["turn_ids"]["description"],
            "Stable turn IDs to remove from the active context."
        );
        assert_eq!(
            remove.function.parameters["properties"]["turn_ids"]["type"],
            "array"
        );
    }

    #[test]
    fn turn_ids_accept_one_id_and_numeric_strings() {
        assert_eq!(
            parse_remove_args(r#"{"turn_ids": 7}"#).unwrap().turn_ids,
            vec![7]
        );
        assert_eq!(
            parse_remove_args(r#"{"turn_ids": [2, "3", "turn_id=4"]}"#)
                .unwrap()
                .turn_ids,
            vec![2, 3, 4]
        );
        assert!(parse_remove_args(r#"{"turn_ids": ["old"]}"#).is_err());
        let summarize = parse_summarize_args(r#"{"turn_ids": "5", "summary": "s"}"#).unwrap();
        assert_eq!(summarize.turn_ids, vec![5]);
    }
}
