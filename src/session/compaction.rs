use std::collections::{HashMap, HashSet};

use crate::chat::MessageRole;
use crate::session::SessionTurn;
use crate::tools::compact::is_compaction_tool;
use crate::{Error, Result};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompactionConfig {
    pub enabled: bool,
    pub trigger_estimated_tokens: usize,
    pub target_estimated_tokens: usize,
    pub max_compaction_rounds: usize,
    pub always_offer_tools: bool,
    pub protect_recent_turns: usize,
    pub top_consumers: usize,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            trigger_estimated_tokens: 200_000,
            target_estimated_tokens: 160_000,
            max_compaction_rounds: 4,
            always_offer_tools: true,
            protect_recent_turns: 4,
            top_consumers: 30,
        }
    }
}

impl CompactionConfig {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContextUsage {
    pub turn_id: u64,
    pub role: MessageRole,
    pub name: Option<String>,
    pub bytes: usize,
    pub estimated_tokens: usize,
    pub summary: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolCallUsage {
    pub tool_call_id: String,
    pub name: String,
    pub request_turn_id: u64,
    pub result_turn_ids: Vec<u64>,
    pub bytes: usize,
    pub estimated_tokens: usize,
    pub arguments_preview: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ContextUsageBreakdown {
    pub total_bytes: usize,
    pub total_estimated_tokens: usize,
    pub largest_turns: Vec<ContextUsage>,
    pub largest_tool_calls: Vec<ToolCallUsage>,
}

#[derive(Default)]
struct CompactionEdit {
    remove_turn_ids: HashSet<u64>,
    remove_tool_call_ids: HashSet<String>,
}

pub(crate) fn usage_breakdown(turns: &[SessionTurn], limit: usize) -> ContextUsageBreakdown {
    let mut largest_turns = turns.iter().map(turn_usage).collect::<Vec<_>>();
    largest_turns.sort_by(|a, b| b.estimated_tokens.cmp(&a.estimated_tokens));
    if limit < largest_turns.len() {
        largest_turns.truncate(limit);
    }

    let mut largest_tool_calls = tool_call_usages(turns);
    largest_tool_calls.sort_by(|a, b| b.estimated_tokens.cmp(&a.estimated_tokens));
    if limit < largest_tool_calls.len() {
        largest_tool_calls.truncate(limit);
    }

    let total_bytes = turns.iter().map(estimate_turn_bytes).sum();
    let total_estimated_tokens =
        estimate_tokens_from_chars(turns.iter().map(estimate_turn_chars).sum());

    ContextUsageBreakdown {
        total_bytes,
        total_estimated_tokens,
        largest_turns,
        largest_tool_calls,
    }
}

pub(crate) fn remove_messages(
    turns: &mut Vec<SessionTurn>,
    config: &CompactionConfig,
    turn_ids: &[u64],
    reason: &str,
) -> Result<String> {
    let edit = compaction_edit(turns, config, turn_ids)?;
    let removed = apply_compaction_edit(turns, edit, None);
    Ok(format!(
        "Removed {removed} turn(s) from active context. Reason: {reason}"
    ))
}

pub(crate) fn summarize_messages(
    turns: &mut Vec<SessionTurn>,
    next_turn_id: &mut u64,
    config: &CompactionConfig,
    turn_ids: &[u64],
    summary: &str,
    reason: &str,
) -> Result<String> {
    if summary.trim().is_empty() {
        return Err(Error::Tool(
            "compact_summarize_messages requires a non-empty summary".to_string(),
        ));
    }
    let edit = compaction_edit(turns, config, turn_ids)?;
    let insert_at = turns
        .iter()
        .position(|turn| edit.remove_turn_ids.contains(&turn.id))
        .unwrap_or(turns.len());
    let summary_turn = SessionTurn {
        id: allocate_turn_id(next_turn_id),
        role: MessageRole::System,
        content: format!(
            "Compacted context summary: {}\nCompaction reason: {}",
            summary.trim(),
            reason.trim()
        ),
        reasoning: None,
        tool_calls: Vec::new(),
        tool_call_id: None,
        name: Some("context_summary".to_string()),
    };
    let removed = apply_compaction_edit(turns, edit, Some((insert_at, summary_turn)));
    Ok(format!(
        "Summarized {removed} turn(s) in active context. Reason: {reason}"
    ))
}

pub(crate) fn remove_previous_tool_messages(turns: &mut Vec<SessionTurn>) -> usize {
    let mut remove_tool_call_ids = HashSet::new();
    let mut remove_turn_ids = HashSet::new();

    for turn in turns.iter() {
        for call in &turn.tool_calls {
            if is_compaction_tool(&call.function.name) {
                remove_tool_call_ids.insert(call.id.clone());
                remove_turn_ids.insert(turn.id);
            }
        }
        if turn.role == MessageRole::Tool && turn.name.as_deref().is_some_and(is_compaction_tool) {
            remove_turn_ids.insert(turn.id);
            if let Some(tool_call_id) = &turn.tool_call_id {
                remove_tool_call_ids.insert(tool_call_id.clone());
            }
        }
    }

    if remove_turn_ids.is_empty() && remove_tool_call_ids.is_empty() {
        return 0;
    }

    apply_compaction_edit(
        turns,
        CompactionEdit {
            remove_turn_ids,
            remove_tool_call_ids,
        },
        None,
    )
}

pub(crate) fn instruction(config: &CompactionConfig, breakdown: &ContextUsageBreakdown) -> String {
    format!(
        "Context compaction is required before answering the user.\n\
         Reuse this existing conversation and system message; do not create a separate context.\n\
         Use only the built-in compaction tools to reduce the active context toward about {} estimated tokens.\n\
         Prefer trimming old read/ripgrep tool results, stale searches, and obsolete file dumps.\n\
         Summarize message sets when their conclusions are still useful, and remove them entirely when irrelevant.\n\
         Never remove the root system prompt or recent conversation.\n\n\
         Current estimated context: {} tokens, {} bytes.\n\n\
         Largest turns:\n{}\n\n\
         Largest tool calls:\n{}",
        config.target_estimated_tokens,
        breakdown.total_estimated_tokens,
        breakdown.total_bytes,
        format_turn_breakdown(&breakdown.largest_turns),
        format_tool_breakdown(&breakdown.largest_tool_calls),
    )
}

fn compaction_edit(
    turns: &[SessionTurn],
    config: &CompactionConfig,
    turn_ids: &[u64],
) -> Result<CompactionEdit> {
    if turn_ids.is_empty() {
        return Err(Error::Tool(
            "compaction requires at least one turn_id".to_string(),
        ));
    }

    let protected = protected_turn_ids(turns, config);
    let known = turns.iter().map(|turn| turn.id).collect::<HashSet<_>>();
    let mut edit = CompactionEdit::default();

    for turn_id in turn_ids {
        if !known.contains(turn_id) {
            return Err(Error::Tool(format!("unknown turn_id {turn_id}")));
        }
        if protected.contains(turn_id) {
            return Err(Error::Tool(format!("turn_id {turn_id} is protected")));
        }
        add_turn_to_compaction_edit(turns, *turn_id, &mut edit);
    }

    Ok(edit)
}

fn protected_turn_ids(turns: &[SessionTurn], config: &CompactionConfig) -> HashSet<u64> {
    let mut protected = HashSet::new();
    if let Some(first) = turns.first()
        && first.role == MessageRole::System
    {
        protected.insert(first.id);
    }
    for turn in turns.iter().rev().take(config.protect_recent_turns) {
        protected.insert(turn.id);
    }
    protected
}

fn add_turn_to_compaction_edit(turns: &[SessionTurn], turn_id: u64, edit: &mut CompactionEdit) {
    let Some(turn) = turns.iter().find(|turn| turn.id == turn_id) else {
        return;
    };

    if turn.role == MessageRole::Tool {
        edit.remove_turn_ids.insert(turn.id);
        if let Some(tool_call_id) = &turn.tool_call_id {
            edit.remove_tool_call_ids.insert(tool_call_id.clone());
        }
        return;
    }

    edit.remove_turn_ids.insert(turn.id);
    for call in &turn.tool_calls {
        edit.remove_tool_call_ids.insert(call.id.clone());
        for result in turns
            .iter()
            .filter(|candidate| candidate.tool_call_id.as_deref() == Some(call.id.as_str()))
        {
            edit.remove_turn_ids.insert(result.id);
        }
    }
}

fn apply_compaction_edit(
    turns: &mut Vec<SessionTurn>,
    edit: CompactionEdit,
    insert: Option<(usize, SessionTurn)>,
) -> usize {
    let before = turns.len();
    let mut updated = Vec::with_capacity(turns.len() + usize::from(insert.is_some()));
    let insert_at = insert.as_ref().map(|(index, _)| *index);
    let mut inserted = false;

    for (index, mut turn) in turns.drain(..).enumerate() {
        if Some(index) == insert_at
            && let Some((_, summary)) = &insert
        {
            updated.push(summary.clone());
            inserted = true;
        }

        if edit.remove_turn_ids.contains(&turn.id) {
            continue;
        }

        if !edit.remove_tool_call_ids.is_empty() {
            turn.tool_calls
                .retain(|call| !edit.remove_tool_call_ids.contains(&call.id));
            if turn.role == MessageRole::Assistant
                && turn.content.trim().is_empty()
                && turn.tool_calls.is_empty()
            {
                continue;
            }
        }

        updated.push(turn);
    }

    if !inserted && let Some((_, summary)) = insert {
        updated.push(summary);
    }

    let removed = before.saturating_sub(updated.len());
    *turns = updated;
    removed
}

fn tool_call_usages(turns: &[SessionTurn]) -> Vec<ToolCallUsage> {
    let mut results_by_call = HashMap::<String, Vec<&SessionTurn>>::new();
    for turn in turns {
        if turn.role == MessageRole::Tool
            && let Some(tool_call_id) = &turn.tool_call_id
        {
            results_by_call
                .entry(tool_call_id.clone())
                .or_default()
                .push(turn);
        }
    }

    let mut usages = Vec::new();
    for turn in turns {
        for call in &turn.tool_calls {
            let result_turns = results_by_call.get(&call.id).cloned().unwrap_or_default();
            let request_bytes = call.function.name.len() + call.function.arguments.len();
            let request_chars =
                call.function.name.chars().count() + call.function.arguments.chars().count();
            let result_bytes = result_turns
                .iter()
                .map(|turn| estimate_turn_bytes(turn))
                .sum::<usize>();
            let result_chars = result_turns
                .iter()
                .map(|turn| estimate_turn_chars(turn))
                .sum::<usize>();
            let bytes = request_bytes + result_bytes;
            usages.push(ToolCallUsage {
                tool_call_id: call.id.clone(),
                name: call.function.name.clone(),
                request_turn_id: turn.id,
                result_turn_ids: result_turns.iter().map(|turn| turn.id).collect(),
                bytes,
                estimated_tokens: estimate_tokens_from_chars(request_chars + result_chars),
                arguments_preview: preview(&call.function.arguments, 180),
            });
        }
    }
    usages
}

fn format_turn_breakdown(turns: &[ContextUsage]) -> String {
    if turns.is_empty() {
        return "- none".to_string();
    }
    turns
        .iter()
        .map(|usage| {
            format!(
                "- turn_id={} role={:?} name={} estimated_tokens={} bytes={} summary={}",
                usage.turn_id,
                usage.role,
                usage.name.as_deref().unwrap_or("-"),
                usage.estimated_tokens,
                usage.bytes,
                usage.summary
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_tool_breakdown(calls: &[ToolCallUsage]) -> String {
    if calls.is_empty() {
        return "- none".to_string();
    }
    calls
        .iter()
        .map(|usage| {
            format!(
                "- tool_call_id={} name={} request_turn_id={} result_turn_ids={:?} estimated_tokens={} bytes={} args={}",
                usage.tool_call_id,
                usage.name,
                usage.request_turn_id,
                usage.result_turn_ids,
                usage.estimated_tokens,
                usage.bytes,
                usage.arguments_preview
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn turn_usage(turn: &SessionTurn) -> ContextUsage {
    let bytes = estimate_turn_bytes(turn);
    ContextUsage {
        turn_id: turn.id,
        role: turn.role.clone(),
        name: turn.name.clone(),
        bytes,
        estimated_tokens: estimate_tokens_from_chars(estimate_turn_chars(turn)),
        summary: preview(&turn.content, 180),
    }
}

fn estimate_turn_bytes(turn: &SessionTurn) -> usize {
    let mut bytes = turn.content.len();
    if let Some(reasoning) = &turn.reasoning {
        if let Some(reasoning) = &reasoning.reasoning {
            bytes += reasoning.to_string().len();
        }
        if let Some(reasoning_content) = &reasoning.reasoning_content {
            bytes += reasoning_content.len();
        }
        if let Some(reasoning_details) = &reasoning.reasoning_details {
            bytes += reasoning_details.to_string().len();
        }
    }
    for call in &turn.tool_calls {
        bytes += call.id.len() + call.kind.len() + call.function.name.len();
        bytes += call.function.arguments.len();
    }
    if let Some(tool_call_id) = &turn.tool_call_id {
        bytes += tool_call_id.len();
    }
    if let Some(name) = &turn.name {
        bytes += name.len();
    }
    bytes
}

fn estimate_turn_chars(turn: &SessionTurn) -> usize {
    let mut chars = turn.content.chars().count();
    if let Some(reasoning) = &turn.reasoning {
        if let Some(reasoning) = &reasoning.reasoning {
            chars += reasoning.to_string().chars().count();
        }
        if let Some(reasoning_content) = &reasoning.reasoning_content {
            chars += reasoning_content.chars().count();
        }
        if let Some(reasoning_details) = &reasoning.reasoning_details {
            chars += reasoning_details.to_string().chars().count();
        }
    }
    for call in &turn.tool_calls {
        chars += call.id.chars().count() + call.kind.chars().count();
        chars += call.function.name.chars().count();
        chars += call.function.arguments.chars().count();
    }
    if let Some(tool_call_id) = &turn.tool_call_id {
        chars += tool_call_id.chars().count();
    }
    if let Some(name) = &turn.name {
        chars += name.chars().count();
    }
    chars
}

fn estimate_tokens_from_chars(chars: usize) -> usize {
    chars.saturating_add(3) / 4
}

fn allocate_turn_id(next_turn_id: &mut u64) -> u64 {
    let id = *next_turn_id;
    *next_turn_id = next_turn_id.saturating_add(1).max(1);
    id
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
