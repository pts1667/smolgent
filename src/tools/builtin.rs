use std::path::{Path, PathBuf};
use std::process::Command;

use schemars::JsonSchema;
use serde::Deserialize;

use crate::state::AgentState;
use crate::tools::{Tool, ToolDefinition};
use crate::{Error, Result};

const READ_DESCRIPTION: &str = r#"Read a UTF-8 text file from an allowed read root.

Required parameters:
- path: file path to read. Relative paths are resolved from the current process directory.

Example:
{"path":"src/lib.rs"}"#;

const APPLY_PATCH_DESCRIPTION: &str = r#"Apply a small patch to files under allowed write roots.

Required parameters:
- patch: a patch string using this format:
  *** Begin Patch
  *** Add File: path
  +new line
  *** Update File: path
  @@
   unchanged context line
  -old line
  +new line
  *** Delete File: path
  *** End Patch

Every touched file must be under an allowed write root. Update hunks use exact line matching, so include nearby context lines when possible.

Example:
{"patch":"*** Begin Patch\n*** Update File: README.md\n@@\n-Old text\n+New text\n*** End Patch\n"}"#;

const RIPGREP_DESCRIPTION: &str = r#"Search allowed read roots with ripgrep.

Required parameters:
- pattern: regex or literal text to search for.
- paths: one or more files/directories to search. Each path must be under an allowed read root.

Optional parameters:
- fixed_strings: true to pass --fixed-strings for literal matching.
- ignore_case: true to pass --ignore-case.
- smart_case: true to pass --smart-case.
- line_number: true to pass --line-number.
- context: number of context lines, passed as --context.
- glob: glob filters, passed as --glob. May include exclusions like "!target/**".
- max_count: maximum matching lines per file, passed as --max-count.
- files_with_matches: true to print only files with matches.
- files: true to list searchable files instead of searching for pattern.

Examples:
{"pattern":"ToolRegistry","paths":["src"],"line_number":true}
{"pattern":"TODO","paths":["src","tests"],"fixed_strings":true,"glob":["*.rs"]}"#;

#[derive(Clone, Debug, Deserialize, JsonSchema)]
pub struct ReadArgs {
    /// File path to read. Relative paths are resolved from the current process directory.
    pub path: PathBuf,
}

#[derive(Clone, Debug, Deserialize, JsonSchema)]
pub struct ApplyPatchArgs {
    /// Patch text in the documented begin/end patch format.
    pub patch: String,
}

#[derive(Clone, Debug, Default, Deserialize, JsonSchema)]
pub struct RgArgs {
    /// Regex pattern to search for. Optional only when `files` is true.
    pub pattern: Option<String>,
    /// Files or directories to search. Required; every path must be under an allowed read root.
    pub paths: Vec<PathBuf>,
    /// Treat the pattern as a literal string instead of a regex.
    #[serde(default)]
    pub fixed_strings: bool,
    /// Search case-insensitively.
    #[serde(default)]
    pub ignore_case: bool,
    /// Search case-insensitively only if the pattern is all lowercase.
    #[serde(default)]
    pub smart_case: bool,
    /// Print 1-based line numbers.
    #[serde(default)]
    pub line_number: bool,
    /// Number of context lines before and after matches.
    pub context: Option<u32>,
    /// Glob filters passed to ripgrep, e.g. "*.rs" or "!target/**".
    #[serde(default)]
    pub glob: Vec<String>,
    /// Maximum matching lines per file.
    pub max_count: Option<u32>,
    /// Print only paths with at least one match.
    #[serde(default)]
    pub files_with_matches: bool,
    /// List files ripgrep would search instead of searching for a pattern.
    #[serde(default)]
    pub files: bool,
}

pub fn read_tool(state: AgentState) -> Tool {
    Tool::new(
        ToolDefinition::new("read", READ_DESCRIPTION, schemars::schema_for!(ReadArgs)),
        move |arguments| {
            let state = state.clone();
            Box::pin(async move {
                let args: ReadArgs = serde_json::from_value(arguments)?;
                read(&state, args).map(serde_json::Value::String)
            })
        },
    )
}

pub fn apply_patch_tool(state: AgentState) -> Tool {
    Tool::new(
        ToolDefinition::new(
            "apply_patch",
            APPLY_PATCH_DESCRIPTION,
            schemars::schema_for!(ApplyPatchArgs),
        ),
        move |arguments| {
            let state = state.clone();
            Box::pin(async move {
                let args: ApplyPatchArgs = serde_json::from_value(arguments)?;
                apply_patch(&state, args).map(serde_json::Value::String)
            })
        },
    )
}

pub fn ripgrep_tool(state: AgentState) -> Tool {
    Tool::new(
        ToolDefinition::new(
            "ripgrep",
            RIPGREP_DESCRIPTION,
            schemars::schema_for!(RgArgs),
        ),
        move |arguments| {
            let state = state.clone();
            Box::pin(async move {
                let args: RgArgs = serde_json::from_value(arguments)?;
                ripgrep(&state, args).map(serde_json::Value::String)
            })
        },
    )
}

pub fn builtin_registry(state: AgentState) -> crate::ToolRegistry {
    crate::ToolRegistry::new()
        .with_tool(read_tool(state.clone()))
        .with_tool(apply_patch_tool(state.clone()))
        .with_tool(ripgrep_tool(state))
}

fn read(state: &AgentState, args: ReadArgs) -> Result<String> {
    ensure_can_read(state, &args.path)?;
    Ok(std::fs::read_to_string(tool_path(&args.path))?)
}

fn ripgrep(state: &AgentState, args: RgArgs) -> Result<String> {
    if args.paths.is_empty() {
        return Err(Error::Tool(
            "ripgrep requires at least one path under an allowed read root".to_string(),
        ));
    }
    if !args.files && args.pattern.as_deref().unwrap_or_default().is_empty() {
        return Err(Error::Tool(
            "ripgrep requires `pattern` unless `files` is true".to_string(),
        ));
    }
    for path in &args.paths {
        ensure_can_read(state, path)?;
    }

    let mut command = Command::new("rg");
    command.arg("--no-config");
    if args.files {
        command.arg("--files");
    } else {
        if args.fixed_strings {
            command.arg("--fixed-strings");
        }
        if args.ignore_case {
            command.arg("--ignore-case");
        }
        if args.smart_case {
            command.arg("--smart-case");
        }
        if args.line_number {
            command.arg("--line-number");
        }
        if args.files_with_matches {
            command.arg("--files-with-matches");
        }
        if let Some(context) = args.context {
            command.arg("--context").arg(context.to_string());
        }
        if let Some(max_count) = args.max_count {
            command.arg("--max-count").arg(max_count.to_string());
        }
        for glob in &args.glob {
            command.arg("--glob").arg(glob);
        }
        command.arg(args.pattern.unwrap_or_default());
    }
    command.args(args.paths.iter().map(|path| tool_path(path)));

    let output = command.output()?;
    let stdout = sanitize_tool_output(&String::from_utf8_lossy(&output.stdout));
    let stderr = sanitize_tool_output(&String::from_utf8_lossy(&output.stderr));
    if output.status.success() {
        return Ok(stdout);
    }
    if output.status.code() == Some(1) {
        return Ok(format!("{stdout}{stderr}"));
    }
    Err(Error::Tool(format!(
        "ripgrep failed with status {}: {}",
        output.status, stderr
    )))
}

fn apply_patch(state: &AgentState, args: ApplyPatchArgs) -> Result<String> {
    let operations = parse_patch(&args.patch)?;
    if operations.is_empty() {
        return Err(Error::Tool("patch contained no operations".to_string()));
    }

    for operation in &operations {
        ensure_can_write(state, operation.path())?;
    }

    let mut changed = Vec::new();
    for operation in operations {
        match operation {
            PatchOperation::Add { path, lines } => {
                let fs_path = tool_path(&path);
                if fs_path.exists() {
                    return Err(Error::Tool(format!(
                        "cannot add file that already exists: {}",
                        display_path(&path)
                    )));
                }
                if let Some(parent) = fs_path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(&fs_path, lines_to_text(&lines))?;
                changed.push(format!("added {}", display_path(&path)));
            }
            PatchOperation::Delete { path } => {
                std::fs::remove_file(tool_path(&path))?;
                changed.push(format!("deleted {}", display_path(&path)));
            }
            PatchOperation::Update { path, hunks } => {
                let fs_path = tool_path(&path);
                let original = std::fs::read_to_string(&fs_path)?;
                let updated = apply_hunks(&original, &hunks)?;
                std::fs::write(&fs_path, updated)?;
                changed.push(format!("updated {}", display_path(&path)));
            }
        }
    }

    Ok(changed.join("\n"))
}

fn ensure_can_read(state: &AgentState, path: &Path) -> Result<()> {
    if state.can_read(path) {
        Ok(())
    } else {
        Err(Error::PathNotAllowed {
            path: display_path(path),
            access: "read",
        })
    }
}

fn ensure_can_write(state: &AgentState, path: &Path) -> Result<()> {
    if state.can_write(path) {
        Ok(())
    } else {
        Err(Error::PathNotAllowed {
            path: display_path(path),
            access: "write",
        })
    }
}

fn tool_path(path: &Path) -> PathBuf {
    strip_windows_verbatim_prefix(path)
}

fn display_path(path: &Path) -> String {
    strip_windows_verbatim_prefix(path).display().to_string()
}

fn sanitize_tool_output(output: &str) -> String {
    output.replace(r"\\?\UNC\", r"\\").replace(r"\\?\", "")
}

fn strip_windows_verbatim_prefix(path: &Path) -> PathBuf {
    let text = path.to_string_lossy();
    if let Some(rest) = text.strip_prefix(r"\\?\UNC\") {
        PathBuf::from(format!(r"\\{rest}"))
    } else if let Some(rest) = text.strip_prefix(r"\\?\") {
        PathBuf::from(rest)
    } else {
        path.to_path_buf()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum PatchOperation {
    Add { path: PathBuf, lines: Vec<String> },
    Delete { path: PathBuf },
    Update { path: PathBuf, hunks: Vec<HunkLine> },
}

impl PatchOperation {
    fn path(&self) -> &Path {
        match self {
            PatchOperation::Add { path, .. }
            | PatchOperation::Delete { path }
            | PatchOperation::Update { path, .. } => path,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum HunkLine {
    Context(String),
    Remove(String),
    Add(String),
}

fn parse_patch(patch: &str) -> Result<Vec<PatchOperation>> {
    let lines = patch.lines().collect::<Vec<_>>();
    if lines.first() != Some(&"*** Begin Patch") || lines.last() != Some(&"*** End Patch") {
        return Err(Error::Tool(
            "patch must start with `*** Begin Patch` and end with `*** End Patch`".to_string(),
        ));
    }

    let mut operations = Vec::new();
    let mut index = 1;
    while index < lines.len() - 1 {
        let line = lines[index];
        if let Some(path) = line.strip_prefix("*** Add File: ") {
            index += 1;
            let mut add_lines = Vec::new();
            while index < lines.len() - 1 && !lines[index].starts_with("*** ") {
                let Some(content) = lines[index].strip_prefix('+') else {
                    return Err(Error::Tool(
                        "add file lines must start with `+`".to_string(),
                    ));
                };
                add_lines.push(content.to_string());
                index += 1;
            }
            operations.push(PatchOperation::Add {
                path: PathBuf::from(path),
                lines: add_lines,
            });
        } else if let Some(path) = line.strip_prefix("*** Delete File: ") {
            operations.push(PatchOperation::Delete {
                path: PathBuf::from(path),
            });
            index += 1;
        } else if let Some(path) = line.strip_prefix("*** Update File: ") {
            index += 1;
            let mut hunk_lines = Vec::new();
            while index < lines.len() - 1 && !lines[index].starts_with("*** ") {
                let line = lines[index];
                if line == "@@" || line.starts_with("@@ ") {
                    index += 1;
                    continue;
                }
                if line.is_empty() {
                    return Err(Error::Tool("empty patch hunk line".to_string()));
                }
                let (prefix, content) = line.split_at(1);
                match prefix {
                    " " => hunk_lines.push(HunkLine::Context(content.to_string())),
                    "-" => hunk_lines.push(HunkLine::Remove(content.to_string())),
                    "+" => hunk_lines.push(HunkLine::Add(content.to_string())),
                    _ => {
                        return Err(Error::Tool(format!(
                            "unsupported update hunk line prefix `{prefix}`"
                        )));
                    }
                }
                index += 1;
            }
            operations.push(PatchOperation::Update {
                path: PathBuf::from(path),
                hunks: hunk_lines,
            });
        } else if line.trim().is_empty() {
            index += 1;
        } else {
            return Err(Error::Tool(format!("unsupported patch operation `{line}`")));
        }
    }

    Ok(operations)
}

fn apply_hunks(original: &str, hunk_lines: &[HunkLine]) -> Result<String> {
    let had_trailing_newline = original.ends_with('\n');
    let original_lines = original.lines().map(str::to_string).collect::<Vec<_>>();
    let mut output = Vec::new();
    let mut cursor = 0;
    let mut index = 0;

    while index < hunk_lines.len() {
        let mut expected = Vec::new();
        while index < hunk_lines.len() {
            match &hunk_lines[index] {
                HunkLine::Context(line) | HunkLine::Remove(line) => expected.push(line.clone()),
                HunkLine::Add(_) => {}
            }
            index += 1;
        }

        let start = find_sequence(&original_lines, cursor, &expected)
            .ok_or_else(|| Error::Tool("update hunk did not match the target file".to_string()))?;
        output.extend_from_slice(&original_lines[cursor..start]);

        let mut expected_index = 0;
        for line in hunk_lines {
            match line {
                HunkLine::Context(context) => {
                    output.push(context.clone());
                    expected_index += 1;
                }
                HunkLine::Remove(_) => {
                    expected_index += 1;
                }
                HunkLine::Add(addition) => output.push(addition.clone()),
            }
        }
        cursor = start + expected_index;
    }

    output.extend_from_slice(&original_lines[cursor..]);
    let mut text = output.join("\n");
    if had_trailing_newline || !text.is_empty() {
        text.push('\n');
    }
    Ok(text)
}

fn find_sequence(lines: &[String], start: usize, expected: &[String]) -> Option<usize> {
    if expected.is_empty() {
        return Some(start);
    }
    (start..=lines.len().saturating_sub(expected.len()))
        .find(|&index| lines[index..index + expected.len()] == *expected)
}

fn lines_to_text(lines: &[String]) -> String {
    let mut text = lines.join("\n");
    if !lines.is_empty() {
        text.push('\n');
    }
    text
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::tools::ToolRegistry;

    #[tokio::test]
    async fn read_tool_validates_read_roots() {
        let temp = tempfile::tempdir().unwrap();
        let allowed = temp.path().join("allowed");
        let denied = temp.path().join("denied");
        std::fs::create_dir_all(&allowed).unwrap();
        std::fs::create_dir_all(&denied).unwrap();
        let file = allowed.join("note.txt");
        std::fs::write(&file, "hello").unwrap();
        let denied_file = denied.join("secret.txt");
        std::fs::write(&denied_file, "nope").unwrap();

        let registry = ToolRegistry::new().with_tool(read_tool(AgentState::new([allowed], [])));
        assert_eq!(
            registry
                .get("read")
                .unwrap()
                .call(json!({ "path": file }))
                .await
                .unwrap(),
            json!("hello")
        );
        assert!(
            registry
                .get("read")
                .unwrap()
                .call(json!({ "path": denied_file }))
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn ripgrep_tool_searches_allowed_paths() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("lib.rs"), "pub struct ToolRegistry;\n").unwrap();
        let registry =
            ToolRegistry::new().with_tool(ripgrep_tool(AgentState::new([temp.path().into()], [])));

        let output = registry
            .get("ripgrep")
            .unwrap()
            .call(json!({
                "pattern": "ToolRegistry",
                "paths": [temp.path()],
                "line_number": true
            }))
            .await
            .unwrap();

        assert!(output.as_str().unwrap().contains("ToolRegistry"));
    }

    #[tokio::test]
    async fn apply_patch_tool_validates_write_roots_and_updates_files() {
        let temp = tempfile::tempdir().unwrap();
        let file = temp.path().join("README.md");
        std::fs::write(&file, "Old text\n").unwrap();
        let registry = ToolRegistry::new()
            .with_tool(apply_patch_tool(AgentState::new([], [temp.path().into()])));

        registry
            .get("apply_patch")
            .unwrap()
            .call(json!({
                "patch": format!(
                    "*** Begin Patch\n*** Update File: {}\n@@\n-Old text\n+New text\n*** End Patch\n",
                    file.display()
                )
            }))
            .await
            .unwrap();

        assert_eq!(std::fs::read_to_string(file).unwrap(), "New text\n");
    }

    #[test]
    fn builtin_tool_definitions_explain_parameters() {
        let state = AgentState::default();
        let read = read_tool(state.clone());
        let patch = apply_patch_tool(state.clone());
        let rg = ripgrep_tool(state);

        assert!(read.definition().function.description.contains("Example"));
        assert!(
            patch
                .definition()
                .function
                .description
                .contains("Update File")
        );
        assert!(
            rg.definition()
                .function
                .description
                .contains("fixed_strings")
        );
        assert!(
            rg.definition().function.parameters["properties"]
                .as_object()
                .unwrap()
                .contains_key("pattern")
        );
    }
}
