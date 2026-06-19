use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Command;

use schemars::JsonSchema;
use serde::Deserialize;

use crate::state::AgentState;
use crate::tools::{Tool, ToolDefinition};
use crate::{Error, Result};

const READ_DESCRIPTION: &str = r#"Read a UTF-8 text file from an allowed read root.

Output limits:
- At most 100 lines or 10,000 characters are returned, whichever is reached first.
- character-offset: optional zero-based Unicode character offset, applied after line-offset.
- line-offset: optional zero-based number of complete lines to skip first.
- character-count: optional requested character count, clamped to 10,000.
- line-count: optional requested line count, clamped to 100.
- A truncation notice is appended when additional file content exists. Continue with offsets or use ripgrep to locate a targeted passage.

Required parameters:
- path: file path to read. Relative paths are resolved from the current process directory.

Example:
{"path":"src/lib.rs"}"#;

const READ_MAX_LINES: usize = 100;
const READ_MAX_CHARS: usize = 10_000;
const READ_TRUNCATION_NOTICE: &str = "\n\n[Read truncated: more content is available. Continue with line-offset or character-offset; each read is capped at 100 lines or 10,000 characters.]";

const CREATE_FILE_DESCRIPTION: &str = r#"Create a UTF-8 text file under an allowed write root.

Required parameters:
- path: file path to create. Relative paths are resolved from the current process directory.
- content: complete file contents to write.

Optional parameters:
- overwrite: true to replace an existing file. Defaults to false.
- create_parent_dirs: true to create missing parent directories. Defaults to true.

Example:
{"path":"notes/todo.txt","content":"- first task\n"}"#;

const DELETE_FILE_DESCRIPTION: &str = r#"Delete a file under an allowed write root.

Required parameters:
- path: file path to delete. The path must refer to an existing file, not a directory.

Example:
{"path":"notes/old.txt"}"#;

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
/// Arguments for the built-in `read` tool.
pub struct ReadArgs {
    /// File path to read. Relative paths are resolved from the current process directory.
    pub path: PathBuf,
    /// Zero-based Unicode character offset applied after line-offset.
    #[serde(default, rename = "character-offset")]
    pub character_offset: Option<usize>,
    /// Zero-based number of complete lines to skip before character-offset.
    #[serde(default, rename = "line-offset")]
    pub line_offset: Option<usize>,
    /// Requested character count. Values above 10,000 are clamped.
    #[serde(default, rename = "character-count")]
    pub character_count: Option<usize>,
    /// Requested line count. Values above 100 are clamped.
    #[serde(default, rename = "line-count")]
    pub line_count: Option<usize>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema)]
/// Arguments for the built-in `create_file` tool.
pub struct CreateFileArgs {
    /// File path to create. Relative paths are resolved from the current process directory.
    pub path: PathBuf,
    /// Complete UTF-8 text contents to write.
    pub content: String,
    /// Replace an existing file instead of failing.
    #[serde(default)]
    pub overwrite: bool,
    /// Create missing parent directories before writing.
    #[serde(default = "default_true")]
    pub create_parent_dirs: bool,
}

#[derive(Clone, Debug, Deserialize, JsonSchema)]
/// Arguments for the built-in `delete_file` tool.
pub struct DeleteFileArgs {
    /// File path to delete. The path must refer to an existing file, not a directory.
    pub path: PathBuf,
}

#[derive(Clone, Debug, Deserialize, JsonSchema)]
/// Arguments for the built-in `apply_patch` tool.
pub struct ApplyPatchArgs {
    /// Patch text in the documented begin/end patch format.
    pub patch: String,
}

#[derive(Clone, Debug, Default, Deserialize, JsonSchema)]
/// Arguments for the built-in `ripgrep` tool.
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

/// Create the built-in `read` tool.
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

/// Create the built-in `create_file` tool.
pub fn create_file_tool(state: AgentState) -> Tool {
    Tool::new(
        ToolDefinition::new(
            "create_file",
            CREATE_FILE_DESCRIPTION,
            schemars::schema_for!(CreateFileArgs),
        ),
        move |arguments| {
            let state = state.clone();
            Box::pin(async move {
                let args: CreateFileArgs = serde_json::from_value(arguments)?;
                create_file(&state, args).map(serde_json::Value::String)
            })
        },
    )
}

/// Create the built-in `delete_file` tool.
pub fn delete_file_tool(state: AgentState) -> Tool {
    Tool::new(
        ToolDefinition::new(
            "delete_file",
            DELETE_FILE_DESCRIPTION,
            schemars::schema_for!(DeleteFileArgs),
        ),
        move |arguments| {
            let state = state.clone();
            Box::pin(async move {
                let args: DeleteFileArgs = serde_json::from_value(arguments)?;
                delete_file(&state, args).map(serde_json::Value::String)
            })
        },
    )
}

/// Create the built-in `apply_patch` tool.
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

/// Create the built-in `ripgrep` search tool.
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

/// Create the default built-in file-tool registry.
///
/// Includes `read`, `create_file`, `delete_file`, `apply_patch`, and `ripgrep`.
pub fn builtin_registry(state: AgentState) -> crate::ToolRegistry {
    crate::ToolRegistry::new()
        .with_tool(read_tool(state.clone()))
        .with_tool(create_file_tool(state.clone()))
        .with_tool(delete_file_tool(state.clone()))
        .with_tool(apply_patch_tool(state.clone()))
        .with_tool(ripgrep_tool(state))
}

fn read(state: &AgentState, args: ReadArgs) -> Result<String> {
    let path = ensure_can_read(state, &args.path)?;
    let fs_path = tool_path(&path);
    if fs_path.is_dir() {
        return Err(Error::Tool(format!(
            "read only accepts UTF-8 files, not directories: {}. Use ripgrep with files=true to list files.",
            display_path(&path)
        )));
    }
    let file = std::fs::File::open(fs_path)?;
    let mut reader = BufReader::new(file);
    let requested_lines = args.line_count.unwrap_or(READ_MAX_LINES);
    let requested_chars = args.character_count.unwrap_or(READ_MAX_CHARS);
    if requested_lines == 0 || requested_chars == 0 {
        return Err(Error::Tool(
            "line-count and character-count must be positive when provided".to_string(),
        ));
    }
    let max_lines = requested_lines.min(READ_MAX_LINES);
    let max_chars = requested_chars.min(READ_MAX_CHARS);

    for _ in 0..args.line_offset.unwrap_or(0) {
        let mut discarded = String::new();
        if reader.read_line(&mut discarded)? == 0 {
            return Ok(String::new());
        }
    }

    let mut pending = String::new();
    let mut characters_to_skip = args.character_offset.unwrap_or(0);
    while characters_to_skip > 0 {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            return Ok(String::new());
        }
        let line_chars = line.chars().count();
        if characters_to_skip >= line_chars {
            characters_to_skip -= line_chars;
        } else {
            pending.extend(line.chars().skip(characters_to_skip));
            characters_to_skip = 0;
        }
    }

    let mut output = String::new();
    let mut line_count = 0;
    let mut char_count = 0;
    let mut truncated = false;

    while line_count < max_lines && char_count < max_chars {
        let line = if pending.is_empty() {
            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 {
                break;
            }
            line
        } else {
            std::mem::take(&mut pending)
        };
        if line.is_empty() {
            break;
        }
        let remaining = max_chars - char_count;
        let line_chars = line.chars().count();
        if line_chars > remaining {
            output.extend(line.chars().take(remaining));
            truncated = true;
            break;
        }
        output.push_str(&line);
        char_count += line_chars;
        line_count += 1;
    }

    if !truncated && (line_count == max_lines || char_count == max_chars) {
        truncated = !pending.is_empty() || !reader.fill_buf()?.is_empty();
    }
    if truncated {
        output.push_str(READ_TRUNCATION_NOTICE);
    }
    Ok(output)
}

fn create_file(state: &AgentState, args: CreateFileArgs) -> Result<String> {
    let path = ensure_can_write(state, &args.path)?;
    let fs_path = tool_path(&path);
    if fs_path.try_exists()? && !args.overwrite {
        return Err(Error::Tool(format!(
            "cannot create file that already exists without overwrite=true: {}",
            display_path(&path)
        )));
    }
    if let Some(parent) = fs_path.parent()
        && args.create_parent_dirs
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&fs_path, args.content)?;
    Ok(format!("created {}", display_path(&path)))
}

fn delete_file(state: &AgentState, args: DeleteFileArgs) -> Result<String> {
    let path = ensure_can_write(state, &args.path)?;
    let fs_path = tool_path(&path);
    if !fs_path.try_exists()? {
        return Err(Error::Tool(format!(
            "cannot delete missing file: {}",
            display_path(&path)
        )));
    }
    if fs_path.is_dir() {
        return Err(Error::Tool(format!(
            "delete_file only deletes files, not directories: {}",
            display_path(&path)
        )));
    }
    std::fs::remove_file(&fs_path)?;
    Ok(format!("deleted {}", display_path(&path)))
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
    let paths = args
        .paths
        .iter()
        .map(|path| ensure_can_read(state, path))
        .collect::<Result<Vec<_>>>()?;

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
    command.args(paths.iter().map(|path| tool_path(path)));

    let output = command.output()?;
    let stdout = sanitize_tool_output(&String::from_utf8_lossy(&output.stdout));
    let stderr = sanitize_tool_output(&String::from_utf8_lossy(&output.stderr));
    if output.status.success() {
        return Ok(if stdout.trim().is_empty() {
            if args.files {
                "No files found under the requested paths.".to_string()
            } else {
                "No matches found.".to_string()
            }
        } else {
            stdout
        });
    }
    if output.status.code() == Some(1) {
        let combined = format!("{stdout}{stderr}");
        return Ok(if combined.trim().is_empty() {
            if args.files {
                "No files found under the requested paths.".to_string()
            } else {
                "No matches found.".to_string()
            }
        } else {
            combined
        });
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

    let mut staged = BTreeMap::new();
    let mut changed = Vec::new();
    for operation in operations {
        let path = ensure_can_write(state, operation.path())?;
        match operation {
            PatchOperation::Add { lines, .. } => {
                if staged_file_exists(&staged, &path)? {
                    return Err(Error::Tool(format!(
                        "cannot add file that already exists: {}",
                        display_path(&path)
                    )));
                }
                staged.insert(path.clone(), Some(lines_to_text(&lines)));
                changed.push(format!("added {}", display_path(&path)));
            }
            PatchOperation::Delete { .. } => {
                if !staged_file_exists(&staged, &path)? {
                    return Err(Error::Tool(format!(
                        "cannot delete missing file: {}",
                        display_path(&path)
                    )));
                }
                staged.insert(path.clone(), None);
                changed.push(format!("deleted {}", display_path(&path)));
            }
            PatchOperation::Update { hunks, .. } => {
                let original = staged_file_contents(&staged, &path)?;
                let updated = apply_hunks(&original, &hunks)?;
                staged.insert(path.clone(), Some(updated));
                changed.push(format!("updated {}", display_path(&path)));
            }
        }
    }

    commit_staged_files(&staged)?;

    Ok(changed.join("\n"))
}

fn ensure_can_read(state: &AgentState, path: &Path) -> Result<PathBuf> {
    state
        .readable_path(path)
        .ok_or_else(|| Error::PathNotAllowed {
            path: display_path(path),
            access: "read",
        })
}

fn ensure_can_write(state: &AgentState, path: &Path) -> Result<PathBuf> {
    state
        .writable_path(path)
        .ok_or_else(|| Error::PathNotAllowed {
            path: display_path(path),
            access: "write",
        })
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
    Add {
        path: PathBuf,
        lines: Vec<String>,
    },
    Delete {
        path: PathBuf,
    },
    Update {
        path: PathBuf,
        hunks: Vec<Vec<HunkLine>>,
    },
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
            let mut hunks = Vec::new();
            let mut hunk_lines = Vec::new();
            while index < lines.len() - 1 && !lines[index].starts_with("*** ") {
                let line = lines[index];
                if line == "@@" || line.starts_with("@@ ") {
                    if !hunk_lines.is_empty() {
                        hunks.push(hunk_lines);
                        hunk_lines = Vec::new();
                    }
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
            if !hunk_lines.is_empty() {
                hunks.push(hunk_lines);
            }
            if hunks.is_empty() {
                return Err(Error::Tool("update file contained no hunks".to_string()));
            }
            operations.push(PatchOperation::Update {
                path: PathBuf::from(path),
                hunks,
            });
        } else if line.trim().is_empty() {
            index += 1;
        } else {
            return Err(Error::Tool(format!("unsupported patch operation `{line}`")));
        }
    }

    Ok(operations)
}

fn apply_hunks(original: &str, hunks: &[Vec<HunkLine>]) -> Result<String> {
    let had_trailing_newline = original.ends_with('\n');
    let original_lines = original.lines().map(str::to_string).collect::<Vec<_>>();
    let mut output = Vec::new();
    let mut cursor = 0;

    for hunk_lines in hunks {
        let mut expected = Vec::new();
        for line in hunk_lines {
            match line {
                HunkLine::Context(line) | HunkLine::Remove(line) => expected.push(line.clone()),
                HunkLine::Add(_) => {}
            }
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
    if had_trailing_newline && !text.is_empty() {
        text.push('\n');
    }
    Ok(text)
}

fn staged_file_exists(staged: &BTreeMap<PathBuf, Option<String>>, path: &Path) -> Result<bool> {
    match staged.get(path) {
        Some(Some(_)) => Ok(true),
        Some(None) => Ok(false),
        None => Ok(tool_path(path).try_exists()?),
    }
}

fn staged_file_contents(staged: &BTreeMap<PathBuf, Option<String>>, path: &Path) -> Result<String> {
    match staged.get(path) {
        Some(Some(content)) => Ok(content.clone()),
        Some(None) => Err(Error::Tool(format!(
            "cannot update deleted file: {}",
            display_path(path)
        ))),
        None => Ok(std::fs::read_to_string(tool_path(path))?),
    }
}

#[derive(Clone, Debug)]
enum FileSnapshot {
    Missing,
    File(String),
}

impl FileSnapshot {
    fn capture(path: &Path) -> Result<Self> {
        let fs_path = tool_path(path);
        if fs_path.try_exists()? {
            Ok(Self::File(std::fs::read_to_string(fs_path)?))
        } else {
            Ok(Self::Missing)
        }
    }

    fn restore(&self, path: &Path) -> std::io::Result<()> {
        let fs_path = tool_path(path);
        match self {
            Self::Missing => {
                if fs_path.try_exists()? {
                    std::fs::remove_file(fs_path)?;
                }
            }
            Self::File(content) => {
                if let Some(parent) = fs_path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(fs_path, content)?;
            }
        }
        Ok(())
    }
}

fn commit_staged_files(staged: &BTreeMap<PathBuf, Option<String>>) -> Result<()> {
    let snapshots = staged
        .keys()
        .map(|path| Ok((path.clone(), FileSnapshot::capture(path)?)))
        .collect::<Result<BTreeMap<_, _>>>()?;

    for (path, content) in staged {
        if let Err(error) = write_staged_file(path, content.as_deref()) {
            if let Err(rollback_error) = rollback_staged_files(&snapshots) {
                return Err(Error::Tool(format!(
                    "patch failed: {error}; rollback failed: {rollback_error}"
                )));
            }
            return Err(error);
        }
    }

    Ok(())
}

fn write_staged_file(path: &Path, content: Option<&str>) -> Result<()> {
    let fs_path = tool_path(path);
    match content {
        Some(content) => {
            if let Some(parent) = fs_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(fs_path, content)?;
        }
        None => {
            std::fs::remove_file(fs_path)?;
        }
    }
    Ok(())
}

fn rollback_staged_files(snapshots: &BTreeMap<PathBuf, FileSnapshot>) -> std::io::Result<()> {
    for (path, snapshot) in snapshots {
        snapshot.restore(path)?;
    }
    Ok(())
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

fn default_true() -> bool {
    true
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
        let directory_error = registry
            .get("read")
            .unwrap()
            .call(json!({ "path": file.parent().unwrap() }))
            .await
            .unwrap_err()
            .to_string();
        assert!(directory_error.contains("not directories"));
    }

    #[tokio::test]
    async fn read_tool_caps_lines_and_reports_truncation() {
        let temp = tempfile::tempdir().unwrap();
        let file = temp.path().join("long.txt");
        let content = (1..=101)
            .map(|line| format!("line {line}\n"))
            .collect::<String>();
        std::fs::write(&file, content).unwrap();
        let registry =
            ToolRegistry::new().with_tool(read_tool(AgentState::new([temp.path().into()], [])));

        let output = registry
            .get("read")
            .unwrap()
            .call(json!({"path": file}))
            .await
            .unwrap();
        let output = output.as_str().unwrap();
        assert!(output.contains("line 100\n"));
        assert!(!output.contains("line 101\n"));
        assert!(output.contains("[Read truncated:"));
    }

    #[tokio::test]
    async fn read_tool_caps_unicode_characters_not_bytes() {
        let temp = tempfile::tempdir().unwrap();
        let file = temp.path().join("characters.txt");
        std::fs::write(&file, "é".repeat(READ_MAX_CHARS + 1)).unwrap();
        let registry =
            ToolRegistry::new().with_tool(read_tool(AgentState::new([temp.path().into()], [])));

        let output = registry
            .get("read")
            .unwrap()
            .call(json!({"path": file}))
            .await
            .unwrap();
        let output = output.as_str().unwrap();
        let content = output.split(READ_TRUNCATION_NOTICE).next().unwrap();
        assert_eq!(content.chars().count(), READ_MAX_CHARS);
        assert!(output.ends_with(READ_TRUNCATION_NOTICE));
    }

    #[tokio::test]
    async fn read_tool_does_not_mark_exact_limit_at_eof_as_truncated() {
        let temp = tempfile::tempdir().unwrap();
        let file = temp.path().join("exact.txt");
        std::fs::write(&file, "x".repeat(READ_MAX_CHARS)).unwrap();
        let registry =
            ToolRegistry::new().with_tool(read_tool(AgentState::new([temp.path().into()], [])));

        let output = registry
            .get("read")
            .unwrap()
            .call(json!({"path": file}))
            .await
            .unwrap();
        let output = output.as_str().unwrap();
        assert_eq!(output.chars().count(), READ_MAX_CHARS);
        assert!(!output.contains("[Read truncated:"));
    }

    #[tokio::test]
    async fn read_tool_applies_line_offset_and_count() {
        let temp = tempfile::tempdir().unwrap();
        let file = temp.path().join("lines.txt");
        std::fs::write(&file, "zero\none\ntwo\nthree\n").unwrap();
        let registry =
            ToolRegistry::new().with_tool(read_tool(AgentState::new([temp.path().into()], [])));

        let output = registry
            .get("read")
            .unwrap()
            .call(json!({
                "path": file,
                "line-offset": 1,
                "line-count": 2
            }))
            .await
            .unwrap();
        let output = output.as_str().unwrap();
        assert!(output.starts_with("one\ntwo\n"));
        assert!(output.ends_with(READ_TRUNCATION_NOTICE));
    }

    #[tokio::test]
    async fn read_tool_applies_unicode_character_offset_and_count() {
        let temp = tempfile::tempdir().unwrap();
        let file = temp.path().join("characters.txt");
        std::fs::write(&file, "aé日bc").unwrap();
        let registry =
            ToolRegistry::new().with_tool(read_tool(AgentState::new([temp.path().into()], [])));

        let output = registry
            .get("read")
            .unwrap()
            .call(json!({
                "path": file,
                "character-offset": 1,
                "character-count": 3
            }))
            .await
            .unwrap();
        let output = output.as_str().unwrap();
        assert!(output.starts_with("é日b"));
        assert!(output.ends_with(READ_TRUNCATION_NOTICE));
    }

    #[tokio::test]
    async fn read_tool_composes_line_then_character_offsets() {
        let temp = tempfile::tempdir().unwrap();
        let file = temp.path().join("combined.txt");
        std::fs::write(&file, "skip\néclair\nlast\n").unwrap();
        let registry =
            ToolRegistry::new().with_tool(read_tool(AgentState::new([temp.path().into()], [])));

        let output = registry
            .get("read")
            .unwrap()
            .call(json!({
                "path": file,
                "line-offset": 1,
                "character-offset": 1,
                "line-count": 1
            }))
            .await
            .unwrap();
        let output = output.as_str().unwrap();
        assert!(output.starts_with("clair\n"));
        assert!(output.ends_with(READ_TRUNCATION_NOTICE));
    }

    #[test]
    fn read_tool_schema_uses_hyphenated_pagination_parameters() {
        let definition = read_tool(AgentState::default()).definition().clone();
        let properties = definition.function.parameters["properties"]
            .as_object()
            .unwrap();
        for name in [
            "character-offset",
            "line-offset",
            "character-count",
            "line-count",
        ] {
            assert!(properties.contains_key(name), "missing {name}");
        }
    }

    #[tokio::test]
    async fn ripgrep_file_listing_describes_an_empty_directory() {
        let temp = tempfile::tempdir().unwrap();
        let registry =
            ToolRegistry::new().with_tool(ripgrep_tool(AgentState::new([temp.path().into()], [])));
        let output = registry
            .get("ripgrep")
            .unwrap()
            .call(json!({"files": true, "paths": [temp.path()]}))
            .await
            .unwrap();
        assert_eq!(output, json!("No files found under the requested paths."));
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
    async fn create_and_delete_file_tools_validate_write_roots() {
        let temp = tempfile::tempdir().unwrap();
        let allowed = temp.path().join("allowed");
        let denied = temp.path().join("denied");
        std::fs::create_dir_all(&allowed).unwrap();
        std::fs::create_dir_all(&denied).unwrap();
        let file = allowed.join("nested").join("note.txt");
        let denied_file = denied.join("note.txt");
        let registry = ToolRegistry::new()
            .with_tool(create_file_tool(AgentState::new([], [allowed.clone()])))
            .with_tool(delete_file_tool(AgentState::new([], [allowed.clone()])));

        let created = registry
            .get("create_file")
            .unwrap()
            .call(json!({
                "path": file,
                "content": "hello\n"
            }))
            .await
            .unwrap();
        assert!(created.as_str().unwrap().contains("created"));
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "hello\n");

        let duplicate = registry
            .get("create_file")
            .unwrap()
            .call(json!({
                "path": file,
                "content": "replacement\n"
            }))
            .await
            .unwrap_err();
        assert!(duplicate.to_string().contains("overwrite=true"));
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "hello\n");

        registry
            .get("create_file")
            .unwrap()
            .call(json!({
                "path": file,
                "content": "replacement\n",
                "overwrite": true
            }))
            .await
            .unwrap();
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "replacement\n");

        assert!(
            registry
                .get("create_file")
                .unwrap()
                .call(json!({
                    "path": denied_file,
                    "content": "nope\n"
                }))
                .await
                .is_err()
        );

        let deleted = registry
            .get("delete_file")
            .unwrap()
            .call(json!({ "path": file }))
            .await
            .unwrap();
        assert!(deleted.as_str().unwrap().contains("deleted"));
        assert!(!file.exists());
    }

    #[tokio::test]
    async fn delete_file_tool_rejects_directories() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path().join("directory");
        std::fs::create_dir_all(&dir).unwrap();
        let registry = ToolRegistry::new()
            .with_tool(delete_file_tool(AgentState::new([], [temp.path().into()])));

        let error = registry
            .get("delete_file")
            .unwrap()
            .call(json!({ "path": dir }))
            .await
            .unwrap_err();

        assert!(error.to_string().contains("not directories"));
        assert!(dir.exists());
    }

    #[test]
    fn builtin_registry_includes_write_file_tools_by_default() {
        let registry = builtin_registry(AgentState::default());

        assert!(registry.get("create_file").is_some());
        assert!(registry.get("delete_file").is_some());
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

    #[tokio::test]
    async fn apply_patch_tool_handles_multiple_disjoint_hunks() {
        let temp = tempfile::tempdir().unwrap();
        let file = temp.path().join("main.rs");
        std::fs::write(&file, "foo\nunchanged\nbar\n").unwrap();
        let registry = ToolRegistry::new()
            .with_tool(apply_patch_tool(AgentState::new([], [temp.path().into()])));

        registry
            .get("apply_patch")
            .unwrap()
            .call(json!({
                "patch": format!(
                    "*** Begin Patch\n*** Update File: {}\n@@\n-foo\n+FOO\n@@\n-bar\n+BAR\n*** End Patch\n",
                    file.display()
                )
            }))
            .await
            .unwrap();

        assert_eq!(
            std::fs::read_to_string(file).unwrap(),
            "FOO\nunchanged\nBAR\n"
        );
    }

    #[tokio::test]
    async fn apply_patch_tool_rolls_back_failed_multi_operation_patch() {
        let temp = tempfile::tempdir().unwrap();
        let created = temp.path().join("created.txt");
        let missing = temp.path().join("missing.txt");
        let registry = ToolRegistry::new()
            .with_tool(apply_patch_tool(AgentState::new([], [temp.path().into()])));

        let error = registry
            .get("apply_patch")
            .unwrap()
            .call(json!({
                "patch": format!(
                    "*** Begin Patch\n*** Add File: {}\n+new\n*** Delete File: {}\n*** End Patch\n",
                    created.display(),
                    missing.display()
                )
            }))
            .await
            .unwrap_err();

        assert!(error.to_string().contains("cannot delete missing file"));
        assert!(!created.exists());
    }

    #[tokio::test]
    async fn apply_patch_tool_can_remove_all_content_without_leaving_newline() {
        let temp = tempfile::tempdir().unwrap();
        let file = temp.path().join("empty-me.txt");
        std::fs::write(&file, "delete me\n").unwrap();
        let registry = ToolRegistry::new()
            .with_tool(apply_patch_tool(AgentState::new([], [temp.path().into()])));

        registry
            .get("apply_patch")
            .unwrap()
            .call(json!({
                "patch": format!(
                    "*** Begin Patch\n*** Update File: {}\n@@\n-delete me\n*** End Patch\n",
                    file.display()
                )
            }))
            .await
            .unwrap();

        assert_eq!(std::fs::read_to_string(file).unwrap(), "");
    }

    #[test]
    fn builtin_tool_definitions_explain_parameters() {
        let state = AgentState::default();
        let read = read_tool(state.clone());
        let create = create_file_tool(state.clone());
        let delete = delete_file_tool(state.clone());
        let patch = apply_patch_tool(state.clone());
        let rg = ripgrep_tool(state);

        assert!(read.definition().function.description.contains("Example"));
        assert!(
            create
                .definition()
                .function
                .description
                .contains("overwrite")
        );
        assert!(
            delete
                .definition()
                .function
                .description
                .contains("not a directory")
        );
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
