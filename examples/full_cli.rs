use std::env;
use std::io::{self, Stdout};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Position};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use serde_json::Value;
use smolgent::{
    AgentEvent, AgentEventReceiver, AgentState, ApiKeyRef, ChatProvider, ChatResponse, ChatSession,
    Error, NotificationConfig, ProviderConfig, SessionConfig, builtin_registry,
};

const MODEL: &str = "deepseek/deepseek-v4-flash";

/**
 * Example of a full CLI app for using smolagent.
 * Some notes:
 *  1. OPENROUTER_API_KEY is fetched from environment variables, but you should really be using the keyring store.
 *  2. The TUI is kinda crap, for example you can't really scroll "all the way" and there are issues with the input box. Polish isn't really the point
 *  */

#[tokio::main]
async fn main() -> smolgent::Result<()> {
    dotenvy::dotenv().ok();
    let args = CliArgs::parse(env::args().skip(1))?;
    let input_dir = args.input_dir.canonicalize()?;
    let api_key = env::var("OPENROUTER_API_KEY")
        .map_err(|_| Error::Tool("OPENROUTER_API_KEY is missing from .env/environment".into()))?;

    let app = TerminalGuard::enter()?;
    let result = run_app(app.terminal, input_dir, args.read_only, api_key).await;
    TerminalGuard::leave()?;
    result
}

async fn run_app(
    mut terminal: Terminal<CrosstermBackend<Stdout>>,
    input_dir: PathBuf,
    read_only: bool,
    api_key: String,
) -> smolgent::Result<()> {
    let state = if read_only {
        AgentState::new([input_dir.clone()], [])
    } else {
        AgentState::new([input_dir.clone()], [input_dir.clone()])
    };
    let display_input_dir = user_facing_path(&input_dir);
    let registry = builtin_registry(state);
    let provider = openrouter_provider(api_key)?;
    let (session, events) = ChatSession::with_system_prompt_and_config(
        system_prompt(&display_input_dir, read_only),
        SessionConfig {
            notifications: NotificationConfig::all(),
            ..SessionConfig::default()
        },
    );
    let events = events.expect("all notifications should create an event stream");
    let statuses = spawn_agent_status_thread(events);
    let mut session = Some(session);
    let mut running: Option<RunningAnswer> = None;
    let mut app = AppState::new(display_input_dir, read_only);

    draw(&mut terminal, &app)?;

    loop {
        drain_agent_statuses(&statuses, &mut app);
        if let Some(answer) = &running
            && answer.handle.is_finished()
        {
            let answer: RunningAnswer = running.take().unwrap();
            let (returned_session, result) = answer
                .handle
                .await
                .map_err(|err| Error::Tool(format!("agent task failed: {err}")))?;
            session = Some(returned_session);

            match result {
                Ok(response) => {
                    drain_agent_statuses(&statuses, &mut app);
                    app.set_status("ready");
                    app.push_chat("Assistant", response.message.content);
                }
                Err(err) => {
                    drain_agent_statuses(&statuses, &mut app);
                    app.set_status(format!("error: {err}"));
                    app.push_chat("Error", err.to_string());
                }
            }
        }

        if event::poll(Duration::from_millis(100))? {
            let Event::Key(key) = event::read()? else {
                continue;
            };
            if key.kind != KeyEventKind::Press {
                continue;
            }

            match key.code {
                KeyCode::Esc => return Ok(()),
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    return Ok(());
                }
                KeyCode::Char(ch) => app.input.push(ch),
                KeyCode::Backspace => {
                    app.input.pop();
                }
                KeyCode::PageUp => {
                    let height = chat_view_height(&terminal)?;
                    app.scroll_chat_up(height, height.saturating_sub(1).max(1));
                }
                KeyCode::PageDown => {
                    let height = chat_view_height(&terminal)?;
                    app.scroll_chat_down(height, height.saturating_sub(1).max(1));
                }
                KeyCode::Up if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    let height = chat_view_height(&terminal)?;
                    app.scroll_chat_up(height, 1);
                }
                KeyCode::Down if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    let height = chat_view_height(&terminal)?;
                    app.scroll_chat_down(height, 1);
                }
                KeyCode::Home if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    app.scroll_chat_top();
                }
                KeyCode::End if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    app.scroll_chat_bottom();
                }
                KeyCode::Enter => {
                    drain_agent_statuses(&statuses, &mut app);
                    if running.is_some() {
                        app.push_status("assistant is still working");
                        continue;
                    }
                    let prompt = app.input.trim().to_string();
                    if prompt.is_empty() {
                        continue;
                    }
                    app.input.clear();
                    app.push_chat("You", prompt.clone());
                    app.set_status("waiting for model...");
                    draw(&mut terminal, &app)?;

                    let mut active_session = session
                        .take()
                        .expect("session should be available when no answer is running");
                    let provider = provider.clone();
                    let registry = registry.clone();
                    running = Some(RunningAnswer {
                        handle: tokio::spawn(async move {
                            let result = active_session
                                .run_user_message_with_tools(&provider, &registry, prompt)
                                .await;
                            (active_session, result)
                        }),
                    });
                }
                _ => {}
            }
        }

        draw(&mut terminal, &app)?;
    }
}

struct RunningAnswer {
    handle: tokio::task::JoinHandle<(ChatSession, smolgent::Result<ChatResponse>)>,
}

fn spawn_agent_status_thread(events: AgentEventReceiver) -> Receiver<String> {
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        while let Ok(event) = events.recv() {
            if sender.send(agent_event_status(&event)).is_err() {
                break;
            }
        }
    });
    receiver
}

fn openrouter_provider(api_key: String) -> smolgent::Result<ChatProvider> {
    let mut config = ProviderConfig::openrouter(MODEL)?;
    config.api_key = ApiKeyRef::Literal(api_key);
    Ok(ChatProvider::new(config))
}

fn system_prompt(input_dir: &Path, read_only: bool) -> String {
    let write_note = if read_only {
        "You are in read-only mode. Do not call apply_patch."
    } else {
        "You may call apply_patch, but only when the user explicitly asks for edits."
    };
    format!(
        "You are a coding assistant helping the user understand this input directory: {}.\n\
         Use ripgrep to search, read to inspect specific files, and cite concrete paths in answers.\n\
         Prefer targeted reads over dumping many large files. Use ordinary Windows paths; do not add a \\\\?\\ prefix. {write_note}",
        input_dir.display()
    )
}

fn user_facing_path(path: &Path) -> PathBuf {
    strip_windows_verbatim_prefix(path)
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

fn drain_agent_statuses(statuses: &Receiver<String>, app: &mut AppState) {
    while let Ok(status) = statuses.try_recv() {
        app.set_status(status.clone());
        app.push_status(status);
    }
}

fn agent_event_status(event: &AgentEvent) -> String {
    match event {
        AgentEvent::ModelRequestStarted {
            completed_tool_rounds,
        } => {
            if *completed_tool_rounds == 0 {
                "waiting for model...".to_string()
            } else {
                format!("waiting for model after {completed_tool_rounds} tool round(s)...")
            }
        }
        AgentEvent::ModelResponseReceived { tool_calls, .. } => {
            format!("model requested {tool_calls} tool call(s)")
        }
        AgentEvent::ToolRoundStarted { round, tool_calls } => {
            format!("starting tool round {round} with {tool_calls} call(s)")
        }
        AgentEvent::ToolCallStarted {
            name, arguments, ..
        } => tool_call_status(name, arguments),
        AgentEvent::ToolCallFinished {
            name, content_len, ..
        } => {
            format!("{name} returned {content_len} byte(s)")
        }
        AgentEvent::ToolCallFailed { name, error, .. } => format!("{name} failed: {error}"),
        AgentEvent::MaxToolRoundsReached { max_tool_rounds } => {
            format!("stopped after {max_tool_rounds} tool rounds")
        }
    }
}

fn tool_call_status(name: &str, arguments: &str) -> String {
    let args = serde_json::from_str::<Value>(arguments).unwrap_or(Value::Null);
    match name {
        "read" => format!("reading {}", json_path(&args, "path")),
        "ripgrep" => {
            let pattern = args
                .get("pattern")
                .and_then(Value::as_str)
                .unwrap_or("<files>");
            let paths = args
                .get("paths")
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(Value::as_str)
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .filter(|text| !text.is_empty())
                .unwrap_or_else(|| "<no paths>".to_string());
            format!("searching {paths} for {pattern:?}")
        }
        "apply_patch" => "applying patch".to_string(),
        other => format!("running {other}"),
    }
}

fn json_path(args: &Value, key: &str) -> String {
    args.get(key)
        .and_then(Value::as_str)
        .unwrap_or("<missing path>")
        .to_string()
}

fn draw(terminal: &mut Terminal<CrosstermBackend<Stdout>>, app: &AppState) -> smolgent::Result<()> {
    terminal.draw(|frame| {
        let area = frame.area();
        let root = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(8),
                Constraint::Length(7),
                Constraint::Length(3),
            ])
            .split(area);
        let body = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(68), Constraint::Percentage(32)])
            .split(root[0]);
        let chat_height = body[0].height.saturating_sub(2) as usize;
        let chat_lines = chat_lines(app);
        let chat_scroll = app.chat_scroll_offset(chat_lines.len(), chat_height);

        let chat = Paragraph::new(chat_lines)
            .block(Block::default().borders(Borders::ALL).title("Conversation"))
            .scroll((chat_scroll as u16, 0))
            .wrap(Wrap { trim: false });
        frame.render_widget(chat, body[0]);

        let status = Paragraph::new(status_lines(app))
            .block(Block::default().borders(Borders::ALL).title("Status"))
            .wrap(Wrap { trim: false });
        frame.render_widget(status, body[1]);

        let help = Paragraph::new(vec![
            Line::from(format!(
                "Directory: {}{}",
                app.input_dir.display(),
                if app.read_only { " (read-only)" } else { "" }
            )),
            Line::from("Enter sends. Esc/Ctrl-C exits. PageUp/PageDown scrolls chat."),
            Line::from(format!("State: {}", app.current_status)),
        ])
        .block(Block::default().borders(Borders::ALL).title("Session"));
        frame.render_widget(help, root[1]);

        let input = Paragraph::new(app.input.as_str())
            .block(Block::default().borders(Borders::ALL).title("Ask"));
        frame.render_widget(input, root[2]);
        let cursor_x = root[2].x
            + 1
            + app
                .input
                .chars()
                .count()
                .min(root[2].width.saturating_sub(2) as usize) as u16;
        let cursor_y = root[2].y + 1;
        frame.set_cursor_position(Position::new(cursor_x, cursor_y));
    })?;
    Ok(())
}

fn chat_view_height(terminal: &Terminal<CrosstermBackend<Stdout>>) -> smolgent::Result<usize> {
    let size = terminal.size()?;
    Ok(size.height.saturating_sub(10).saturating_sub(2) as usize)
}

fn chat_lines(app: &AppState) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for entry in &app.chat {
        lines.push(Line::from(vec![Span::styled(
            format!("{}: ", entry.speaker),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )]));
        lines.extend(entry.text.lines().map(|line| Line::from(line.to_string())));
        lines.push(Line::from(""));
    }
    lines
}

fn status_lines(app: &AppState) -> Vec<Line<'static>> {
    app.status
        .iter()
        .rev()
        .take(24)
        .rev()
        .map(|line| Line::from(line.clone()))
        .collect()
}

struct TerminalGuard {
    terminal: Terminal<CrosstermBackend<Stdout>>,
}

impl TerminalGuard {
    fn enter() -> smolgent::Result<Self> {
        enable_raw_mode()?;
        execute!(io::stdout(), EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(io::stdout());
        let terminal = Terminal::new(backend)?;
        Ok(Self { terminal })
    }

    fn leave() -> smolgent::Result<()> {
        disable_raw_mode()?;
        execute!(io::stdout(), LeaveAlternateScreen)?;
        Ok(())
    }
}

#[derive(Debug, Eq, PartialEq)]
struct CliArgs {
    input_dir: PathBuf,
    read_only: bool,
}

impl CliArgs {
    fn parse(args: impl IntoIterator<Item = String>) -> smolgent::Result<Self> {
        let mut input_dir = None;
        let mut read_only = false;

        for arg in args {
            if arg == "--read-only" {
                read_only = true;
            } else if input_dir.is_none() {
                input_dir = Some(PathBuf::from(arg));
            } else {
                return Err(Error::Tool(format!("unexpected argument `{arg}`")));
            }
        }

        let input_dir = input_dir
            .ok_or_else(|| Error::Tool("usage: full_cli <input-dir> [--read-only]".into()))?;
        if !input_dir.is_dir() {
            return Err(Error::Tool(format!(
                "input directory does not exist: {}",
                input_dir.display()
            )));
        }

        Ok(Self {
            input_dir,
            read_only,
        })
    }
}

#[derive(Clone, Debug)]
struct ChatEntry {
    speaker: String,
    text: String,
}

#[derive(Debug)]
struct AppState {
    input_dir: PathBuf,
    read_only: bool,
    input: String,
    current_status: String,
    chat: Vec<ChatEntry>,
    chat_scroll: usize,
    follow_chat: bool,
    status: Vec<String>,
}

impl AppState {
    fn new(input_dir: PathBuf, read_only: bool) -> Self {
        let mut app = Self {
            input_dir,
            read_only,
            input: String::new(),
            current_status: "ready".to_string(),
            chat: Vec::new(),
            chat_scroll: 0,
            follow_chat: true,
            status: Vec::new(),
        };
        app.push_chat(
            "System",
            "Ask a question about the directory. The agent can search and read files.",
        );
        app
    }

    fn set_status(&mut self, status: impl Into<String>) {
        self.current_status = status.into();
    }

    fn push_status(&mut self, status: impl Into<String>) {
        self.status.push(status.into());
        if self.status.len() > 200 {
            self.status.drain(..50);
        }
    }

    fn push_chat(&mut self, speaker: impl Into<String>, text: impl Into<String>) {
        self.chat.push(ChatEntry {
            speaker: speaker.into(),
            text: text.into(),
        });
        if self.follow_chat {
            self.chat_scroll = usize::MAX;
        }
        if self.chat.len() > 100 {
            self.chat.drain(..20);
        }
    }

    fn chat_scroll_offset(&self, line_count: usize, view_height: usize) -> usize {
        let max_scroll = line_count.saturating_sub(view_height.max(1));
        if self.follow_chat {
            max_scroll
        } else {
            self.chat_scroll.min(max_scroll)
        }
    }

    fn scroll_chat_up(&mut self, view_height: usize, amount: usize) {
        let current = self.chat_scroll_offset(chat_line_count(&self.chat), view_height);
        self.chat_scroll = current.saturating_sub(amount);
        self.follow_chat = false;
    }

    fn scroll_chat_down(&mut self, view_height: usize, amount: usize) {
        let line_count = chat_line_count(&self.chat);
        let max_scroll = line_count.saturating_sub(view_height.max(1));
        let next = self
            .chat_scroll_offset(line_count, view_height)
            .saturating_add(amount)
            .min(max_scroll);
        self.chat_scroll = next;
        self.follow_chat = next == max_scroll;
    }

    fn scroll_chat_top(&mut self) {
        self.chat_scroll = 0;
        self.follow_chat = false;
    }

    fn scroll_chat_bottom(&mut self) {
        self.chat_scroll = usize::MAX;
        self.follow_chat = true;
    }
}

fn chat_line_count(chat: &[ChatEntry]) -> usize {
    chat.iter()
        .map(|entry| 2 + entry.text.lines().count().max(1))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_args() {
        let cwd = env::current_dir().unwrap();
        assert_eq!(
            CliArgs::parse([cwd.display().to_string(), "--read-only".to_string()]).unwrap(),
            CliArgs {
                input_dir: cwd,
                read_only: true,
            }
        );
    }

    #[test]
    fn formats_read_status() {
        let status = tool_call_status("read", r#"{"path":"src/lib.rs"}"#);
        assert_eq!(status, "reading src/lib.rs");
    }

    #[test]
    fn formats_ripgrep_status() {
        let status = tool_call_status("ripgrep", r#"{"pattern":"ProviderConfig","paths":["src"]}"#);
        assert_eq!(status, "searching src for \"ProviderConfig\"");
    }

    #[test]
    fn strips_windows_verbatim_paths_for_model_display() {
        assert_eq!(
            user_facing_path(Path::new(r"\\?\C:\repo"))
                .display()
                .to_string(),
            r"C:\repo"
        );
        assert_eq!(
            user_facing_path(Path::new(r"\\?\UNC\server\share"))
                .display()
                .to_string(),
            r"\\server\share"
        );
    }

    #[test]
    fn chat_scroll_follows_bottom_by_default() {
        let cwd = env::current_dir().unwrap();
        let mut app = AppState::new(cwd, true);
        app.push_chat(
            "Assistant",
            (0..20)
                .map(|n| format!("line {n}"))
                .collect::<Vec<_>>()
                .join("\n"),
        );

        assert!(app.chat_scroll_offset(chat_line_count(&app.chat), 5) > 0);
        assert!(app.follow_chat);
    }

    #[test]
    fn chat_scroll_can_move_up_and_back_to_bottom() {
        let cwd = env::current_dir().unwrap();
        let mut app = AppState::new(cwd, true);
        app.push_chat(
            "Assistant",
            (0..20)
                .map(|n| format!("line {n}"))
                .collect::<Vec<_>>()
                .join("\n"),
        );

        let bottom = app.chat_scroll_offset(chat_line_count(&app.chat), 5);
        app.scroll_chat_up(5, 3);
        assert_eq!(app.chat_scroll, bottom - 3);
        assert!(!app.follow_chat);

        app.scroll_chat_down(5, 3);
        assert!(app.follow_chat);
    }
}
