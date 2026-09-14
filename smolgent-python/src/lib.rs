use std::{path::PathBuf, sync::Arc, time::Duration};

use pyo3::{exceptions::PyRuntimeError, exceptions::PyValueError, prelude::*};
use pyo3_async_runtimes::{TaskLocals, tokio as bridge};
use serde::Deserialize;
use serde_json::json;
use smolgent::{
    AgentState, ApiKeyRef, ChatProvider, ChatSession, KeyringCoreSecretStore, MessageContent,
    ProviderConfig, ProviderKind, SecretStore, SessionConfig, Tool, ToolDefinition, ToolRegistry,
};
use tokio::sync::Mutex;

pyo3::create_exception!(_native, SmolgentError, PyRuntimeError);

fn runtime_error(error: impl std::fmt::Display) -> PyErr {
    SmolgentError::new_err(error.to_string())
}

#[derive(Deserialize)]
struct Config {
    provider: String,
    model: String,
    endpoint: Option<String>,
    api_key: Option<String>,
    keyring_id: Option<String>,
    system_prompt: String,
    read_roots: Vec<PathBuf>,
    write_roots: Vec<PathBuf>,
    max_tool_rounds: usize,
    compaction: bool,
    timeout: f64,
}

/// Internal bridge. The public, typed API lives in python/smolgent/__init__.py.
#[pyclass(module = "smolgent._native")]
struct NativeAgent {
    provider: ChatProvider,
    session: Arc<Mutex<ChatSession>>,
    initial_session: ChatSession,
    builtins: ToolRegistry,
}

fn python_tool(definition: ToolDefinition, callback: Py<PyAny>, locals: TaskLocals) -> Tool {
    let callback = Arc::new(callback);
    let locals = Arc::new(locals);
    Tool::new(definition, move |arguments| {
        let callback = callback.clone();
        let locals = locals.clone();
        Box::pin(async move {
            // The Python adapter is always async, including for synchronous user tools.
            let future = Python::attach(|py| {
                let awaitable = callback.bind(py).call1((arguments.to_string(),))?;
                pyo3_async_runtimes::into_future_with_locals(&locals, awaitable)
            })
            .map_err(|error| smolgent::Error::Tool(error.to_string()))?;
            let output = future
                .await
                .map_err(|error| smolgent::Error::Tool(error.to_string()))?;
            let output = Python::attach(|py| output.extract::<String>(py))
                .map_err(|error| smolgent::Error::Tool(error.to_string()))?;
            Ok(serde_json::from_str(&output)?)
        })
    })
}

#[pymethods]
impl NativeAgent {
    #[new]
    fn new(config_json: &str) -> PyResult<Self> {
        let config: Config = serde_json::from_str(config_json)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let timeout = Duration::try_from_secs_f64(config.timeout)
            .map_err(|_| PyValueError::new_err("timeout must be finite and positive"))?;
        if timeout.is_zero() {
            return Err(PyValueError::new_err("timeout must be positive"));
        }
        let mut provider_config = match config.provider.as_str() {
            "openrouter" => ProviderConfig::openrouter(&config.model),
            "llama_cpp" => ProviderConfig::llama_cpp(
                config
                    .endpoint
                    .as_deref()
                    .unwrap_or("http://127.0.0.1:8080"),
                &config.model,
            ),
            "compatible" => ProviderConfig::openrouter(&config.model),
            _ => return Err(PyValueError::new_err("unknown provider")),
        }
        .map_err(runtime_error)?;
        if config.provider == "compatible" {
            provider_config.name = "compatible".into();
            provider_config.kind = ProviderKind::OpenAiCompatible;
            provider_config.reasoning = None;
            provider_config.chat_completions_url = config
                .endpoint
                .as_deref()
                .ok_or_else(|| PyValueError::new_err("chat_completions_url is required"))?
                .parse()
                .map_err(|_| PyValueError::new_err("invalid chat_completions_url"))?;
        }
        if !matches!(
            provider_config.chat_completions_url.scheme(),
            "http" | "https"
        ) {
            return Err(PyValueError::new_err("provider URL must use http or https"));
        }
        provider_config.api_key = match (&config.api_key, &config.keyring_id) {
            (Some(_), Some(_)) => {
                return Err(PyValueError::new_err("choose api_key or keyring_id"));
            }
            (Some(key), None) => ApiKeyRef::Literal(key.clone()),
            (None, Some(id)) => ApiKeyRef::Keyring(id.clone()),
            (None, None) => ApiKeyRef::None,
        };
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(runtime_error)?;
        let mut provider = ChatProvider::new(provider_config).with_client(client);
        if config.keyring_id.is_some() {
            provider = provider.with_secrets(Arc::new(
                KeyringCoreSecretStore::smolgent().map_err(runtime_error)?,
            ));
        }
        let state = AgentState::new(config.read_roots, config.write_roots);
        let mut builtins = ToolRegistry::new();
        if !state.read_roots().is_empty() {
            builtins.insert(smolgent::read_tool(state.clone()));
            builtins.insert(smolgent::ripgrep_tool(state.clone()));
        }
        if !state.write_roots().is_empty() {
            builtins.insert(smolgent::create_file_tool(state.clone()));
            builtins.insert(smolgent::delete_file_tool(state.clone()));
            builtins.insert(smolgent::apply_patch_tool(state));
        }
        let mut session_config = SessionConfig {
            max_tool_rounds: config.max_tool_rounds,
            ..SessionConfig::default()
        };
        session_config.compaction.enabled = config.compaction;
        // Observability is disabled until Python exposes an actively drained event stream.
        let (session, _) =
            ChatSession::with_system_prompt_and_config(config.system_prompt, session_config);
        Ok(Self {
            provider,
            initial_session: session.clone(),
            session: Arc::new(Mutex::new(session)),
            builtins,
        })
    }

    fn arun<'py>(
        &self,
        py: Python<'py>,
        content_json: &str,
        tools: Vec<(String, Py<PyAny>)>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let content: MessageContent = serde_json::from_str(content_json)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let locals = bridge::get_current_locals(py)?;
        let mut registry = self.builtins.clone();
        for (definition, callback) in tools {
            let definition: ToolDefinition = serde_json::from_str(&definition)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            if registry.get(definition.name()).is_some()
                || matches!(
                    definition.name(),
                    "compact_remove_messages" | "compact_summarize_messages"
                )
            {
                return Err(PyValueError::new_err(format!(
                    "duplicate or reserved tool name: {}",
                    definition.name()
                )));
            }
            registry.insert(python_tool(definition, callback, locals.clone()));
        }
        // Reject overlapping/reentrant calls rather than holding a Python borrow over await.
        let mut guard = self
            .session
            .clone()
            .try_lock_owned()
            .map_err(|_| PyRuntimeError::new_err("agent is already running"))?;
        let provider = self.provider.clone();
        bridge::future_into_py_with_locals(py, locals, async move {
            // Commit history only after success. Dropping a cancelled future releases the guard
            // and discards incomplete tool-call history; external tool effects cannot be undone.
            let mut session = guard.clone();
            let response = session
                .run_user_message_with_tools(&provider, &registry, content)
                .await
                .map_err(runtime_error)?;
            let result = json!({
                "text": response.message.content.text(),
                "message": response.message,
                "reasoning": response.reasoning,
                "raw": response.raw,
            })
            .to_string();
            *guard = session;
            Ok(result)
        })
    }

    fn messages(&self) -> PyResult<String> {
        let session = self
            .session
            .try_lock()
            .map_err(|_| PyRuntimeError::new_err("agent is already running"))?;
        serde_json::to_string(&session.messages()).map_err(runtime_error)
    }

    fn reset(&self) -> PyResult<()> {
        let mut session = self
            .session
            .try_lock()
            .map_err(|_| PyRuntimeError::new_err("agent is already running"))?;
        *session = self.initial_session.clone();
        Ok(())
    }
}

#[pyfunction]
fn set_api_key(py: Python<'_>, keyring_id: String, api_key: String) -> PyResult<()> {
    py.detach(move || KeyringCoreSecretStore::smolgent()?.set_api_key(&keyring_id, &api_key))
        .map_err(runtime_error)
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeAgent>()?;
    module.add("SmolgentError", module.py().get_type::<SmolgentError>())?;
    module.add_function(wrap_pyfunction!(set_api_key, module)?)?;
    Ok(())
}
