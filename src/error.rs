#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("keyring error: {0}")]
    Keyring(#[from] keyring_core::Error),

    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),

    /// An unsuccessful HTTP response. The body is limited to a 16 KiB prefix
    /// (decoded as UTF-8), with a notice appended if it was truncated.
    #[error("provider returned HTTP {status}: {body}")]
    Provider { status: u16, body: String },

    #[error("url error: {0}")]
    Url(#[from] url::ParseError),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid http header name: {0}")]
    InvalidHeaderName(#[from] reqwest::header::InvalidHeaderName),

    #[error("invalid http header value: {0}")]
    InvalidHeaderValue(#[from] reqwest::header::InvalidHeaderValue),

    #[error("api key is not configured for provider '{0}'")]
    MissingApiKey(String),

    #[error("provider response exceeded the {limit} byte limit")]
    ProviderResponseTooLarge { limit: usize },

    #[error("no native keyring store is configured for target OS '{0}'")]
    UnsupportedNativeKeyring(&'static str),

    #[error("provider returned no assistant message")]
    MissingAssistantMessage,

    #[error("unknown tool '{0}'")]
    UnknownTool(String),

    #[error("tool failed: {0}")]
    Tool(String),

    #[error("path '{path}' is outside the allowed {access} roots")]
    PathNotAllowed { path: String, access: &'static str },
}

pub type Result<T> = std::result::Result<T, Error>;
