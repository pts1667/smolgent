//! Ordered chat content using OpenRouter's chat-completions wire format.

use std::{borrow::Cow, fmt};

use base64::{Engine, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};

/// A plain string or an ordered array of multimodal content parts.
///
/// Text-only messages retain their original JSON string representation. Use [`Self::text`]
/// for display; serialize this value to retain attachments.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl Default for MessageContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

impl From<String> for MessageContent {
    fn from(text: String) -> Self {
        Self::Text(text)
    }
}

impl From<&str> for MessageContent {
    fn from(text: &str) -> Self {
        Self::Text(text.into())
    }
}

impl From<&String> for MessageContent {
    fn from(text: &String) -> Self {
        Self::Text(text.clone())
    }
}

impl From<Vec<ContentPart>> for MessageContent {
    fn from(parts: Vec<ContentPart>) -> Self {
        Self::Parts(parts)
    }
}

impl PartialEq<&str> for MessageContent {
    fn eq(&self, other: &&str) -> bool {
        matches!(self, Self::Text(text) if text == other)
    }
}

impl fmt::Display for MessageContent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.text())
    }
}

impl MessageContent {
    /// Concatenate text parts in order, without exposing media URLs or base64 data.
    pub fn text(&self) -> Cow<'_, str> {
        match self {
            Self::Text(text) => Cow::Borrowed(text),
            Self::Parts(parts) => Cow::Owned(
                parts
                    .iter()
                    .filter_map(|part| match part {
                        ContentPart::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect(),
            ),
        }
    }

    /// True only when there is no nonempty text and no media part.
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Text(text) => text.is_empty(),
            Self::Parts(parts) => parts
                .iter()
                .all(|part| matches!(part, ContentPart::Text { text } if text.is_empty())),
        }
    }

    /// Whether any part contains media, including a PDF/file attachment.
    pub fn has_media(&self) -> bool {
        matches!(self, Self::Parts(parts) if parts.iter().any(|part| !matches!(part, ContentPart::Text { .. })))
    }

    /// Approximate payload bytes, including encoded media but excluding JSON overhead.
    pub fn len(&self) -> usize {
        match self {
            Self::Text(text) => text.len(),
            Self::Parts(parts) => parts
                .iter()
                .map(|part| match part {
                    ContentPart::Text { text } => text.len(),
                    ContentPart::ImageUrl { image_url } => image_url.url.len(),
                    ContentPart::File { file } => {
                        file.filename.as_ref().map_or(0, String::len) + file.file_data.len()
                    }
                    ContentPart::InputAudio { input_audio } => {
                        input_audio.data.len() + input_audio.format.len()
                    }
                    ContentPart::VideoUrl { video_url } => video_url.url.len(),
                })
                .sum(),
        }
    }
}

/// One part of an OpenRouter multimodal input. Order is preserved on replay.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
    File { file: FileInput },
    InputAudio { input_audio: InputAudio },
    VideoUrl { video_url: VideoUrl },
}

/// An image URL or a base64 data URL.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<ImageDetail>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    Auto,
    Low,
    High,
}

/// A PDF URL or a base64 data URL, with an optional filename.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct FileInput {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    pub file_data: String,
}

/// Audio uses raw base64, not a URL or a data URL.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct InputAudio {
    pub data: String,
    /// For example, `wav` or `mp3`. Supported formats depend on the model.
    pub format: String,
}

/// A video URL or a base64 data URL. URL support depends on the provider/model.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct VideoUrl {
    pub url: String,
}

impl ContentPart {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Use a public image URL or an already encoded data URL.
    pub fn image_url(url: impl Into<String>) -> Self {
        Self::ImageUrl {
            image_url: ImageUrl {
                url: url.into(),
                detail: None,
            },
        }
    }

    /// Encode image bytes as a data URL (e.g. MIME type `image/png`).
    pub fn image_bytes(mime_type: &str, bytes: &[u8]) -> Self {
        Self::image_url(data_url(mime_type, bytes))
    }

    /// Use a PDF URL or an already encoded data URL.
    pub fn file(filename: impl Into<String>, file_data: impl Into<String>) -> Self {
        Self::File {
            file: FileInput {
                filename: Some(filename.into()),
                file_data: file_data.into(),
            },
        }
    }

    pub fn pdf_bytes(filename: impl Into<String>, bytes: &[u8]) -> Self {
        Self::file(filename, data_url("application/pdf", bytes))
    }

    /// Supply raw base64 audio with its format (e.g. `wav`).
    pub fn input_audio(data: impl Into<String>, format: impl Into<String>) -> Self {
        Self::InputAudio {
            input_audio: InputAudio {
                data: data.into(),
                format: format.into(),
            },
        }
    }

    pub fn audio_bytes(format: impl Into<String>, bytes: &[u8]) -> Self {
        Self::input_audio(STANDARD.encode(bytes), format)
    }

    pub fn video_url(url: impl Into<String>) -> Self {
        Self::VideoUrl {
            video_url: VideoUrl { url: url.into() },
        }
    }

    pub fn video_bytes(mime_type: &str, bytes: &[u8]) -> Self {
        Self::video_url(data_url(mime_type, bytes))
    }
}

fn data_url(mime_type: &str, bytes: &[u8]) -> String {
    format!("data:{mime_type};base64,{}", STANDARD.encode(bytes))
}
