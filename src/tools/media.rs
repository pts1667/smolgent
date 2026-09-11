//! Capability-aware extension of the built-in text reader.

use std::io::Read;
use std::path::Path;

use super::file::{ReadArgs, ensure_can_read, read_tool, tool_path};
use crate::{
    AgentState, ChatProvider, ContentPart, Error, MessageContent, ModelCapabilities, Result, Tool,
    ToolRegistry,
};

/// Maximum size of a local media file before base64 encoding (20 MiB).
/// Model-specific limits can be lower; large media must be resized or split by the application.
pub const READ_MAX_MEDIA_BYTES: usize = 20 * 1024 * 1024;

/// Extend `read` with the media types advertised by a model.
///
/// Common media formats are recognized by filename extension. Text retains its existing
/// pagination and limits. Media is returned whole, as content parts, and cannot be paginated.
/// Root permissions and the media size limit apply before any content is returned.
pub fn read_tool_with_capabilities(state: AgentState, capabilities: ModelCapabilities) -> Tool {
    let text_reader = read_tool(state.clone());
    let mut definition = text_reader.definition().clone();
    let mut formats = Vec::new();
    for (modality, description) in [
        ("image", "images: .png, .jpg, .jpeg, .webp, .gif"),
        ("video", "videos: .mp4, .mpeg, .mpg, .mov, .webm"),
        (
            "audio",
            "audio: .wav, .mp3, .aiff, .aif, .aac, .ogg, .flac, .m4a",
        ),
        ("file", "PDF documents: .pdf"),
    ] {
        if capabilities.supports_input(modality) {
            formats.push(description);
        }
    }
    if formats.is_empty() {
        return text_reader;
    }
    definition.function.description.push_str(&format!(
        "\n\nThis model can also read these local media formats: {}. The path's extension selects the format. Media is attached directly for analysis, rather than converted to text. Media reads accept only path, return the complete file, and are capped at 20 MiB before encoding. Do not supply text offsets/counts for media. Upstream format limits may be narrower.",
        formats.join("; ")
    ));
    Tool::new_multimodal(definition, move |arguments| {
        let state = state.clone();
        let capabilities = capabilities.clone();
        let text_reader = text_reader.clone();
        Box::pin(async move {
            let args: ReadArgs = serde_json::from_value(arguments.clone())?;
            let path = ensure_can_read(&state, &args.path)?;
            let Some((modality, format)) = media_format(&args.path) else {
                return text_reader.call_content(arguments).await;
            };
            if !capabilities.supports_input(modality) {
                return Err(Error::Tool(format!(
                    "model '{}' does not advertise {modality} input support",
                    capabilities.model
                )));
            }
            if args.character_offset.is_some()
                || args.line_offset.is_some()
                || args.character_count.is_some()
                || args.line_count.is_some()
            {
                return Err(Error::Tool(
                    "media reads return the complete file; text offsets/counts are not supported"
                        .into(),
                ));
            }
            let file = std::fs::File::open(tool_path(&path))?;
            let metadata = file.metadata()?;
            if !metadata.is_file() {
                return Err(Error::Tool("read requires a regular file".into()));
            }
            if metadata.len() > READ_MAX_MEDIA_BYTES as u64 {
                return Err(media_too_large());
            }
            let mut bytes = Vec::new();
            file.take(READ_MAX_MEDIA_BYTES as u64 + 1)
                .read_to_end(&mut bytes)?;
            if bytes.len() > READ_MAX_MEDIA_BYTES {
                return Err(media_too_large());
            }
            let part = match modality {
                "image" => ContentPart::image_bytes(format, &bytes),
                "video" => ContentPart::video_bytes(format, &bytes),
                "audio" => ContentPart::audio_bytes(format, &bytes),
                "file" => ContentPart::pdf_bytes(
                    args.path.file_name().unwrap_or_default().to_string_lossy(),
                    &bytes,
                ),
                _ => return Err(Error::Tool("unsupported media modality".into())),
            };
            Ok(MessageContent::Parts(vec![
                ContentPart::text(format!(
                    "Read {} ({format}, {} bytes).",
                    args.path.display(),
                    bytes.len()
                )),
                part,
            ]))
        })
    })
}

/// Construct the built-in registry and discover the configured provider model's media inputs.
///
/// Unknown models and providers without discovery retain the text reader. Network/authentication
/// errors are returned so applications can report them and choose a text-only fallback.
/// Call again when switching models; this captures one capability snapshot for the registry.
pub async fn builtin_registry_for_provider(
    state: AgentState,
    provider: &ChatProvider,
) -> Result<ToolRegistry> {
    let capabilities = provider.model_capabilities().await?;
    let mut registry = super::builtin::builtin_registry(state.clone());
    if let Some(capabilities) = capabilities {
        registry.insert(read_tool_with_capabilities(state, capabilities));
    }
    Ok(registry)
}

fn media_too_large() -> Error {
    Error::Tool(
        "media file exceeds the read limit of 20 MiB; resize or split it before reading".into(),
    )
}

fn media_format(path: &Path) -> Option<(&'static str, &'static str)> {
    Some(
        match path.extension()?.to_str()?.to_ascii_lowercase().as_str() {
            "png" => ("image", "image/png"),
            "jpg" | "jpeg" => ("image", "image/jpeg"),
            "webp" => ("image", "image/webp"),
            "gif" => ("image", "image/gif"),
            "mp4" => ("video", "video/mp4"),
            "mpeg" | "mpg" => ("video", "video/mpeg"),
            "mov" => ("video", "video/mov"),
            "webm" => ("video", "video/webm"),
            "wav" => ("audio", "wav"),
            "mp3" => ("audio", "mp3"),
            "aiff" | "aif" => ("audio", "aiff"),
            "aac" => ("audio", "aac"),
            "ogg" => ("audio", "ogg"),
            "flac" => ("audio", "flac"),
            "m4a" => ("audio", "m4a"),
            "pdf" => ("file", "application/pdf"),
            _ => return None,
        },
    )
}
