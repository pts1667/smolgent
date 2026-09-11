//! Send local images and videos to OpenRouter using z-ai/glm-5.3-flash.
//!
//! cargo run --example openrouter_multimodal -- ./image.png image/png
//! cargo run --example openrouter_multimodal -- ./clip.mp4 video/mp4
//! cargo run --example openrouter_multimodal -- ./image.png image/png ./clip.mp4 video/mp4
//!
//! Add --prompt "Your question" to customize the prompt. Uses the same keyring entry
//! as openrouter_chat, configurable through SMOLGENT_OPENROUTER_KEY_ID.
use std::sync::Arc;

use smolgent::{ChatProvider, ChatSession, ContentPart, KeyringCoreSecretStore, ProviderConfig};

const MODEL: &str = "z-ai/glm-5.3-flash";
const USAGE: &str = "Usage: cargo run --example openrouter_multimodal -- [--prompt <text>] <path> <mime-type> [<path> <mime-type> ...]

Images: ./image.png image/png
Videos: ./clip.mp4 video/mp4
Mixed:  ./image.png image/png ./clip.mp4 video/mp4

Model: z-ai/glm-5.3-flash (OpenRouter)
Keyring: SMOLGENT_OPENROUTER_KEY_ID, defaulting to smolgent/examples/openrouter
Local media is encoded as base64 data URLs before sending.";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1).peekable();
    if args.peek().is_none()
        || args
            .peek()
            .is_some_and(|arg| arg == "--help" || arg == "-h")
    {
        println!("{USAGE}");
        return Ok(());
    }

    let mut prompt = "Describe the provided images and videos. For videos, summarize the main events in chronological order.".to_string();
    let mut attachments = Vec::new();
    while let Some(path) = args.next() {
        if path == "--prompt" {
            prompt = args.next().ok_or("--prompt requires a question")?;
            continue;
        }
        let mime_type = args
            .next()
            .ok_or("each path needs a MIME type, such as image/png or video/mp4")?;
        let is_image = mime_type.starts_with("image/");
        let is_video = mime_type.starts_with("video/");
        if !is_image && !is_video {
            return Err(format!(
                "expected an image/* or video/* MIME type for {path}, got {mime_type}"
            )
            .into());
        }
        let bytes = std::fs::read(&path)?;
        // Videos use the video_url content type with a data URL; images use image_url.
        // Encoding local bytes avoids dependence on upstream support for public video URLs.
        attachments.push(if is_image {
            ContentPart::image_bytes(&mime_type, &bytes)
        } else {
            ContentPart::video_bytes(&mime_type, &bytes)
        });
    }
    if attachments.is_empty() {
        return Err("provide at least one image or video path and MIME type".into());
    }

    // Put text first, then preserve the order of all image and video attachments.
    let mut content = vec![ContentPart::text(prompt)];
    content.extend(attachments);
    let key_id = std::env::var("SMOLGENT_OPENROUTER_KEY_ID")
        .unwrap_or_else(|_| "smolgent/examples/openrouter".to_string());
    let provider = ChatProvider::new(ProviderConfig::openrouter_with_keyring(MODEL, key_id)?)
        .with_secrets(Arc::new(KeyringCoreSecretStore::smolgent()?));

    let mut session = ChatSession::new();
    let response = session.send_user_message(&provider, content).await?;
    println!("{}", response.message.content);
    Ok(())
}
