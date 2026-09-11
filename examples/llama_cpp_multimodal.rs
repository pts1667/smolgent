//! Send local media through llama.cpp's capability-aware reader.
//!
//! cargo run --example llama_cpp_multimodal -- ./image.png ./clip.mp4
//! cargo run --example llama_cpp_multimodal -- --prompt "Transcribe this" ./sound.wav
use std::path::PathBuf;

use serde_json::json;
use smolgent::{
    AgentState, ApiKeyRef, ChatProvider, ChatSession, ContentPart, MessageContent, ProviderConfig,
    builtin_registry_for_provider,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1).peekable();
    if args.peek().is_none()
        || args
            .peek()
            .is_some_and(|arg| arg == "--help" || arg == "-h")
    {
        println!(
            "Usage: cargo run --example llama_cpp_multimodal -- [--prompt <text>] <path> [<path> ...]\n\n\
             Reads model capabilities from /props, then attaches supported local media.\n\
             LLAMA_CPP_BASE_URL: server root (default http://127.0.0.1:8080)\n\
             LLAMA_CPP_MODEL: model alias (default local-model)\n\
             LLAMA_CPP_API_KEY: optional server API key\n\
             Images: PNG/JPEG/GIF/BMP/TGA; audio: WAV/MP3/FLAC.\n\
             Video requires a server advertising video support with FFmpeg available."
        );
        return Ok(());
    }
    let mut prompt = "Describe the attached media.".to_string();
    let mut paths = Vec::new();
    while let Some(arg) = args.next() {
        if arg == "--prompt" {
            prompt = args.next().ok_or("--prompt requires a question")?;
        } else {
            paths.push(PathBuf::from(arg).canonicalize()?);
        }
    }
    if paths.is_empty() {
        return Err("provide at least one local media file".into());
    }

    let base =
        std::env::var("LLAMA_CPP_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:8080".into());
    let model = std::env::var("LLAMA_CPP_MODEL").unwrap_or_else(|_| "local-model".into());
    let mut config = ProviderConfig::llama_cpp(base, model)?;
    if let Ok(key) = std::env::var("LLAMA_CPP_API_KEY") {
        config.api_key = ApiKeyRef::Literal(key);
    }
    let provider = ChatProvider::new(config);
    // Restrict reads to the explicitly supplied paths. The reader also enforces its 20 MiB cap.
    let registry =
        builtin_registry_for_provider(AgentState::new(paths.clone(), []), &provider).await?;
    let reader = registry.get("read").ok_or("read tool missing")?;
    let mut content = vec![ContentPart::text(prompt)];
    for path in paths {
        let result = reader.call_content(json!({"path": path})).await?;
        if !result.has_media() {
            return Err("file was not attached as media; check its extension and the server's /props capabilities".into());
        }
        if let MessageContent::Parts(parts) = result {
            content.extend(parts);
        }
    }
    let mut session = ChatSession::new();
    let response = session.send_user_message(&provider, content).await?;
    println!("{}", response.message.content);
    Ok(())
}
