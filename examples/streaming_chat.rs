//! Set DEEPSEEK_API_KEY, then cargo run --example streaming_chat.
//! Alternatively store a key under smolgent/examples/deepseek in the smolgent service.
use futures_util::StreamExt;
use smolgent::{
    ApiKeyRef, ChatProvider, ChatSession, KeyringCoreSecretStore, ProviderConfig, StreamEvent,
    ToolRegistry,
};
use std::{
    io::{self, Write},
    sync::Arc,
};

/// Add two integers.
#[smolgent::tool]
fn add(a: i64, b: i64) -> i64 {
    a + b
}

#[tokio::main]
async fn main() -> smolgent::Result<()> {
    let model = std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into());
    let mut config = ProviderConfig::deepseek_with_keyring(model, "smolgent/examples/deepseek")?;
    let secrets = if let Ok(key) = std::env::var("DEEPSEEK_API_KEY") {
        config.api_key = ApiKeyRef::Literal(key);
        None
    } else {
        Some(Arc::new(KeyringCoreSecretStore::smolgent()?))
    };
    let mut provider = ChatProvider::new(config).with_client(
        reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()?,
    );
    if let Some(secrets) = secrets {
        provider = provider.with_secrets(secrets);
    }
    let mut session = ChatSession::with_system_prompt("Be concise.");
    let registry = ToolRegistry::new().with_tool(add_tool());
    let mut stream = session.stream_user_message_with_tools(
        &provider,
        &registry,
        "Use add to calculate 123 + 456.",
    );
    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::TextDelta { text } => {
                print!("{text}");
                io::stdout().flush()?;
            }
            StreamEvent::ToolStarted { call } => println!("\n[Calling {}]", call.function.name),
            _ => {}
        }
    }
    println!();
    Ok(())
}
