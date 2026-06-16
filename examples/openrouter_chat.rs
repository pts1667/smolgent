use std::sync::Arc;

use smolgent::{ChatProvider, ChatSession, KeyringCoreSecretStore, ProviderConfig};

#[tokio::main]
async fn main() -> smolgent::Result<()> {
    let model =
        std::env::var("OPENROUTER_MODEL").unwrap_or_else(|_| "openai/gpt-oss-20b".to_string());

    let secrets = Arc::new(KeyringCoreSecretStore::smolgent()?);
    let provider = ChatProvider::new(ProviderConfig::openrouter(model)?).with_secrets(secrets);

    let mut session = ChatSession::with_system_prompt("You are concise and helpful.");
    let response = session
        .send_user_message(&provider, "Say hello in one sentence.")
        .await?;

    println!("{}", response.message.content);
    if response.reasoning.is_some() {
        println!("reasoning returned");
    }

    Ok(())
}
