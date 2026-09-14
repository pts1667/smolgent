use smolgent::{ApiKeyRef, ChatProvider, ChatSession, Error, ProviderConfig};

#[tokio::main]
async fn main() -> smolgent::Result<()> {
    let model = std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into());
    let mut config = ProviderConfig::deepseek(model)?;
    config.api_key = ApiKeyRef::Literal(
        std::env::var("DEEPSEEK_API_KEY")
            .map_err(|_| Error::MissingApiKey("deepseek (DEEPSEEK_API_KEY)".into()))?,
    );
    let provider = ChatProvider::new(config);
    let mut session = ChatSession::with_system_prompt("You are concise and helpful.");
    let response = session
        .send_user_message(&provider, "Say hello in one sentence.")
        .await?;
    println!("{}", response.message.content);
    Ok(())
}
