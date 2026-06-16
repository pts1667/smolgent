use smolgent::{ChatProvider, ChatSession, ProviderConfig};

#[tokio::main]
async fn main() -> smolgent::Result<()> {
    let base_url =
        std::env::var("LLAMA_CPP_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:8080".into());
    let model = std::env::var("LLAMA_CPP_MODEL").unwrap_or_else(|_| "local-model".into());

    let provider = ChatProvider::new(ProviderConfig::llama_cpp(base_url, model)?);
    let mut session = ChatSession::with_system_prompt("You are concise and helpful.");
    let response = session
        .send_user_message(&provider, "Say hello in one sentence.")
        .await?;

    println!("{}", response.message.content);
    Ok(())
}
