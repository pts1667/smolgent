use smolgent::{KeyringCoreSecretStore, SecretStore};

const DEFAULT_KEY_ID: &str = "smolgent/examples/openrouter";

fn main() -> smolgent::Result<()> {
    let mut args = std::env::args().skip(1);
    let key_id = read_key_id();
    let Some(command) = args.next() else {
        print_usage();
        return Ok(());
    };

    let secrets = KeyringCoreSecretStore::smolgent()?;

    match command.as_str() {
        "set" => {
            let Some(api_key) = args.next() else {
                eprintln!("missing API key");
                print_usage();
                return Ok(());
            };
            secrets.set_api_key(&key_id, &api_key)?;
            println!("stored OpenRouter API key as keyring id `{key_id}`");
        }
        "set-from-env" => {
            let api_key = std::env::var("OPENROUTER_API_KEY")
                .map_err(|_| smolgent::Error::Tool("OPENROUTER_API_KEY is not set".to_string()))?;
            secrets.set_api_key(&key_id, &api_key)?;
            println!("stored OPENROUTER_API_KEY as keyring id `{key_id}`");
        }
        "check" => match secrets.get_api_key(&key_id)? {
            Some(api_key) => {
                println!(
                    "OpenRouter API key `{key_id}` is stored ({} chars)",
                    api_key.len()
                )
            }
            None => println!("OpenRouter API key `{key_id}` is not stored"),
        },
        "delete" => {
            secrets.delete_api_key(&key_id)?;
            println!("deleted OpenRouter API key `{key_id}`");
        }
        _ => print_usage(),
    }

    Ok(())
}

fn read_key_id() -> String {
    std::env::var("SMOLGENT_OPENROUTER_KEY_ID").unwrap_or_else(|_| DEFAULT_KEY_ID.to_string())
}

fn print_usage() {
    eprintln!(
        "Usage:
  cargo run --example openrouter_keyring_setup -- set <OPENROUTER_API_KEY>
  cargo run --example openrouter_keyring_setup -- set-from-env
  cargo run --example openrouter_keyring_setup -- check
  cargo run --example openrouter_keyring_setup -- delete

Set SMOLGENT_OPENROUTER_KEY_ID to choose the app-specific keyring id.
Default keyring id: smolgent/examples/openrouter

After setup, examples/openrouter_chat.rs can use the stored key automatically:
  cargo run --example openrouter_chat"
    );
}
