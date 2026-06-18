use smolgent::{KeyringCoreSecretStore, SecretStore};

const PROVIDER_ID: &str = "openrouter";

fn main() -> smolgent::Result<()> {
    let mut args = std::env::args().skip(1);
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
            secrets.set_api_key(PROVIDER_ID, &api_key)?;
            println!("stored OpenRouter API key in the smolgent keyring service");
        }
        "set-from-env" => {
            let api_key = std::env::var("OPENROUTER_API_KEY")
                .map_err(|_| smolgent::Error::Tool("OPENROUTER_API_KEY is not set".to_string()))?;
            secrets.set_api_key(PROVIDER_ID, &api_key)?;
            println!("stored OPENROUTER_API_KEY in the smolgent keyring service");
        }
        "check" => match secrets.get_api_key(PROVIDER_ID)? {
            Some(api_key) => println!("OpenRouter API key is stored ({} chars)", api_key.len()),
            None => println!("OpenRouter API key is not stored"),
        },
        "delete" => {
            secrets.delete_api_key(PROVIDER_ID)?;
            println!("deleted OpenRouter API key from the smolgent keyring service");
        }
        _ => print_usage(),
    }

    Ok(())
}

fn print_usage() {
    eprintln!(
        "Usage:
  cargo run --example openrouter_keyring_setup -- set <OPENROUTER_API_KEY>
  cargo run --example openrouter_keyring_setup -- set-from-env
  cargo run --example openrouter_keyring_setup -- check
  cargo run --example openrouter_keyring_setup -- delete

After setup, examples/openrouter_chat.rs can use the stored key automatically:
  cargo run --example openrouter_chat"
    );
}
