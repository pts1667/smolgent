use std::sync::Arc;

use keyring_core::CredentialStore;
use smolgent::{KeyringCoreSecretStore, SecretStore};

fn main() -> smolgent::Result<()> {
    let store: Arc<CredentialStore> = keyring_core::mock::Store::new()?;
    let secrets = KeyringCoreSecretStore::with_store("smolgent-example", store);

    secrets.set_api_key("openrouter", "sk-example")?;
    println!("{:?}", secrets.get_api_key("openrouter")?);
    secrets.delete_api_key("openrouter")?;

    Ok(())
}
