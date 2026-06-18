use std::sync::Arc;

use keyring_core::{CredentialStore, Entry};

use crate::{Error, Result};

/// Storage backend for provider API keys.
///
/// Provider configurations reference keys by provider id. Apps should choose their own ids, such
/// as `my-app/openrouter`, and pass the same id to
/// [`ProviderConfig::openrouter_with_keyring`].
///
/// [`ProviderConfig::openrouter_with_keyring`]: crate::ProviderConfig::openrouter_with_keyring
pub trait SecretStore: Send + Sync {
    /// Store or replace an API key for a provider id.
    fn set_api_key(&self, provider_id: &str, api_key: &str) -> Result<()>;
    /// Load an API key for a provider id.
    fn get_api_key(&self, provider_id: &str) -> Result<Option<String>>;
    /// Delete an API key for a provider id.
    fn delete_api_key(&self, provider_id: &str) -> Result<()>;
}

/// [`SecretStore`] implementation backed by `keyring-core`.
///
/// On Windows and Linux, [`KeyringCoreSecretStore::new`] installs the native keyring-core store
/// used by this crate's optional platform dependencies. On other targets it falls back to
/// `keyring_core::Entry::new`.
#[derive(Clone)]
pub struct KeyringCoreSecretStore {
    service: String,
    store: Option<Arc<CredentialStore>>,
}

impl KeyringCoreSecretStore {
    /// Create a keyring-backed store for a service name.
    pub fn new(service: impl Into<String>) -> Result<Self> {
        let service = service.into();
        match native_credential_store() {
            Ok(store) => Ok(Self::with_store(service, store)),
            Err(Error::UnsupportedNativeKeyring(_)) => Ok(Self {
                service,
                store: None,
            }),
            Err(error) => Err(error),
        }
    }

    /// Secret store for the default `smolgent` service.
    pub fn smolgent() -> Result<Self> {
        Self::new("smolgent")
    }

    /// Create a store from an explicit keyring-core credential store.
    ///
    /// This is mainly useful for tests or custom keyring-core backends.
    pub fn with_store(service: impl Into<String>, store: Arc<CredentialStore>) -> Self {
        Self {
            service: service.into(),
            store: Some(store),
        }
    }

    fn entry(&self, provider_id: &str) -> Result<Entry> {
        match &self.store {
            Some(store) => Ok(store.build(&self.service, provider_id, None)?),
            None => Ok(Entry::new(&self.service, provider_id)?),
        }
    }
}

/// Construct the platform-native keyring-core credential store.
pub fn native_credential_store() -> Result<Arc<CredentialStore>> {
    native_credential_store_impl()
}

#[cfg(windows)]
fn native_credential_store_impl() -> Result<Arc<CredentialStore>> {
    Ok(windows_native_keyring_store::Store::new()?)
}

#[cfg(target_os = "linux")]
fn native_credential_store_impl() -> Result<Arc<CredentialStore>> {
    Ok(linux_keyutils_keyring_store::Store::new()?)
}

#[cfg(not(any(windows, target_os = "linux")))]
fn native_credential_store_impl() -> Result<Arc<CredentialStore>> {
    Err(crate::Error::UnsupportedNativeKeyring(std::env::consts::OS))
}

impl SecretStore for KeyringCoreSecretStore {
    fn set_api_key(&self, provider_id: &str, api_key: &str) -> Result<()> {
        Ok(self.entry(provider_id)?.set_password(api_key)?)
    }

    fn get_api_key(&self, provider_id: &str) -> Result<Option<String>> {
        match self.entry(provider_id)?.get_password() {
            Ok(api_key) => Ok(Some(api_key)),
            Err(keyring_core::Error::NoEntry) => Ok(None),
            Err(err) => Err(err.into()),
        }
    }

    fn delete_api_key(&self, provider_id: &str) -> Result<()> {
        match self.entry(provider_id)?.delete_credential() {
            Ok(()) | Err(keyring_core::Error::NoEntry) => Ok(()),
            Err(err) => Err(err.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use keyring_core::CredentialStore;

    use super::*;

    #[test]
    fn keyring_core_secret_store_round_trips_api_keys() {
        let store: Arc<CredentialStore> = keyring_core::mock::Store::new().unwrap();
        let secrets = KeyringCoreSecretStore::with_store("smolgent-test", store);

        assert_eq!(secrets.get_api_key("openrouter").unwrap(), None);

        secrets.set_api_key("openrouter", "sk-test").unwrap();
        assert_eq!(
            secrets.get_api_key("openrouter").unwrap(),
            Some("sk-test".to_string())
        );

        secrets.delete_api_key("openrouter").unwrap();
        assert_eq!(secrets.get_api_key("openrouter").unwrap(), None);
    }
}
