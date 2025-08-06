//! High‑level, composable configuration and builder helpers for
//! [`HfClient`].  Ordinary users only need the `HfConfigExt` trait:
//!
//! Down‑stream crates that *embed* an `HfConfig` inside a larger
//! builder can opt‑in by implementing the hidden
//! [`internal::ConfigSlot`] trait:

use secrecy::SecretString;

use crate::{fs::dir::StorageLocation, hf::client::HfClient};

/// Default name of the environment variable consulted for a token.
pub const DEFAULT_ENV_VAR: &str = "HUGGING_FACE_TOKEN";

/// Configuration bundle for a future [`HfClient`].
#[derive(Clone, Debug)]
pub struct HfConfig {
    /// Explicit token.  
    /// If `None`, the client will attempt to load it from
    /// [`token_env_var`].
    pub token: Option<SecretString>,

    /// Name of the environment variable used when [`token`] is `None`.
    pub token_env_var: String,

    /// Where to cache downloaded artefacts (optional).
    pub storage_location: Option<StorageLocation>,

    /// Emit progress bars during transfers.
    pub progress: bool,

    /// Override the retry limit for network calls (`None` = library default).
    pub max_retries: Option<usize>,

    /// Custom endpoint (useful for on‑prem mirrors).
    pub endpoint: Option<String>,
}

impl Default for HfConfig {
    fn default() -> Self {
        Self {
            token: None,
            token_env_var: DEFAULT_ENV_VAR.to_string(),
            storage_location: None,
            progress: true,
            max_retries: None,
            endpoint: None,
        }
    }
}

impl HfConfig {
    pub fn build(self) -> HfClient {
        HfClient::new(self)
    }
}

#[doc(hidden)]
pub mod internal {
    /// Gives mutable access to an embedded [`HfConfig`].
    ///
    /// Down‑stream crates implement this trait for their own builders but
    /// **should not** re‑export it.  End‑users therefore never see or call
    /// [`cfg`]; they interact exclusively through [`HfConfigExt`].
    pub trait HasHfConfig {
        fn cfg(&mut self) -> &mut super::HfConfig;
    }
}

// `HfConfig` itself obviously owns a config slot.
impl internal::HasHfConfig for HfConfig {
    #[inline]
    fn cfg(&mut self) -> &mut HfConfig {
        self
    }
}

/// High‑level setters for [`HfConfig`].
///
/// Implemented automatically for every type that implements
/// [`internal::HasHfConfig`].
pub trait HfConfigExt: internal::HasHfConfig + Sized {
    /// Sets the token explicitly, bypassing environment‑variable lookup.
    #[must_use]
    fn with_token(mut self, t: impl Into<SecretString>) -> Self {
        self.cfg().token = Some(t.into());
        self
    }

    /// Changes which environment variable is inspected when
    /// [`with_token`] has **not** been called.
    #[must_use]
    fn with_token_env_var(mut self, var: impl Into<String>) -> Self {
        self.cfg().token_env_var = var.into();
        self
    }

    /// Chooses a custom download/cache location.
    #[must_use]
    fn with_storage_location(mut self, loc: StorageLocation) -> Self {
        self.cfg().storage_location = Some(loc);
        self
    }

    /// Toggles progress‑bar emission.
    #[must_use]
    fn with_progress(mut self, progress: bool) -> Self {
        self.cfg().progress = progress;
        self
    }

    /// Overrides the retry limit for network operations.
    #[must_use]
    fn with_max_retries(mut self, n: usize) -> Self {
        self.cfg().max_retries = Some(n);
        self
    }

    /// Uses a non‑standard Hugging Face endpoint (e.g. on‑prem mirror).
    #[must_use]
    fn with_endpoint(mut self, ep: impl Into<String>) -> Self {
        self.cfg().endpoint = Some(ep.into());
        self
    }
}

/// Blanket implementation: every *slot owner* automatically receives the
/// high‑level builder helpers.
impl<T: internal::HasHfConfig + Sized> HfConfigExt for T {}
