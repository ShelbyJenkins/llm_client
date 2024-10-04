use reqwest::header::HeaderMap;
use secrecy::Secret;

#[derive(Clone, Debug)]
pub struct ApiConfig {
    pub host: String,
    pub port: Option<String>,
    pub api_key: Option<Secret<String>>,
    pub api_key_env_var: String,
}

impl ApiConfig {
    pub(crate) fn load_api_key(&mut self) -> crate::Result<Secret<String>> {
        if let Some(api_key) = self.api_key.as_ref() {
            crate::trace!("Using api_key from parameter");
            return Ok(api_key.to_owned());
        }
        crate::trace!("api_key not set. Attempting to load from .env");
        dotenvy::dotenv().ok();

        match dotenvy::var(&self.api_key_env_var) {
            Ok(api_key) => {
                crate::trace!("Successfully loaded api_key from .env");
                Ok(api_key.into())
            }
            Err(_) => {
                crate::trace!(
                    "{} not found in dotenv, nor was it set manually",
                    self.api_key_env_var
                );
                crate::bail!("Failed to load api_key from parameter or .env")
            }
        }
    }
}

pub trait LlmApiConfigTrait {
    fn api_base_config_mut(&mut self) -> &mut ApiConfig;

    fn api_config(&self) -> &ApiConfig;

    fn with_api_host<S: AsRef<str>>(mut self, host: S) -> Self
    where
        Self: Sized,
    {
        self.api_base_config_mut().host = host.as_ref().to_string();
        self
    }

    fn with_api_port<S: AsRef<str>>(mut self, port: S) -> Self
    where
        Self: Sized,
    {
        self.api_base_config_mut().port = Some(port.as_ref().to_string());
        self
    }

    fn with_api_key<S: Into<String>>(mut self, api_key: S) -> Self
    where
        Self: Sized,
    {
        self.api_base_config_mut().api_key = Some(Secret::from(api_key.into()));
        self
    }

    /// Set the environment variable name for the API key. Default is set from the backend.
    fn with_api_key_env_var<S: Into<String>>(mut self, api_key_env_var: S) -> Self
    where
        Self: Sized,
    {
        self.api_base_config_mut().api_key_env_var = api_key_env_var.into();
        self
    }
}

pub(crate) trait ApiConfigTrait {
    fn headers(&self) -> HeaderMap;

    fn url(&self, path: &str) -> String;

    fn api_key(&self) -> &Option<Secret<String>>;
}
