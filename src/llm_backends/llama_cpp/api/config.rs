//! Client configurations: [LlamaConfig] for OpenAI, [AzureConfig] for Azure OpenAI Service.
use reqwest::header::HeaderMap;
use serde::Deserialize;

/// Default v1 API base url
pub const OPENAI_AI_BASE: &str = "https://api.openai.com/v1";
/// Name for organization header
pub const OPENAI_ORGANIZATION_HEADER: &str = "OpenAI-Organization";

/// Calls to the Assistants API require that you pass a Beta header
pub const OPENAI_BETA_HEADER: &str = "OpenAI-Beta";

/// [crate::Client] relies on this for every API call on OpenAI
/// or Azure OpenAI service
pub trait Config: Clone {
    fn headers(&self) -> HeaderMap;
    fn url(&self, path: &str) -> String;
    fn query(&self) -> Vec<(&str, &str)>;

    fn api_base(&self) -> &str;
}

/// Configuration for OpenAI API
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct LlamaConfig {
    api_base: String,
}

impl Default for LlamaConfig {
    fn default() -> Self {
        Self {
            api_base: OPENAI_AI_BASE.to_string(),
        }
    }
}

impl LlamaConfig {
    /// Create client with default [OPENAI_AI_BASE] url and default API key from OPENAI_API_KEY env var
    pub fn new() -> Self {
        Default::default()
    }

    /// To use a API base url different from default [OPENAI_AI_BASE]
    pub fn with_api_base<S: Into<String>>(mut self, api_base: S) -> Self {
        self.api_base = api_base.into();
        self
    }
}

impl Config for LlamaConfig {
    fn headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();

        // hack for Assistants APIs
        // Calls to the Assistants API require that you pass a Beta header
        headers.insert(OPENAI_BETA_HEADER, "assistants=v1".parse().unwrap());

        headers
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.api_base, path)
    }

    fn api_base(&self) -> &str {
        &self.api_base
    }

    fn query(&self) -> Vec<(&str, &str)> {
        vec![]
    }
}
