pub mod builder;
pub mod completion;

use super::{
    client::ApiClient,
    config::{ApiConfig, ApiConfigTrait},
};
use crate::requests::completion::{
    error::CompletionError, request::CompletionRequest, response::CompletionResponse,
};
use completion::AnthropicCompletionRequest;
use llm_devices::logging::LoggingConfig;
use llm_utils::models::api_model::ApiLlmModel;
use reqwest::header::HeaderMap;
use secrecy::{ExposeSecret, Secret};

/// Default v1 API base url
pub const ANTHROPIC_API_HOST: &str = "api.anthropic.com/v1";
/// Reguired version header
pub const ANTHROPIC_VERSION_HEADER: &str = "anthropic-version";
/// Optional beta header
pub const ANTHROPIC_BETA_HEADER: &str = "anthropic-beta";

pub struct AnthropicBackend {
    pub(crate) client: ApiClient<AnthropicConfig>,
    pub model: ApiLlmModel,
}

impl AnthropicBackend {
    pub fn new(mut config: AnthropicConfig, model: ApiLlmModel) -> crate::Result<Self> {
        config.logging_config.load_logger()?;
        config.api_config.api_key = Some(config.api_config.load_api_key()?);
        Ok(Self {
            client: ApiClient::new(config),
            model,
        })
    }
    pub(crate) async fn completion_request(
        &self,
        request: &CompletionRequest,
    ) -> crate::Result<CompletionResponse, CompletionError> {
        match self
            .client
            .post("/messages", AnthropicCompletionRequest::new(request)?)
            .await
        {
            Err(e) => Err(CompletionError::ClientError(e)),
            Ok(res) => Ok(CompletionResponse::new_from_anthropic(request, res)?),
        }
    }
}

#[derive(Clone, Debug)]
pub struct AnthropicConfig {
    pub api_config: ApiConfig,
    pub logging_config: LoggingConfig,
    pub anthropic_version: String,
    pub anthropic_beta: Option<String>,
}

impl Default for AnthropicConfig {
    fn default() -> Self {
        Self {
            api_config: ApiConfig {
                host: ANTHROPIC_API_HOST.to_string(),
                port: None,
                api_key: None,
                api_key_env_var: "ANTHROPIC_API_KEY".to_string(),
            },
            logging_config: LoggingConfig {
                logger_name: "anthropic".to_string(),
                ..Default::default()
            },
            anthropic_version: "2023-06-01".to_string(),
            anthropic_beta: None,
        }
    }
}

impl AnthropicConfig {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn with_anthropic_version<S: Into<String>>(mut self, version: S) -> Self {
        self.anthropic_version = version.into();
        self
    }

    pub fn with_anthropic_beta<S: Into<String>>(mut self, beta: S) -> Self {
        self.anthropic_beta = Some(beta.into());
        self
    }
}

impl ApiConfigTrait for AnthropicConfig {
    fn headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            ANTHROPIC_VERSION_HEADER,
            self.anthropic_version.as_str().parse().unwrap(),
        );

        if let Some(anthropic_beta) = &self.anthropic_beta {
            headers.insert(
                ANTHROPIC_BETA_HEADER,
                anthropic_beta.as_str().parse().unwrap(),
            );
        }

        if let Some(api_key) = self.api_key() {
            headers.insert(
                reqwest::header::HeaderName::from_static("x-api-key"),
                reqwest::header::HeaderValue::from_str(api_key.expose_secret()).unwrap(),
            );
        }

        headers
    }

    fn url(&self, path: &str) -> String {
        format!("https://{}{}", self.api_config.host, path)
    }

    fn api_key(&self) -> &Option<Secret<String>> {
        &self.api_config.api_key
    }
}
