pub mod builder;
pub mod completion;
pub mod server;

use super::LocalLlmConfig;
use crate::{
    llms::api::{
        client::ApiClient,
        config::{ApiConfig, ApiConfigTrait},
    },
    requests::completion::{
        error::CompletionError, request::CompletionRequest, response::CompletionResponse,
    },
};
use completion::LlamaCppCompletionRequest;
use llm_devices::logging::LoggingConfig;
use llm_models::local_model::{gguf::GgufLoader, LocalLlmModel};
use reqwest::header::{HeaderMap, AUTHORIZATION};
use secrecy::{ExposeSecret, Secret};
use server::LlamaCppServer;

pub const LLAMA_CPP_API_HOST: &str = "localhost";
pub const LLAMA_CPP_API_PORT: &str = "8080";

pub struct LlamaCppBackend {
    pub model: LocalLlmModel,
    pub server: LlamaCppServer,
    pub(crate) client: ApiClient<LlamaCppConfig>,
}

impl LlamaCppBackend {
    pub async fn new(
        mut config: LlamaCppConfig,
        mut local_config: LocalLlmConfig,
        llm_loader: GgufLoader,
    ) -> crate::Result<Self> {
        config.logging_config.load_logger()?;
        if let Ok(api_key) = config.api_config.load_api_key() {
            config.api_config.api_key = Some(api_key);
        }
        local_config.device_config.initialize()?;
        let model = local_config.load_model(llm_loader)?;

        let mut server = LlamaCppServer::new(
            local_config.device_config,
            &config.api_config.host,
            &config.api_config.port,
            local_config.inference_ctx_size,
        )?;
        let client: ApiClient<LlamaCppConfig> = ApiClient::new(config);
        server.start_server(&client).await?;
        println!(
            "{} with model: {}",
            colorful::Colorful::bold(colorful::Colorful::color(
                "LlamaCppBackend Initialized",
                colorful::RGB::new(220, 0, 115)
            )),
            model.model_base.model_id
        );
        Ok(Self {
            client,
            server,
            model,
        })
    }

    pub(crate) async fn completion_request(
        &self,
        request: &CompletionRequest,
    ) -> crate::Result<CompletionResponse, CompletionError> {
        match self
            .client
            .post("/completion", LlamaCppCompletionRequest::new(request)?)
            .await
        {
            Err(e) => Err(CompletionError::ClientError(e)),
            Ok(res) => Ok(CompletionResponse::new_from_llama(request, res)?),
        }
    }

    pub(crate) fn shutdown(&self) {
        match self.server.shutdown() {
            Ok(_) => (),
            Err(e) => crate::error!("Failed to shutdown server: {}", e),
        }
    }
}

#[derive(Clone, Debug)]
pub struct LlamaCppConfig {
    pub api_config: ApiConfig,
    pub logging_config: LoggingConfig,
}

impl Default for LlamaCppConfig {
    fn default() -> Self {
        Self {
            api_config: ApiConfig {
                host: LLAMA_CPP_API_HOST.to_string(),
                port: Some(LLAMA_CPP_API_PORT.to_string()),
                api_key: None,
                api_key_env_var: "LLAMA_API_KEY".to_string(),
            },
            logging_config: LoggingConfig {
                logger_name: "llama_cpp".to_string(),
                ..Default::default()
            },
        }
    }
}

impl LlamaCppConfig {
    pub fn new() -> Self {
        Default::default()
    }
}

impl ApiConfigTrait for LlamaCppConfig {
    fn headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();

        if let Some(api_key) = self.api_key() {
            headers.insert(
                AUTHORIZATION,
                format!("Bearer {}", api_key.expose_secret())
                    .as_str()
                    .parse()
                    .unwrap(),
            );
        }

        headers
    }

    fn url(&self, path: &str) -> String {
        if let Some(port) = &self.api_config.port {
            format!("http://{}:{}{}", self.api_config.host, port, path)
        } else {
            format!("http://{}:{}", self.api_config.host, path)
        }
    }

    fn api_key(&self) -> &Option<Secret<String>> {
        &self.api_config.api_key
    }
}
