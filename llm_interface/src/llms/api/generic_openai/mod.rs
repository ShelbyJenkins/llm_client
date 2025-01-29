// Internal imports
use super::*;
use openai::completion::req::OpenAiCompletionRequest;

pub struct GenericApiBackend {
    pub(crate) client: ApiClient<GenericApiConfig>,
    pub model: ApiLlmModel,
}

impl GenericApiBackend {
    pub fn new(mut config: GenericApiConfig, model: ApiLlmModel) -> crate::Result<Self> {
        config.logging_config.load_logger()?;
        if let Ok(api_key) = config.api_config.load_api_key() {
            config.api_config.api_key = Some(api_key);
        }
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
            .post(
                &self.client.config.completion_path,
                OpenAiCompletionRequest::new(request)?,
            )
            .await
        {
            Err(e) => Err(CompletionError::ClientError(e)),
            Ok(res) => Ok(CompletionResponse::new_from_openai(request, res)?),
        }
    }
}

#[derive(Clone, Debug)]
pub struct GenericApiConfig {
    pub api_config: ApiConfig,
    pub logging_config: LoggingConfig,
    pub completion_path: String,
}

impl Default for GenericApiConfig {
    fn default() -> Self {
        Self {
            api_config: ApiConfig {
                host: Default::default(),
                port: None,
                api_key: None,
                api_key_env_var: Default::default(),
            },
            logging_config: LoggingConfig {
                logger_name: "generic".to_string(),
                ..Default::default()
            },
            completion_path: "/chat/completions".to_string(),
        }
    }
}

impl GenericApiConfig {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn completion_path<S: Into<String>>(mut self, path: S) -> Self {
        self.completion_path = path.into();
        self
    }
}

impl ApiConfigTrait for GenericApiConfig {
    fn headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        if let Some(api_key) = self.api_key() {
            if let Ok(header_value) =
                HeaderValue::from_str(&format!("Bearer {}", api_key.expose_secret()))
            {
                headers.insert(AUTHORIZATION, header_value);
            } else {
                crate::error!("Failed to create header value from authorization value");
            }
        }

        headers
    }

    fn url(&self, path: &str) -> String {
        if let Some(port) = &self.api_config.port {
            format!("https://{}:{}{}", self.api_config.host, port, path)
        } else {
            format!("https://{}:{}", self.api_config.host, path)
        }
    }

    fn api_key(&self) -> &Option<Secret<String>> {
        &self.api_config.api_key
    }
}
