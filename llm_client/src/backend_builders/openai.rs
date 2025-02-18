use crate::LlmClient;
use llm_devices::{LoggingConfig, LoggingConfigTrait};
use llm_interface::llms::{ApiConfig, LlmApiConfigTrait, LlmBackend, OpenAiBackend, OpenAiConfig};
use llm_models::{
    api_models::{ApiLlmModel, OpenAiModelTrait},
    ApiLlmPreset,
};

pub struct OpenAiBackendBuilder {
    pub config: OpenAiConfig,
    pub model: ApiLlmModel,
}

impl Default for OpenAiBackendBuilder {
    fn default() -> Self {
        Self {
            config: Default::default(),
            model: ApiLlmModel::from_preset(ApiLlmPreset::GPT_4O_MINI),
        }
    }
}

impl OpenAiBackendBuilder {
    pub fn init(self) -> crate::Result<LlmClient> {
        Ok(LlmClient::new(std::sync::Arc::new(LlmBackend::OpenAi(
            OpenAiBackend::new(self.config, self.model)?,
        ))))
    }
}

impl LlmApiConfigTrait for OpenAiBackendBuilder {
    fn api_base_config_mut(&mut self) -> &mut ApiConfig {
        &mut self.config.api_config
    }

    fn api_config(&self) -> &ApiConfig {
        &self.config.api_config
    }
}

impl OpenAiModelTrait for OpenAiBackendBuilder {
    fn model(&mut self) -> &mut ApiLlmModel {
        &mut self.model
    }
}

impl LoggingConfigTrait for OpenAiBackendBuilder {
    fn logging_config_mut(&mut self) -> &mut LoggingConfig {
        &mut self.config.logging_config
    }
}
