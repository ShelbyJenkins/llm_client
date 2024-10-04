use crate::LlmClient;
use llm_devices::logging::{LoggingConfig, LoggingConfigTrait};
use llm_interface::llms::{
    api::{
        config::{ApiConfig, LlmApiConfigTrait},
        openai::{OpenAiBackend, OpenAiConfig},
    },
    LlmBackend,
};
use llm_utils::models::api_model::{openai::OpenAiModelTrait, ApiLlmModel};

// Everything here can be implemented for any struct.
pub struct OpenAiBackendBuilder {
    pub config: OpenAiConfig,
    pub model: ApiLlmModel,
}

impl Default for OpenAiBackendBuilder {
    fn default() -> Self {
        Self {
            config: Default::default(),
            model: ApiLlmModel::gpt_4_o_mini(),
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
