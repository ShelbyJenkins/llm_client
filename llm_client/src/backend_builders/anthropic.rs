use crate::LlmClient;
use llm_devices::logging::{LoggingConfig, LoggingConfigTrait};
use llm_interface::llms::{
    api::{
        anthropic::{AnthropicBackend, AnthropicConfig},
        config::{ApiConfig, LlmApiConfigTrait},
    },
    LlmBackend,
};
use llm_models::api_model::{anthropic::AnthropicModelTrait, ApiLlmModel};

// Everything here can be implemented for any struct.
pub struct AnthropicBackendBuilder {
    pub config: AnthropicConfig,
    pub model: ApiLlmModel,
}

impl Default for AnthropicBackendBuilder {
    fn default() -> Self {
        Self {
            config: Default::default(),
            model: ApiLlmModel::claude_3_5_sonnet(),
        }
    }
}

impl AnthropicBackendBuilder {
    pub fn init(self) -> crate::Result<LlmClient> {
        Ok(LlmClient::new(std::sync::Arc::new(LlmBackend::Anthropic(
            AnthropicBackend::new(self.config, self.model)?,
        ))))
    }
}

impl LlmApiConfigTrait for AnthropicBackendBuilder {
    fn api_base_config_mut(&mut self) -> &mut ApiConfig {
        &mut self.config.api_config
    }

    fn api_config(&self) -> &ApiConfig {
        &self.config.api_config
    }
}

impl AnthropicModelTrait for AnthropicBackendBuilder {
    fn model(&mut self) -> &mut ApiLlmModel {
        &mut self.model
    }
}

impl LoggingConfigTrait for AnthropicBackendBuilder {
    fn logging_config_mut(&mut self) -> &mut LoggingConfig {
        &mut self.config.logging_config
    }
}
