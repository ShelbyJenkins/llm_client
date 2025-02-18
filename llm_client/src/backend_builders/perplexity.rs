use crate::LlmClient;
use llm_devices::{LoggingConfig, LoggingConfigTrait};
use llm_interface::llms::{
    ApiConfig, GenericApiBackend, GenericApiConfig, LlmApiConfigTrait, LlmBackend,
};

use llm_models::{
    api_models::{ApiLlmModel, PerplexityModelTrait},
    ApiLlmPreset,
};

pub struct PerplexityBackendBuilder {
    pub config: GenericApiConfig,
    pub model: ApiLlmModel,
}

impl Default for PerplexityBackendBuilder {
    fn default() -> Self {
        let mut config = GenericApiConfig::default();
        config.api_config.host = "api.perplexity.ai".to_string();
        config.api_config.api_key_env_var = "PERPLEXITY_API_KEY".to_string();
        config.logging_config.logger_name = "perplexity".to_string();
        Self {
            config,
            model: ApiLlmModel::from_preset(ApiLlmPreset::SONAR),
        }
    }
}

impl PerplexityBackendBuilder {
    pub fn init(self) -> crate::Result<LlmClient> {
        Ok(LlmClient::new(std::sync::Arc::new(LlmBackend::GenericApi(
            GenericApiBackend::new(self.config, self.model)?,
        ))))
    }
}

impl PerplexityModelTrait for PerplexityBackendBuilder {
    fn model(&mut self) -> &mut ApiLlmModel {
        &mut self.model
    }
}

impl LlmApiConfigTrait for PerplexityBackendBuilder {
    fn api_base_config_mut(&mut self) -> &mut ApiConfig {
        &mut self.config.api_config
    }

    fn api_config(&self) -> &ApiConfig {
        &self.config.api_config
    }
}

impl LoggingConfigTrait for PerplexityBackendBuilder {
    fn logging_config_mut(&mut self) -> &mut LoggingConfig {
        &mut self.config.logging_config
    }
}
