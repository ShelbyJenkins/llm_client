use crate::LlmClient;
use llm_devices::logging::{LoggingConfig, LoggingConfigTrait};
use llm_interface::llms::{
    api::config::{ApiConfig, LlmApiConfigTrait},
    local::{
        llama_cpp::{LlamaCppBackend, LlamaCppConfig},
        LlmLocalTrait, LocalLlmConfig,
    },
    LlmBackend,
};
use llm_models::local_model::{
    gguf::{loaders::preset::GgufPresetLoader, GgufLoader},
    GgufLoaderTrait, GgufPresetTrait, HfTokenTrait,
};

// Everything here can be implemented for any struct.
#[derive(Default, Clone)]
pub struct LlamaCppBackendBuilder {
    pub config: LlamaCppConfig,
    pub local_config: LocalLlmConfig,
    pub llm_loader: GgufLoader,
}

impl LlamaCppBackendBuilder {
    pub async fn init(self) -> crate::Result<LlmClient> {
        Ok(LlmClient::new(std::sync::Arc::new(LlmBackend::LlamaCpp(
            LlamaCppBackend::new(self.config, self.local_config, self.llm_loader).await?,
        ))))
    }
}

impl LlmLocalTrait for LlamaCppBackendBuilder {
    fn config(&mut self) -> &mut LocalLlmConfig {
        &mut self.local_config
    }
}

impl LlmApiConfigTrait for LlamaCppBackendBuilder {
    fn api_base_config_mut(&mut self) -> &mut ApiConfig {
        &mut self.config.api_config
    }

    fn api_config(&self) -> &ApiConfig {
        &self.config.api_config
    }
}

impl LoggingConfigTrait for LlamaCppBackendBuilder {
    fn logging_config_mut(&mut self) -> &mut LoggingConfig {
        &mut self.config.logging_config
    }
}

impl GgufPresetTrait for LlamaCppBackendBuilder {
    fn preset_loader(&mut self) -> &mut GgufPresetLoader {
        &mut self.llm_loader.gguf_preset_loader
    }
}

impl GgufLoaderTrait for LlamaCppBackendBuilder {
    fn gguf_loader(&mut self) -> &mut GgufLoader {
        &mut self.llm_loader
    }
}

impl HfTokenTrait for LlamaCppBackendBuilder {
    fn hf_token_mut(&mut self) -> &mut Option<String> {
        &mut self.llm_loader.hf_loader.hf_token
    }

    fn hf_token_env_var_mut(&mut self) -> &mut String {
        &mut self.llm_loader.hf_loader.hf_token_env_var
    }
}
