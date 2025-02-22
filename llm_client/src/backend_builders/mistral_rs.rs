use crate::LlmClient;
use llm_devices::{LoggingConfig, LoggingConfigTrait};
use llm_interface::llms::{
    local::{
        mistral_rs::{MistralRsBackend, MistralRsConfig},
        LlmLocalTrait, LocalLlmConfig,
    },
    LlmBackend,
};
use llm_models::local_models::{
    gguf::{loaders::preset::GgufPresetLoader, GgufLoader},
    GgufLoaderTrait, GgufPresetTrait, HfTokenTrait,
};

#[derive(Default, Clone)]
pub struct MistralRsBackendBuilder {
    pub config: MistralRsConfig,
    pub llm_loader: GgufLoader,
}

impl MistralRsBackendBuilder {
    pub async fn init(self) -> crate::Result<LlmClient> {
        Ok(LlmClient::new(std::sync::Arc::new(LlmBackend::MistralRs(
            MistralRsBackend::new(self.config, self.llm_loader).await?,
        ))))
    }
}

impl LlmLocalTrait for MistralRsBackendBuilder {
    fn config(&mut self) -> &mut LocalLlmConfig {
        &mut self.config.local_config
    }
}

impl LoggingConfigTrait for MistralRsBackendBuilder {
    fn logging_config_mut(&mut self) -> &mut LoggingConfig {
        &mut self.config.logging_config
    }
}

impl GgufPresetTrait for MistralRsBackendBuilder {
    fn preset_loader(&mut self) -> &mut GgufPresetLoader {
        &mut self.llm_loader.gguf_preset_loader
    }
}

impl GgufLoaderTrait for MistralRsBackendBuilder {
    fn gguf_loader(&mut self) -> &mut GgufLoader {
        &mut self.llm_loader
    }
}

impl HfTokenTrait for MistralRsBackendBuilder {
    fn hf_token_mut(&mut self) -> &mut Option<String> {
        &mut self.llm_loader.hf_loader.hf_token
    }

    fn hf_token_env_var_mut(&mut self) -> &mut String {
        &mut self.llm_loader.hf_loader.hf_token_env_var
    }
}
