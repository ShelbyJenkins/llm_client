use super::{MistralRsBackend, MistralRsConfig};
use crate::llms::{
    local::{LlmLocalTrait, LocalLlmConfig},
    LlmBackend,
};
use llm_devices::logging::{LoggingConfig, LoggingConfigTrait};
use llm_models::local_model::{
    gguf::{loaders::preset::GgufPresetLoader, GgufLoader},
    GgufLoaderTrait, GgufPresetTrait, HfTokenTrait,
};

// Everything here can be implemented for any struct.
#[derive(Default)]
pub struct MistralRsBackendBuilder {
    pub config: MistralRsConfig,
    pub llm_loader: GgufLoader,
}

impl MistralRsBackendBuilder {
    pub async fn init(self) -> crate::Result<std::sync::Arc<LlmBackend>> {
        Ok(std::sync::Arc::new(LlmBackend::MistralRs(
            MistralRsBackend::new(self.config, self.llm_loader).await?,
        )))
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

#[cfg(test)]
mod tests {
    use crate::{
        llms::local::{CudaConfig, LlmLocalTrait},
        requests::completion::request::CompletionRequest,
        LlmInterface,
    };
    use serial_test::serial;

    #[tokio::test]
    #[serial]
    async fn test_auto_gpu() {
        let backend = LlmInterface::mistral_rs().init().await.unwrap();
        assert!(
            backend
                .mistral_rs()
                .unwrap()
                .config
                .local_config
                .device_config
                .gpu_count()
                > 0
        );
        let mut req = CompletionRequest::new(backend);
        req.prompt
            .add_user_message()
            .unwrap()
            .set_content("Hello, world!");

        let res = req.request().await.unwrap();
        println!("{res}");
    }

    #[tokio::test]
    #[serial]
    async fn test_single_gpu_map() {
        let cuda_config = CudaConfig::new_from_cuda_devices(vec![0]);

        let backend = LlmInterface::mistral_rs()
            .cuda_config(cuda_config)
            .init()
            .await
            .unwrap();
        assert!(
            backend
                .mistral_rs()
                .unwrap()
                .config
                .local_config
                .device_config
                .gpu_count()
                == 1
        );
        let mut req = CompletionRequest::new(backend);
        req.prompt
            .add_user_message()
            .unwrap()
            .set_content("Hello, world!");

        let res = req.request().await.unwrap();
        println!("{res}");
    }

    #[tokio::test]
    #[serial]
    async fn test_two_gpu_map() {
        let cuda_config = CudaConfig::new_from_cuda_devices(vec![0, 1]);

        let backend = LlmInterface::mistral_rs()
            .cuda_config(cuda_config)
            .init()
            .await
            .unwrap();
        assert!(
            backend
                .mistral_rs()
                .unwrap()
                .config
                .local_config
                .device_config
                .gpu_count()
                == 2
        );
        let mut req = CompletionRequest::new(backend);
        req.prompt
            .add_user_message()
            .unwrap()
            .set_content("Hello, world!");

        let res = req.request().await.unwrap();
        println!("{res}");
    }

    #[tokio::test]
    #[serial]
    async fn test_cpu_only() {
        let backend = LlmInterface::mistral_rs().cpu_only().init().await.unwrap();
        assert!(
            backend
                .mistral_rs()
                .unwrap()
                .config
                .local_config
                .device_config
                .gpu_count()
                == 0
        );
        let mut req = CompletionRequest::new(backend);
        req.prompt
            .add_user_message()
            .unwrap()
            .set_content("Hello, world!");

        let res = req.request().await.unwrap();
        println!("{res}");
    }
}
