use super::{LlamaCppBackend, LlamaCppConfig};
use crate::llms::{
    api::config::{ApiConfig, LlmApiConfigTrait},
    local::{LlmLocalTrait, LocalLlmConfig},
    LlmBackend,
};
use llm_devices::logging::{LoggingConfig, LoggingConfigTrait};
use llm_utils::models::local_model::{
    gguf::{loaders::preset::GgufPresetLoader, GgufLoader},
    GgufLoaderTrait, GgufPresetTrait, HfTokenTrait,
};
// Everything here can be implemented for any struct.
#[derive(Default)]
pub struct LlamaCppBackendBuilder {
    pub config: LlamaCppConfig,
    pub local_config: LocalLlmConfig,
    pub llm_loader: GgufLoader,
}

impl LlamaCppBackendBuilder {
    pub async fn init(self) -> crate::Result<std::sync::Arc<LlmBackend>> {
        Ok(std::sync::Arc::new(LlmBackend::LlamaCpp(
            LlamaCppBackend::new(self.config, self.local_config, self.llm_loader).await?,
        )))
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

#[cfg(test)]
mod tests {
        #[cfg(any(target_os = "linux", target_os = "windows"))]
    use llm_devices::devices::CudaConfig;
    #[cfg(target_os = "macos")]
    use llm_devices::devices::MetalConfig;

    use crate::{
        llms::local::LlmLocalTrait, requests::completion::request::CompletionRequest, LlmInterface,
    };
    use serial_test::serial;

    #[tokio::test]
    #[serial]
    async fn test_auto_gpu() {
        let backend = LlmInterface::llama_cpp().init().await.unwrap();
        assert!(
            backend
                .llama_cpp()
                .unwrap()
                .server
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

        #[cfg(any(target_os = "linux", target_os = "windows"))]
    #[tokio::test]
    #[serial]
    async fn test_single_gpu_map() {
        let cuda_config = CudaConfig::new_from_cuda_devices(vec![0]);

        let backend = LlmInterface::llama_cpp()
            .cuda_config(cuda_config)
            .init()
            .await
            .unwrap();
        assert!(
            backend
                .llama_cpp()
                .unwrap()
                .server
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

        #[cfg(any(target_os = "linux", target_os = "windows"))]
    #[tokio::test]
    #[serial]
    async fn test_two_gpu_map() {
        let cuda_config = CudaConfig::new_from_cuda_devices(vec![1, 2]);

        let backend = LlmInterface::llama_cpp()
            .cuda_config(cuda_config)
            .init()
            .await
            .unwrap();
        assert!(
            backend
                .llama_cpp()
                .unwrap()
                .server
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
        let backend = LlmInterface::llama_cpp().cpu_only().init().await.unwrap();
        assert!(
            backend
                .llama_cpp()
                .unwrap()
                .server
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

    #[cfg(target_os = "macos")]
    #[tokio::test]
    #[serial]
    async fn test_metal() {
        let metal_config = MetalConfig::new_from_ram_gb(5.0);
        let backend = LlmInterface::llama_cpp()
            .metal_config(metal_config)
            .init()
            .await
            .unwrap();
        assert!(
            backend
                .llama_cpp()
                .unwrap()
                .server
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
}
