use super::{OpenAiBackend, OpenAiConfig};
use crate::llms::{
    api::config::{ApiConfig, LlmApiConfigTrait},
    LlmBackend,
};
use llm_devices::logging::{LoggingConfig, LoggingConfigTrait};
use llm_models::api_model::{openai::OpenAiModelTrait, ApiLlmModel};
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
    pub fn init(self) -> crate::Result<std::sync::Arc<LlmBackend>> {
        Ok(std::sync::Arc::new(LlmBackend::OpenAi(OpenAiBackend::new(
            self.config,
            self.model,
        )?)))
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

#[cfg(test)]
mod tests {
    use crate::{requests::completion::request::CompletionRequest, LlmInterface};
    use serial_test::serial;

    #[tokio::test]
    #[serial]
    async fn test_openai() {
        let backend = LlmInterface::openai().init().unwrap();
        let mut req = CompletionRequest::new(backend);
        req.prompt
            .add_user_message()
            .unwrap()
            .set_content("Hello, world!");

        let res = req.request().await.unwrap();
        println!("{res}");
    }
}
