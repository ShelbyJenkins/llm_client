pub mod api;
pub mod server;

use super::LlmBackend;
use crate::{
    components::{base_request::BaseLlmRequest, response::LlmClientResponse},
    logging,
    LlmClient,
    LlmClientResponseError,
};
use api::{
    client::LlamaClient,
    config::LlamaConfig,
    types::{LlamaCompletionsRequest, LlamaCompletionsRequestArgs},
};
use llm_utils::models::open_source_model::{
    gguf::{GgufLoader, GgufLoaderTrait},
    HfTokenTrait,
    LlmPresetLoader,
    LlmPresetTrait,
    OsLlm,
};
use server::LlamaServerConfig;

pub struct LlamaBackend {
    pub model: OsLlm,
    pub server_config: LlamaServerConfig,
    client: LlamaClient<LlamaConfig>,
    _tracing_guard: Option<tracing::subscriber::DefaultGuard>,
}

impl LlamaBackend {
    pub async fn llm_request(
        &self,
        base_req: &BaseLlmRequest,
    ) -> crate::Result<LlmClientResponse, LlmClientResponseError> {
        let mut request_builder = LlamaCompletionsRequestArgs::default();
        request_builder
            .prompt(
                base_req
                    .instruct_prompt
                    .prompt
                    .built_prompt_as_tokens
                    .clone()
                    .unwrap(),
            )
            .prompt_string(
                base_req
                    .instruct_prompt
                    .prompt
                    .built_chat_template_prompt
                    .clone()
                    .unwrap(),
            )
            .frequency_penalty(base_req.config.frequency_penalty)
            .presence_penalty(base_req.config.presence_penalty)
            .temperature(base_req.config.temperature)
            .top_p(base_req.config.top_p)
            .n_predict(base_req.config.actual_request_tokens.unwrap());

        if let Some(grammar_string) = &base_req.grammar_string {
            request_builder.grammar(grammar_string.clone());
        }

        if let Some(logit_bias) = &base_req.logit_bias {
            request_builder.logit_bias(logit_bias.llama());
        }

        if !base_req.stop_sequences.sequences.is_empty() {
            request_builder.stop(base_req.stop_sequences.to_vec());
        }

        if base_req.config.cache_prompt {
            request_builder.cache_prompt(true);
        }
        let request: LlamaCompletionsRequest =
            request_builder
                .build()
                .map_err(|e| LlmClientResponseError::RequestBuilderError {
                    error: format!("LlamaBackend builder error: {}", e),
                })?;

        tracing::info!(?request);

        match self.client.completions().create(request).await {
            Err(e) => Err(LlmClientResponseError::InferenceError {
                error: format!("LlamaBackend request error: {}", e,),
            }),
            Ok(completion) => {
                tracing::info!(?completion);
                let response_stop_word = base_req
                    .stop_sequences
                    .parse_string_response(&completion.stopping_word);
                LlmClientResponse::new_from_llama(completion, response_stop_word)
            }
        }
    }

    pub async fn set_cache(
        &self,
        clear: bool,
        base_req: &BaseLlmRequest,
    ) -> crate::Result<(), LlmClientResponseError> {
        let mut request_builder = LlamaCompletionsRequestArgs::default();
        request_builder.n_predict(0_u32);
        if clear {
            request_builder.cache_prompt(false).prompt(vec![0u32]);
        } else {
            request_builder
                .cache_prompt(true)
                .prompt(
                    base_req
                        .instruct_prompt
                        .prompt
                        .built_prompt_as_tokens
                        .clone()
                        .unwrap(),
                )
                .prompt_string(
                    base_req
                        .instruct_prompt
                        .prompt
                        .built_chat_template_prompt
                        .clone()
                        .unwrap(),
                );
        }
        let request: LlamaCompletionsRequest =
            request_builder
                .build()
                .map_err(|e| LlmClientResponseError::RequestBuilderError {
                    error: format!("LlamaBackend builder error: {}", e),
                })?;
        tracing::info!(?request);
        match self.client.completions().create(request).await {
            Err(e) => Err(LlmClientResponseError::InferenceError {
                error: format!("LlamaBackend request error: {}", e,),
            }),
            Ok(_) => Ok(()),
        }
    }
}

pub struct LlamaBackendBuilder {
    pub logging_enabled: bool,
    pub server_config: LlamaServerConfig,
}

impl Default for LlamaBackendBuilder {
    fn default() -> Self {
        LlamaBackendBuilder {
            logging_enabled: true,
            server_config: LlamaServerConfig::default(),
        }
    }
}

impl LlamaBackendBuilder {
    pub fn new() -> Self {
        LlamaBackendBuilder::default()
    }

    /// If set to false, will disable logging. By defaults logs to the logs dir.
    pub fn logging_enabled(mut self, logging_enabled: bool) -> Self {
        self.logging_enabled = logging_enabled;
        self
    }

    pub async fn init(mut self) -> crate::Result<LlmClient> {
        let _tracing_guard = if self.logging_enabled {
            Some(logging::create_logger("llama_cpp"))
        } else {
            None
        };
        let model = self.server_config.load_model()?;

        self.server_config
            .start_server(&model.local_model_path)
            .await?;

        let client = LlamaClient::new(&self.server_config.host, &self.server_config.port);

        Ok(LlmClient {
            backend: std::rc::Rc::new(LlmBackend::Llama(LlamaBackend {
                model,
                server_config: self.server_config,
                client,
                _tracing_guard,
            })),
        })
    }
}

impl LlmPresetTrait for LlamaBackendBuilder {
    fn preset_loader(&mut self) -> &mut LlmPresetLoader {
        &mut self.server_config.llm_loader.preset_loader
    }
}

impl GgufLoaderTrait for LlamaBackendBuilder {
    fn gguf_loader(&mut self) -> &mut GgufLoader {
        &mut self.server_config.llm_loader
    }
}

impl HfTokenTrait for LlamaBackendBuilder {
    fn hf_token_mut(&mut self) -> &mut Option<String> {
        &mut self.server_config.llm_loader.hf_loader.hf_token
    }

    fn hf_token_env_var_mut(&mut self) -> &mut String {
        &mut self.server_config.llm_loader.hf_loader.hf_token_env_var
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use serial_test::serial;
    #[tokio::test]
    #[serial]
    async fn test_builder() -> crate::Result<()> {
        let llm_client = LlamaBackendBuilder::new()
            .available_vram(40)
            .use_ctx_size(2048)
            .phi3_mini4k_instruct()
            .init()
            .await?;

        assert_eq!(llm_client.backend.model_id(), "Phi-3-mini-4k-instruct");
        Ok(())
    }
}
