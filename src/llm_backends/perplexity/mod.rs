use super::{LlmBackend, LlmClientApiBuilderTrait};
use crate::{
    components::{base_request::BaseLlmRequest, response::LlmClientResponse},
    logging,
    LlmClient,
    LlmClientResponseError,
};
use anyhow::Result;
use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestAssistantMessageArgs,
        ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequestArgs,
    },
    Client as OpenAiClient,
};
use llm_utils::models::api_model::{perplexity::PerplexityModelTrait, ApiLlm};
use std::rc::Rc;

const ENV_VAR_NAME: &str = "PERPLEXITY_API_KEY";
const API_BASE: &str = " https://api.perplexity.ai";
pub struct PerplexityBackend {
    client: OpenAiClient<OpenAIConfig>,
    pub model: ApiLlm,
    _tracing_guard: Option<tracing::subscriber::DefaultGuard>,
}

impl PerplexityBackend {
    /// A function to create text completions from a given prompt. Called by various agents, and not meant to be called directly.
    pub async fn llm_request(
        &self,
        base: &BaseLlmRequest,
    ) -> Result<LlmClientResponse, LlmClientResponseError> {
        let mut messages = Vec::new();
        if let Some(prompt_message) = &base.instruct_prompt.prompt.built_openai_prompt {
            for m in prompt_message {
                messages.push(match m.get("role").unwrap().as_str() {
                    "system" => ChatCompletionRequestSystemMessageArgs::default()
                        .content(m.get("content").unwrap().to_string())
                        .build()
                        .map_err(|e| LlmClientResponseError::RequestBuilderError {
                            error: format!("PerplexityBackend builder error: {}", e),
                        })?
                        .into(),
                    "user" => ChatCompletionRequestUserMessageArgs::default()
                        .content(m.get("content").unwrap().to_string())
                        .build()
                        .map_err(|e| LlmClientResponseError::RequestBuilderError {
                            error: format!("PerplexityBackend builder error: {}", e),
                        })?
                        .into(),
                    "assistant" => ChatCompletionRequestAssistantMessageArgs::default()
                        .content(m.get("content").unwrap().to_string())
                        .build()
                        .map_err(|e| LlmClientResponseError::RequestBuilderError {
                            error: format!("PerplexityBackend builder error: {}", e),
                        })?
                        .into(),
                    _ => {
                        panic!("Role not found");
                    }
                })
            }
        } else {
            panic!("Prompt not built");
        }

        let mut request_builder = CreateChatCompletionRequestArgs::default();
        request_builder
            .model(self.model.model_id.to_string())
            .messages(messages)
            .max_tokens(base.config.actual_request_tokens.unwrap() as u16)
            .frequency_penalty(1.0)
            .presence_penalty(base.config.presence_penalty)
            .temperature(base.config.temperature)
            .top_p(base.config.top_p);

        if let Some(logit_bias) = &base.logit_bias {
            request_builder.logit_bias(logit_bias.openai());
        }

        let request =
            request_builder
                .build()
                .map_err(|e| LlmClientResponseError::RequestBuilderError {
                    error: format!("PerplexityBackend builder error: {}", e),
                })?;
        tracing::info!(?request);

        match self.client.chat().create(request).await {
            Err(e) => Err(LlmClientResponseError::InferenceError {
                error: format!("PerplexityBackend request error: {}", e,),
            }),
            Ok(completion) => {
                tracing::info!(?completion);
                LlmClientResponse::new_from_openai(completion)
            }
        }
    }
}

pub struct PerplexityBackendBuilder {
    pub model: Option<ApiLlm>,
    pub logging_enabled: bool,
    pub api_key: Option<String>,
}

impl Default for PerplexityBackendBuilder {
    fn default() -> Self {
        PerplexityBackendBuilder {
            model: None,
            logging_enabled: true,
            api_key: None,
        }
    }
}

impl PerplexityBackendBuilder {
    pub fn new() -> Self {
        PerplexityBackendBuilder::default()
    }

    /// If set to false, will disable logging. By defaults logs to the logs dir.
    pub fn logging_enabled(mut self, logging_enabled: bool) -> Self {
        self.logging_enabled = logging_enabled;
        self
    }

    pub fn init(self) -> Result<LlmClient> {
        let _tracing_guard = if self.logging_enabled {
            Some(logging::create_logger("perplexity"))
        } else {
            None
        };
        let api_key = self.load_api_key(ENV_VAR_NAME)?;
        // let model = if let Some(model) = self.model {
        //     model
        // } else {
        //     panic!("Model not set");
        // };
        let model = ApiLlm::gpt_4();

        let backoff = backoff::ExponentialBackoffBuilder::new()
            .with_max_elapsed_time(Some(std::time::Duration::from_secs(60)))
            .build();
        let config = OpenAIConfig::new()
            .with_api_key(api_key)
            .with_api_base(API_BASE);
        let client = OpenAiClient::with_config(config).with_backoff(backoff);

        let backend = PerplexityBackend {
            client,
            model,
            _tracing_guard,
        };
        Ok(LlmClient {
            backend: Rc::new(LlmBackend::Perplexity(backend)),
        })
    }
}

impl PerplexityModelTrait for PerplexityBackendBuilder {
    fn model(&mut self) -> &mut Option<ApiLlm> {
        &mut self.model
    }
}

impl LlmClientApiBuilderTrait for PerplexityBackendBuilder {
    fn set_api_key(&mut self) -> &mut Option<String> {
        &mut self.api_key
    }

    fn get_api_key(&self) -> &Option<String> {
        &self.api_key
    }
}
