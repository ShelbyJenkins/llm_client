use super::{LlmBackend, LlmClientApiBuilderTrait};
use crate::{
    components::{base_request::BaseLlmRequest, response::LlmClientResponse},
    logging,
    LlmClient,
    LlmClientResponseError,
};
use anyhow::Result;
use clust::{
    messages::{
        ClaudeModel,
        Content,
        ContentBlock,
        MaxTokens,
        Message,
        MessagesRequestBuilder,
        StreamOption,
        SystemPrompt,
        Temperature,
        TopP,
    },
    reqwest::ClientBuilder as ReqwestClientBuilder,
    ApiKey,
    Client as AnthropicClient,
    ClientBuilder as AnthropicClientBuilder,
};
use llm_utils::models::api_model::{anthropic::AnthropicModelTrait, ApiLlm};
use std::{rc::Rc, time::Duration};
use tokio::time::sleep;

const ENV_VAR_NAME: &str = "ANTHROPIC_API_KEY";
pub struct AnthropicBackend {
    client: AnthropicClient,
    pub model: ApiLlm,
    _tracing_guard: Option<tracing::subscriber::DefaultGuard>,
}

impl AnthropicBackend {
    /// A function to create text completions from a given prompt. Called by various agents, and not meant to be called directly.
    pub async fn llm_request(
        &self,
        base: &BaseLlmRequest,
    ) -> Result<LlmClientResponse, LlmClientResponseError> {
        // Maybe will help with rate limiting
        sleep(Duration::from_millis(222)).await;
        let mut messages = Vec::new();
        let mut system_message = None;
        if let Some(prompt_message) = &base.instruct_prompt.prompt.built_openai_prompt {
            for m in prompt_message {
                match m.get("role").unwrap().as_str() {
                    "system" => {
                        system_message =
                            Some(SystemPrompt::new(m.get("content").unwrap().to_string()));
                    }
                    "user" => {
                        messages.push(Message::user(m.get("content").unwrap().to_string()));
                    }
                    "assistant" => {
                        messages.push(Message::assistant(m.get("content").unwrap().to_string()));
                    }
                    _ => {
                        panic!("Role not found");
                    }
                }
            }
        } else {
            panic!("Prompt not built");
        }

        let builder = MessagesRequestBuilder::new(model_id_to_enum(&self.model.model_id))
            .stream(StreamOption::ReturnOnce)
            .messages(messages)
            .max_tokens(
                MaxTokens::new(
                    base.config.actual_request_tokens.unwrap(),
                    model_id_to_enum(&self.model.model_id),
                )
                .map_err(|e| LlmClientResponseError::RequestBuilderError {
                    error: format!("AnthropicBackend builder error: {}", e),
                })?,
            )
            .temperature(
                Temperature::new(convert_temperature(base.config.temperature)).map_err(|e| {
                    LlmClientResponseError::RequestBuilderError {
                        error: format!("AnthropicBackend builder error: {}", e),
                    }
                })?,
            )
            .top_p(TopP::new(base.config.top_p).map_err(|e| {
                LlmClientResponseError::RequestBuilderError {
                    error: format!("AnthropicBackend builder error: {}", e),
                }
            })?);

        let builder = if let Some(system_message) = system_message {
            builder.system(system_message)
        } else {
            builder
        };
        let request = builder.build();

        tracing::info!(?request);

        match self.client.create_a_message(request).await {
            Ok(response) => match &response.content {
                Content::SingleText(content) => {
                    tracing::info!(?response);
                    let stop_word = if let Some(stop_sequence) = &response.stop_sequence {
                        base.stop_sequences
                            .parse_string_response(stop_sequence.to_string())
                    } else {
                        None
                    };

                    LlmClientResponse::new_from_anthropic(&response, content, stop_word)
                }
                Content::MultipleBlocks(blocks) => {
                    tracing::info!(?response);

                    if blocks.len() == 1 {
                        match &blocks[0] {
                            ContentBlock::Text(content) => {
                                let stop_word = if let Some(stop_sequence) = &response.stop_sequence
                                {
                                    base.stop_sequences
                                        .parse_string_response(stop_sequence.to_string())
                                } else {
                                    None
                                };
                                LlmClientResponse::new_from_anthropic(
                                    &response,
                                    &content.text,
                                    stop_word,
                                )
                            }
                            _ => panic!("Images not supported"),
                        }
                    } else {
                        panic!("MultipleBlocks not supported")
                    }
                }
            },
            Err(e) => Err(LlmClientResponseError::InferenceError {
                error: format!("AnthropicBackend request error: {}", e,),
            }),
        }
    }
}

pub struct AnthropicBackendBuilder {
    pub model: Option<ApiLlm>,
    pub logging_enabled: bool,
    pub api_key: Option<String>,
}

impl Default for AnthropicBackendBuilder {
    fn default() -> Self {
        AnthropicBackendBuilder {
            model: None,
            logging_enabled: true,
            api_key: None,
        }
    }
}

impl AnthropicBackendBuilder {
    pub fn new() -> Self {
        AnthropicBackendBuilder::default()
    }

    /// If set to false, will disable logging. By defaults logs to the logs dir.
    pub fn logging_enabled(mut self, logging_enabled: bool) -> Self {
        self.logging_enabled = logging_enabled;
        self
    }

    pub fn init(self) -> Result<LlmClient> {
        let _tracing_guard = if self.logging_enabled {
            Some(logging::create_logger("openai"))
        } else {
            None
        };
        let api_key = self.load_api_key(ENV_VAR_NAME)?;
        let model = if let Some(model) = self.model {
            model
        } else {
            panic!("Model not set");
        };

        let client = AnthropicClientBuilder::new(ApiKey::new(api_key))
            .client(
                ReqwestClientBuilder::new()
                    .timeout(std::time::Duration::from_secs(10))
                    .build()
                    .unwrap(),
            )
            .build();
        let backend = AnthropicBackend {
            client,
            model,
            _tracing_guard,
        };
        Ok(LlmClient {
            backend: Rc::new(LlmBackend::Anthropic(backend)),
        })
    }
}

impl AnthropicModelTrait for AnthropicBackendBuilder {
    fn model(&mut self) -> &mut Option<ApiLlm> {
        &mut self.model
    }
}

impl LlmClientApiBuilderTrait for AnthropicBackendBuilder {
    fn set_api_key(&mut self) -> &mut Option<String> {
        &mut self.api_key
    }

    fn get_api_key(&self) -> &Option<String> {
        &self.api_key
    }
}

fn model_id_to_enum(model_id: &str) -> ClaudeModel {
    if model_id.starts_with("claude-3-opus") {
        ClaudeModel::Claude3Opus20240229
    } else if model_id.starts_with("claude-3-sonnet") {
        ClaudeModel::Claude3Sonnet20240229
    } else if model_id.starts_with("claude-3-haiku") {
        ClaudeModel::Claude3Haiku20240307
    } else if model_id.starts_with("claude-3.5-sonnet") {
        ClaudeModel::Claude35Sonnet20240620
    } else {
        panic!("{model_id} not found");
    }
}

/// Convert temperature from 0.0 to 2.0 to 0.0 to 1.0
fn convert_temperature(value: f32) -> f32 {
    if !(0.0..=2.0).contains(&value) {
        panic!("Temperature should of been limited to 0.0 to 2.0 before this point!");
    }
    let clamped_value = value.clamp(0.0, 2.0);
    clamped_value / 2.0
}
