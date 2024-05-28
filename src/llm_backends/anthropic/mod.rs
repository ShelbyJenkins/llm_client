use crate::{logging, LlmBackend, LlmClient, RequestConfig};
use anyhow::{anyhow, Result};
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
use dotenv::dotenv;
use llm_utils::{models::anthropic::AnthropicModel, tokenizer::LlmUtilsTokenizer};
use std::time::Duration;
use tokio::time::sleep;

pub struct AnthropicBackend {
    client: Option<AnthropicClient>,
    api_key: Option<String>,
    pub model: AnthropicModel,
    pub tokenizer: LlmUtilsTokenizer,
    pub logging_enabled: bool,
    tracing_guard: Option<tracing::subscriber::DefaultGuard>,
}

impl Default for AnthropicBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl AnthropicBackend {
    pub fn new() -> Self {
        AnthropicBackend {
            client: None,
            api_key: None,
            model: AnthropicModel::claude_3_haiku(),
            // Anthropic does not have a public tokenizer. Since we're just counting tokens, tiktoken will be close enough.
            tokenizer: LlmUtilsTokenizer::new_tiktoken("gpt-4"),
            logging_enabled: true,
            tracing_guard: None,
        }
    }

    fn setup(&mut self) {
        if self.client.is_some() {
            return;
        }
        let api_key = if let Some(api_key) = &self.api_key {
            api_key.to_owned()
        } else {
            if self.logging_enabled {
                tracing::info!("anthropic_backend api_key not set. Attempting to load from .env");
            } else {
                println!("anthropic_backend api_key not set. Attempting to load from .env");
            }
            dotenv().ok();
            if let Ok(api_key) = dotenv::var("ANTHROPIC_API_KEY") {
                api_key
            } else {
                if self.logging_enabled {
                    tracing::info!(
                        "ANTHROPIC_API_KEY not fund in in dotenv, nor was it set manually."
                    );
                }
                panic!("ANTHROPIC_API_KEY not fund in in dotenv, nor was it set manually.");
            }
        };
        let client = AnthropicClientBuilder::new(ApiKey::new(api_key))
            .client(
                ReqwestClientBuilder::new()
                    .timeout(std::time::Duration::from_secs(10))
                    .build()
                    .unwrap(),
            )
            .build();
        self.client = Some(client);
    }

    /// Initializes the AnthropicBackend and returns the LlmClient for usage.
    pub fn init(mut self) -> Result<LlmClient> {
        if self.logging_enabled {
            self.tracing_guard = Some(logging::create_logger("anthropic_backend"));
        }
        self.setup();
        Ok(LlmClient::new(LlmBackend::Anthropic(self)))
    }

    fn client(&self) -> &AnthropicClient {
        self.client.as_ref().unwrap()
    }

    /// Set the API key for the OpenAI client. Otherwise it will attempt to load it from the .env file.
    pub fn api_key(mut self, api_key: &str) -> Self {
        self.api_key = Some(api_key.to_string());
        self
    }

    /// Set the model for the OpenAI client using the model_id string.
    pub fn from_model_id(mut self, model_id: &str) -> Self {
        self.model = AnthropicModel::anthropic_model_from_model_id(model_id);
        self
    }

    /// Use the Claude 3 Opus model for the Anthropic client.
    pub fn claude_3_opus(mut self) -> Self {
        self.model = AnthropicModel::claude_3_opus();
        self
    }

    /// Use the Claude 3 Sonnet model for the Anthropic client.
    pub fn claude_3_sonnet(mut self) -> Self {
        self.model = AnthropicModel::claude_3_sonnet();
        self
    }

    /// Use the Claude 3 Haiku model for the Anthropic client.
    pub fn claude_3_haiku(mut self) -> Self {
        self.model = AnthropicModel::claude_3_haiku();
        self
    }

    /// If set to false, will disable logging. By defaults logs to the logs dir.
    pub fn logging_enabled(mut self, logging_enabled: bool) -> Self {
        self.logging_enabled = logging_enabled;
        self
    }

    /// A function to create text completions from a given prompt. Called by various agents, and not meant to be called directly.
    pub async fn text_generation_request(&self, req_config: &RequestConfig) -> Result<String> {
        // Maybe will help with rate limiting
        sleep(Duration::from_millis(222)).await;
        let prompt = req_config.default_formatted_prompt.as_ref().unwrap();

        let request = MessagesRequestBuilder::new(model_id_to_enum(&self.model.model_id))
            .stream(StreamOption::ReturnOnce)
            .messages(vec![Message::user(prompt["user"]["content"].clone())])
            .system(SystemPrompt::new(prompt["system"]["content"].clone()))
            .max_tokens(
                MaxTokens::new(
                    req_config.actual_request_tokens.unwrap(),
                    model_id_to_enum(&self.model.model_id),
                )
                .map_err(|e| anyhow!(e))?,
            )
            .temperature(
                Temperature::new(convert_temperature(req_config.temperature))
                    .map_err(|e| anyhow!(e))?,
            )
            .top_p(TopP::new(req_config.top_p).map_err(|e| anyhow!(e))?)
            .build();

        if self.logging_enabled {
            tracing::info!(?request);
        }
        let max_retries = 3;
        let mut retry_count = 0;
        loop {
            match self.client().create_a_message(request.clone()).await {
                Ok(response) => match &response.content {
                    Content::SingleText(content) => {
                        if self.logging_enabled {
                            tracing::info!(?response);
                        }
                        return Ok(content.to_owned());
                    }
                    Content::MultipleBlocks(blocks) => {
                        if self.logging_enabled {
                            tracing::info!(?response);
                        }
                        if blocks.len() == 1 {
                            match &blocks[0] {
                                ContentBlock::Text(content) => {
                                    return Ok(content.text.to_owned());
                                }
                                _ => panic!("Images not supported"),
                            }
                        } else {
                            panic!("MultipleBlocks not supported")
                        }
                    }
                },
                Err(e) => {
                    retry_count += 1;

                    if retry_count <= max_retries {
                        let backoff_duration = 2u64.pow(retry_count as u32) * 1000;
                        let error = anyhow::format_err!(
                            "AnthropicBackend text_generation_request error (attempt {}/{}): {}",
                            retry_count,
                            max_retries,
                            e
                        );

                        if self.logging_enabled {
                            tracing::warn!(?error, "Retrying after {} ms", backoff_duration);
                        }

                        sleep(Duration::from_millis(backoff_duration)).await;
                        continue;
                    } else {
                        let error = anyhow::format_err!("AnthropicBackend text_generation_request error (max retries exceeded): {}", e);

                        if self.logging_enabled {
                            tracing::error!(?error);
                        }

                        return Err(error);
                    }
                }
            }
        }
    }
}

fn model_id_to_enum(model_id: &str) -> ClaudeModel {
    if model_id.starts_with("claude-3-opus") {
        ClaudeModel::Claude3Opus20240229
    } else if model_id.starts_with("claude-3-sonnet") {
        ClaudeModel::Claude3Sonnet20240229
    } else if model_id.starts_with("claude-3-haiku") {
        ClaudeModel::Claude3Haiku20240307
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
