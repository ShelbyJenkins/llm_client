pub mod agents;
pub mod benchmark;
pub mod llm_backends;
mod logging;
pub mod prelude;
use crate::llm_backends::openai::OpenAiBackend;
use llm_backends::anthropic::AnthropicBackend;
use llm_utils::{logit_bias, tokenizer::LlmUtilsTokenizer};
pub use prelude::*;
use std::collections::HashMap;
pub use RequestConfigTrait;

pub struct LlmClient {
    pub default_request_config: RequestConfig,
    pub backend: LlmBackend,
}

impl Default for LlmClient {
    fn default() -> Self {
        let backend = LlmBackend::Llama(LlamaBackend::new());
        LlmClient {
            default_request_config: RequestConfig::new(&backend),
            backend,
        }
    }
}

impl LlmClient {
    /// Creates a new `LlmClient` instance with the specified backend. This is called from the backend's `init` function.
    pub fn new(backend: LlmBackend) -> Self {
        LlmClient {
            default_request_config: RequestConfig::new(&backend),
            backend,
        }
    }

    /// Creates a new instance of the Llama backend. This is a builder that allows you to specify the model and other parameters. It is converted to an `LlmClient` instance using the `init` method.
    pub fn llama_backend() -> LlamaBackend {
        LlamaBackend::new()
    }

    // pub fn mistral_rs_backend() -> MistraRsBackend {
    //     MistraRsBackend::new()
    // }

    /// Creates a new instance of the OpenAI backend. This is a builder that allows you to specify the model and other parameters. It is converted to an `LlmClient` instance using the `init` method.
    pub fn openai_backend() -> OpenAiBackend {
        OpenAiBackend::new()
    }

    /// Creates a new instance of the Anthropic backend. This is a builder that allows you to specify the model and other parameters. It is converted to an `LlmClient` instance using the `init` method.
    pub fn anthropic_backend() -> AnthropicBackend {
        AnthropicBackend::new()
    }

    /// Returns a `TextGenerator` instance to access text generation request builders.
    pub fn text(&self) -> TextGenerator {
        TextGenerator::new(self)
    }

    /// Returns a `Decider` instance to access decider request builders.
    pub fn decider(&self) -> Decider {
        Decider::new(self)
    }
}

impl RequestConfigTrait for LlmClient {
    fn config_mut(&mut self) -> &mut RequestConfig {
        &mut self.default_request_config
    }
}

pub enum LlmBackend {
    Llama(LlamaBackend),
    // MistralRs(MistraRsBackend),
    OpenAi(OpenAiBackend),
    Anthropic(AnthropicBackend),
}

impl LlmBackend {
    pub fn get_model_id(&self) -> String {
        match self {
            LlmBackend::Llama(backend) => backend.model.as_ref().unwrap().model_id.clone(),
            // LlmBackend::MistralRs(backend) => backend.model.as_ref().unwrap().model_id.clone(),
            LlmBackend::OpenAi(backend) => backend.model.model_id.clone(),
            LlmBackend::Anthropic(backend) => backend.model.model_id.clone(),
        }
    }

    pub fn get_model_url(&self) -> String {
        match self {
            LlmBackend::Llama(backend) => backend.model.as_ref().unwrap().model_url.clone(),
            // LlmBackend::MistralRs(backend) => backend.model.as_ref().unwrap().model_id.clone(),
            LlmBackend::OpenAi(backend) => backend.model.model_id.clone(),
            LlmBackend::Anthropic(backend) => backend.model.model_id.clone(),
        }
    }

    pub fn logging_enabled(&self) -> bool {
        match self {
            LlmBackend::Llama(backend) => backend.logging_enabled,
            // LlmBackend::MistralRs(backend) => backend.logging_enabled,
            LlmBackend::OpenAi(backend) => backend.logging_enabled,
            LlmBackend::Anthropic(backend) => backend.logging_enabled,
        }
    }

    pub fn get_tokenizer(&self) -> &LlmUtilsTokenizer {
        match self {
            LlmBackend::Llama(_backend) => unimplemented!(),
            // LlmBackend::MistralRs(backend) => backend.tokenizer.as_ref().unwrap(),
            LlmBackend::OpenAi(backend) => backend.tokenizer.as_ref().unwrap(),
            LlmBackend::Anthropic(backend) => &backend.tokenizer,
        }
    }

    // This is all temporary until we have a proper way to build a tokenizer from a GGUF. Will be moved to llm_utils::tokenizer then.
    pub async fn try_into_single_token(&self, str: &str) -> anyhow::Result<u32> {
        match self {
            LlmBackend::Llama(backend) => backend.try_into_single_token(str).await,
            // LlmBackend::MistralRs(_) => self.get_tokenizer().try_into_single_token(str),
            LlmBackend::OpenAi(_) => self.get_tokenizer().try_into_single_token(str),
            LlmBackend::Anthropic(_) => self.get_tokenizer().try_into_single_token(str),
        }
    }
    pub async fn validate_logit_bias_token_ids(
        &self,
        logit_bias: &HashMap<u32, f32>,
    ) -> anyhow::Result<()> {
        match self {
            LlmBackend::Llama(backend) => backend.validate_logit_bias_token_ids(logit_bias).await,
            // LlmBackend::MistralRs(backend) => logit_bias::validate_logit_bias_token_ids(
            //     backend.tokenizer.as_ref().unwrap(),
            //     logit_bias,
            // ),
            LlmBackend::OpenAi(_) => {
                logit_bias::validate_logit_bias_token_ids(self.get_tokenizer(), logit_bias)
            }
            LlmBackend::Anthropic(_) => {
                logit_bias::validate_logit_bias_token_ids(self.get_tokenizer(), logit_bias)
            }
        }
    }
    pub async fn logit_bias_from_chars(
        &self,
        logit_bias: &HashMap<char, f32>,
    ) -> anyhow::Result<HashMap<u32, f32>> {
        match self {
            LlmBackend::Llama(backend) => backend.logit_bias_from_chars(logit_bias).await,
            // LlmBackend::MistralRs(backend) => {
            //     logit_bias::logit_bias_from_chars(backend.tokenizer.as_ref().unwrap(), logit_bias)
            // }
            LlmBackend::OpenAi(_) => {
                logit_bias::logit_bias_from_chars(self.get_tokenizer(), logit_bias)
            }
            LlmBackend::Anthropic(_) => {
                logit_bias::logit_bias_from_chars(self.get_tokenizer(), logit_bias)
            }
        }
    }
    pub async fn logit_bias_from_words(
        &self,
        logit_bias: &HashMap<String, f32>,
    ) -> anyhow::Result<HashMap<u32, f32>> {
        match self {
            LlmBackend::Llama(backend) => backend.logit_bias_from_words(logit_bias).await,
            // LlmBackend::MistralRs(backend) => {
            //     logit_bias::logit_bias_from_words(backend.tokenizer.as_ref().unwrap(), logit_bias)
            // }
            LlmBackend::OpenAi(_) => {
                logit_bias::logit_bias_from_words(self.get_tokenizer(), logit_bias)
            }
            LlmBackend::Anthropic(_) => {
                logit_bias::logit_bias_from_words(self.get_tokenizer(), logit_bias)
            }
        }
    }
    pub async fn logit_bias_from_texts(
        &self,
        logit_bias: &HashMap<String, f32>,
    ) -> anyhow::Result<HashMap<u32, f32>> {
        match self {
            LlmBackend::Llama(backend) => backend.logit_bias_from_texts(logit_bias).await,
            // LlmBackend::MistralRs(backend) => {
            //     logit_bias::logit_bias_from_texts(backend.tokenizer.as_ref().unwrap(), logit_bias)
            // }
            LlmBackend::OpenAi(_) => {
                logit_bias::logit_bias_from_texts(self.get_tokenizer(), logit_bias)
            }
            LlmBackend::Anthropic(_) => {
                logit_bias::logit_bias_from_texts(self.get_tokenizer(), logit_bias)
            }
        }
    }
}
