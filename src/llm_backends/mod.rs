pub mod anthropic;
pub mod llama_cpp;
#[cfg(feature = "mistralrs_backend")]
pub mod mistral_rs;
pub mod openai;
pub mod perplexity;

use crate::components::constraints::logit_bias::LogitBias;
use anthropic::AnthropicBackend;
use anyhow::{anyhow, Result};
use dotenv::dotenv;
use llama_cpp::LlamaBackend;
use llm_utils::{models::open_source_model::OsLlmChatTemplate, tokenizer::LlmTokenizer};
#[cfg(feature = "mistralrs_backend")]
use mistral_rs::MistraRsBackend;
use openai::OpenAiBackend;
use perplexity::PerplexityBackend;
use std::sync::Arc;
use tracing::{debug, error, info, span, Level};

pub enum LlmBackend {
    Llama(LlamaBackend),
    #[cfg(feature = "mistralrs_backend")]
    MistralRs(MistraRsBackend),
    OpenAi(OpenAiBackend),
    Anthropic(AnthropicBackend),
    Perplexity(PerplexityBackend),
}

impl LlmBackend {
    pub fn model_id(&self) -> String {
        match self {
            LlmBackend::Llama(b) => b.model.model_id.clone(),
            #[cfg(feature = "mistralrs_backend")]
            LlmBackend::MistralRs(b) => b.model.model_id.clone(),
            LlmBackend::OpenAi(b) => b.model.model_id.clone(),
            LlmBackend::Anthropic(b) => b.model.model_id.clone(),
            LlmBackend::Perplexity(b) => b.model.model_id.clone(),
        }
    }

    pub fn get_bos_eos(&self) -> (String, String) {
        match self {
            LlmBackend::Llama(b) => (
                b.model.chat_template.bos_token.clone(),
                b.model.chat_template.eos_token.clone(),
            ),
            #[cfg(feature = "mistralrs_backend")]
            LlmBackend::MistralRs(b) => (
                b.model.chat_template.bos_token.clone(),
                b.model.chat_template.eos_token.clone(),
            ),
            _ => unimplemented!("Chat template not supported for this backend"),
        }
    }

    pub fn get_chat_template(&self) -> OsLlmChatTemplate {
        match self {
            LlmBackend::Llama(b) => b.model.chat_template.clone(),
            #[cfg(feature = "mistralrs_backend")]
            LlmBackend::MistralRs(b) => b.model.chat_template.clone(),
            _ => unimplemented!("Chat template not supported for this backend"),
        }
    }

    pub fn get_tokenizer(&self) -> Arc<LlmTokenizer> {
        match self {
            LlmBackend::Llama(backend) => Arc::clone(&backend.model.tokenizer),
            #[cfg(feature = "mistralrs_backend")]
            LlmBackend::MistralRs(backend) => Arc::clone(&backend.model.tokenizer),
            LlmBackend::OpenAi(backend) => Arc::clone(&backend.model.tokenizer),
            LlmBackend::Anthropic(backend) => Arc::clone(&backend.model.tokenizer),
            LlmBackend::Perplexity(backend) => Arc::clone(&backend.model.tokenizer),
        }
    }

    pub fn get_logit_bias(&self) -> LogitBias {
        match self {
            LlmBackend::Llama(_) => LogitBias::new(self.get_tokenizer()),
            #[cfg(feature = "mistralrs_backend")]
            LlmBackend::MistralRs(_) => LogitBias::new(self.get_tokenizer()),
            LlmBackend::OpenAi(_) => LogitBias::new(self.get_tokenizer()),
            LlmBackend::Anthropic(_) => panic!("Anthropic backend does not support logit bias"),
            LlmBackend::Perplexity(_) => panic!("Perplexity backend does not support logit bias"),
        }
    }
}

pub trait LlmClientApiBuilderTrait {
    fn set_api_key(&mut self) -> &mut Option<String>;

    fn get_api_key(&self) -> &Option<String>;

    /// Set the API key for the client. Otherwise it will attempt to load it from the .env file.
    fn with_api_key<T: AsRef<str>>(mut self, api_key: T) -> Self
    where
        Self: Sized,
    {
        *self.set_api_key() = Some(api_key.as_ref().to_owned());
        self
    }

    fn load_api_key(&self, env_var_name: &str) -> Result<String> {
        let span = span!(Level::INFO, "load_api_key");
        let _enter = span.enter();

        if let Some(api_key) = self.get_api_key() {
            debug!("Using api_key from parameter");
            return Ok(api_key.to_owned());
        }
        info!("api_key not set. Attempting to load from .env");
        dotenv().ok();

        match dotenv::var(env_var_name) {
            Ok(api_key) => {
                debug!("Successfully loaded api_key from .env");
                Ok(api_key)
            }
            Err(_) => {
                error!("{env_var_name} not found in dotenv, nor was it set manually");
                Err(anyhow!("Failed to load api_key from parameter or .env"))
            }
        }
    }
}
