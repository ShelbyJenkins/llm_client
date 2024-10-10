#[allow(unused_imports)]
pub(crate) use anyhow::{Error, anyhow, bail, Result};
#[allow(unused_imports)]
pub(crate) use tracing::{debug, error, info, span, trace, warn, Level};

pub mod llms;
pub mod requests;

pub struct LlmInterface {}

// These are examples and bare minimum implementations. For full featured implementation see the llm_client crate.
impl LlmInterface {
    #[cfg(feature = "llama_cpp_backend")]
    pub fn llama_cpp() -> llms::local::llama_cpp::builder::LlamaCppBackendBuilder {
        llms::local::llama_cpp::builder::LlamaCppBackendBuilder::default()
    }

    #[cfg(feature = "mistral_rs_backend")]
    pub fn mistral_rs() -> llms::local::mistral_rs::builder::MistralRsBackendBuilder {
        llms::local::mistral_rs::builder::MistralRsBackendBuilder::default()
    }

    pub fn openai() -> llms::api::openai::builder::OpenAiBackendBuilder {
        llms::api::openai::builder::OpenAiBackendBuilder::default()
    }

    pub fn anthropic() -> llms::api::anthropic::builder::AnthropicBackendBuilder {
        llms::api::anthropic::builder::AnthropicBackendBuilder::default()
    }

    pub fn perplexity() -> llms::api::perplexity::builder::PerplexityBackendBuilder {
        llms::api::perplexity::builder::PerplexityBackendBuilder::default()
    }
}
