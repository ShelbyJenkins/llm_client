#![feature(lazy_cell)]
pub mod basic_completion;
pub mod components;
pub mod llm_backends;
mod logging;
pub mod prelude;
pub mod primitives;
pub mod workflows;

pub(crate) use anyhow::{anyhow, bail, Result};
use llm_backends::LlmBackend;
pub use prelude::*;
pub(crate) use tracing::info;

pub struct LlmClient {
    pub backend: std::rc::Rc<LlmBackend>,
}

impl LlmClient {
    /// Creates a new instance of the [`LlamaBackendBuilder`]. This builder that allows you to specify the model and other parameters. It is converted to an `LlmClient` instance using the `init` method.
    pub fn llama_cpp() -> llm_backends::llama_cpp::LlamaBackendBuilder {
        llm_backends::llama_cpp::LlamaBackendBuilder::default()
    }

    #[cfg(feature = "mistralrs_backend")]
    /// Creates a new instance of the [`MistraRsBackendBuilder`] This builder that allows you to specify the model and other parameters. It is converted to an `LlmClient` instance using the `init` method.
    pub fn mistral_rs() -> MistraRsBackendBuilder {
        MistraRsBackendBuilder::default()
    }

    /// Creates a new instance of the [`OpenAiBackendBuilder`]. This builder that allows you to specify the model and other parameters. It is converted to an `LlmClient` instance using the `init` method.
    pub fn openai() -> llm_backends::openai::OpenAiBackendBuilder {
        llm_backends::openai::OpenAiBackendBuilder::default()
    }

    /// Creates a new instance of the [`AnthropicBackendBuilder`]. This builder that allows you to specify the model and other parameters. It is converted to an `LlmClient` instance using the `init` method.
    pub fn anthropic() -> llm_backends::anthropic::AnthropicBackendBuilder {
        llm_backends::anthropic::AnthropicBackendBuilder::default()
    }

    pub fn perplexity() -> llm_backends::perplexity::PerplexityBackendBuilder {
        llm_backends::perplexity::PerplexityBackendBuilder::default()
    }

    pub fn basic_completion(&self) -> basic_completion::BasicCompletion {
        basic_completion::BasicCompletion::new(&self.backend)
    }

    pub fn basic_primitive(&self) -> workflows::basic_primitive::BasicPrimitiveWorkflowBuilder {
        workflows::basic_primitive::BasicPrimitiveWorkflowBuilder::new(&self.backend)
    }

    pub fn reason(&self) -> workflows::reason::ReasonWorkflowBuilder {
        workflows::reason::ReasonWorkflowBuilder::new(&self.backend)
    }

    pub fn nlp(&self) -> workflows::nlp::Nlp {
        workflows::nlp::Nlp::new(&self.backend)
    }
}
