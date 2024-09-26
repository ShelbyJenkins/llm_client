pub mod backend_builders;
pub mod basic_completion;
pub mod components;
pub mod prelude;
pub mod primitives;
pub mod workflows;
#[allow(unused_imports)]
pub(crate) use anyhow::{anyhow, bail, Result};
pub use prelude::*;
#[allow(unused_imports)]
pub(crate) use tracing::{debug, error, info, span, trace, warn, Level};

pub struct LlmClient {
    pub backend: std::sync::Arc<llm_interface::llms::LlmBackend>,
}

impl LlmClient {
    pub fn new(backend: std::sync::Arc<llm_interface::llms::LlmBackend>) -> Self {
        Self { backend }
    }
    #[cfg(feature = "llama_cpp_backend")]
    /// Creates a new instance of the [`LlamaCppBackendBuilder`]. This builder that allows you to specify the model and other parameters. It is converted to an `LlmClient` instance using the `init` method.
    pub fn llama_cpp() -> backend_builders::llama_cpp::LlamaCppBackendBuilder {
        backend_builders::llama_cpp::LlamaCppBackendBuilder::default()
    }

    #[cfg(feature = "mistral_rs_backend")]
    /// Creates a new instance of the [`MistralRsBackendBuilder`] This builder that allows you to specify the model and other parameters. It is converted to an `LlmClient` instance using the `init` method.
    pub fn mistral_rs() -> backend_builders::mistral_rs::MistralRsBackendBuilder {
        backend_builders::mistral_rs::MistralRsBackendBuilder::default()
    }

    /// Creates a new instance of the [`OpenAiBackendBuilder`]. This builder that allows you to specify the model and other parameters. It is converted to an `LlmClient` instance using the `init` method.
    pub fn openai() -> backend_builders::openai::OpenAiBackendBuilder {
        backend_builders::openai::OpenAiBackendBuilder::default()
    }

    /// Creates a new instance of the [`AnthropicBackendBuilder`]. This builder that allows you to specify the model and other parameters. It is converted to an `LlmClient` instance using the `init` method.
    pub fn anthropic() -> backend_builders::anthropic::AnthropicBackendBuilder {
        backend_builders::anthropic::AnthropicBackendBuilder::default()
    }

    /// Creates a new instance of the [`PerplexityBackendBuilder`]. This builder that allows you to specify the model and other parameters. It is converted to an `LlmClient` instance using the `init` method.
    pub fn perplexity() -> backend_builders::perplexity::PerplexityBackendBuilder {
        backend_builders::perplexity::PerplexityBackendBuilder::default()
    }

    pub fn basic_completion(&self) -> basic_completion::BasicCompletion {
        basic_completion::BasicCompletion::new(self.backend.clone())
    }

    pub fn basic_primitive(&self) -> workflows::basic_primitive::BasicPrimitiveWorkflowBuilder {
        workflows::basic_primitive::BasicPrimitiveWorkflowBuilder::new(self.backend.clone())
    }

    pub fn reason(&self) -> workflows::reason::ReasonWorkflowBuilder {
        workflows::reason::ReasonWorkflowBuilder::new(self.backend.clone())
    }

    pub fn nlp(&self) -> workflows::nlp::Nlp {
        workflows::nlp::Nlp::new(self.backend.clone())
    }

    pub fn shutdown(&self) {
        self.backend.shutdown();
    }
}
