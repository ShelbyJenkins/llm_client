pub mod agents;
pub mod benchmark;
pub mod llm_backends;
mod logging;
pub mod prelude;
#[cfg(feature = "mistralrs_backend")]
use llm_backends::mistral_rs::MistraRsBackend;
use llm_backends::{anthropic::AnthropicBackend, llama_cpp::LlamaBackend, openai::OpenAiBackend};
use llm_utils::tokenizer::LlmTokenizer;
pub use prelude::*;
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

    #[cfg(feature = "mistralrs_backend")]
    pub fn mistral_rs_backend() -> MistraRsBackend {
        MistraRsBackend::new()
    }

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
    #[cfg(feature = "mistralrs_backend")]
    MistralRs(MistraRsBackend),
    OpenAi(OpenAiBackend),
    Anthropic(AnthropicBackend),
}

impl LlmBackend {
    pub fn get_model_id(&self) -> String {
        match self {
            LlmBackend::Llama(backend) => backend.model.as_ref().unwrap().model_id.clone(),
            #[cfg(feature = "mistralrs_backend")]
            LlmBackend::MistralRs(backend) => backend.model.as_ref().unwrap().model_id.clone(),
            LlmBackend::OpenAi(backend) => backend.model.model_id.clone(),
            LlmBackend::Anthropic(backend) => backend.model.model_id.clone(),
        }
    }

    pub fn get_model_url(&self) -> String {
        match self {
            LlmBackend::Llama(backend) => backend.model.as_ref().unwrap().model_url.clone(),
            #[cfg(feature = "mistralrs_backend")]
            LlmBackend::MistralRs(backend) => backend.model.as_ref().unwrap().model_id.clone(),
            LlmBackend::OpenAi(backend) => backend.model.model_id.clone(),
            LlmBackend::Anthropic(backend) => backend.model.model_id.clone(),
        }
    }

    pub fn logging_enabled(&self) -> bool {
        match self {
            LlmBackend::Llama(backend) => backend.logging_enabled,
            #[cfg(feature = "mistralrs_backend")]
            LlmBackend::MistralRs(backend) => backend.logging_enabled,
            LlmBackend::OpenAi(backend) => backend.logging_enabled,
            LlmBackend::Anthropic(backend) => backend.logging_enabled,
        }
    }

    pub fn get_tokenizer(&self) -> &LlmTokenizer {
        match self {
            LlmBackend::Llama(backend) => {
                backend.model.as_ref().unwrap().tokenizer.as_ref().unwrap()
            }
            #[cfg(feature = "mistralrs_backend")]
            LlmBackend::MistralRs(backend) => {
                backend.model.as_ref().unwrap().tokenizer.as_ref().unwrap()
            }
            LlmBackend::OpenAi(backend) => backend.model.tokenizer.as_ref().unwrap(),
            LlmBackend::Anthropic(backend) => backend.model.tokenizer.as_ref().unwrap(),
        }
    }
}
