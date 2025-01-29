mod api;
mod llama_cpp;
#[cfg(feature = "mistral_rs_backend")]
mod mistral_rs;

#[cfg(any(target_os = "linux", target_os = "windows"))]
use llm_devices::CudaConfig;
#[cfg(target_os = "macos")]
use llm_devices::MetalConfig;
use llm_interface::{llms::*, requests::*};
use serial_test::serial;

pub struct LlmInterface {}

// These are examples and bare minimum implementations. For full featured implementation see the llm_client crate.
impl LlmInterface {
    #[cfg(feature = "llama_cpp_backend")]
    pub fn llama_cpp() -> LlamaCppBackendBuilder {
        LlamaCppBackendBuilder::default()
    }

    #[cfg(feature = "mistral_rs_backend")]
    pub fn mistral_rs() -> local::mistral_rs::builder::MistralRsBackendBuilder {
        local::mistral_rs::builder::MistralRsBackendBuilder::default()
    }

    pub fn openai() -> OpenAiBackendBuilder {
        OpenAiBackendBuilder::default()
    }

    pub fn anthropic() -> AnthropicBackendBuilder {
        AnthropicBackendBuilder::default()
    }

    pub fn perplexity() -> PerplexityBackendBuilder {
        PerplexityBackendBuilder::default()
    }
}
