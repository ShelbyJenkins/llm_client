#[cfg(feature = "mistral_rs_backend")]
use llm_client::backend_builders::mistral_rs::MistralRsBackendBuilder;
use llm_client::{backend_builders::llama_cpp::LlamaCppBackendBuilder, LlmClient};

use llm_models::gguf_presets::GgufPreset;

const TINY_LLM_PRESET: GgufPreset = GgufPreset::LLAMA_3_2_3B_INSTRUCT;
const MEDIUM_LLM_PRESET: GgufPreset = GgufPreset::LLAMA_3_1_8B_INSTRUCT;
const LARGE_LLM_PRESET: GgufPreset = GgufPreset::MISTRAL_NEMO_INSTRUCT_2407;
const MAX_LLM_PRESET: GgufPreset = GgufPreset::MISTRAL_SMALL_24B_INSTRUCT_2501;
const DEFAULT_BACKEND: &str = "LlamaCpp";

#[derive(Clone)]
pub enum TestBackendConfig {
    #[cfg(feature = "llama_cpp_backend")]
    LlamaCpp(LlamaCppBackendBuilder),
    #[cfg(feature = "mistral_rs_backend")]
    MistralRs(MistralRsBackendBuilder),
}

impl TestBackendConfig {
    pub fn from_llama_cpp(builder: LlamaCppBackendBuilder) -> TestBackendConfig {
        TestBackendConfig::LlamaCpp(builder)
    }
    #[cfg(feature = "mistral_rs_backend")]
    pub fn from_mistral_rs(builder: MistralRsBackendBuilder) -> TestBackendConfig {
        TestBackendConfig::MistralRs(builder)
    }

    pub async fn to_llm_client_with_preset(&self, preset: &GgufPreset) -> crate::Result<LlmClient> {
        match self.clone() {
            #[cfg(feature = "llama_cpp_backend")]
            Self::LlamaCpp(mut b) => {
                b.llm_loader.gguf_preset_loader.llm_preset = preset.clone();
                b.init().await
            }
            #[cfg(feature = "mistral_rs_backend")]
            Self::MistralRs(mut b) => {
                b.llm_loader.gguf_preset_loader.llm_preset = preset.clone();
                b.init().await
            }
        }
    }

    pub fn to_string(&self) -> String {
        match self {
            #[cfg(feature = "llama_cpp_backend")]
            Self::LlamaCpp(_) => "LlamaCpp".to_string(),
            #[cfg(feature = "mistral_rs_backend")]
            Self::MistralRs(_) => "MistralRs".to_string(),
        }
    }

    fn default_from_str(backend: &str) -> TestBackendConfig {
        match backend {
            #[cfg(feature = "llama_cpp_backend")]
            "LlamaCpp" => TestBackendConfig::LlamaCpp(LlamaCppBackendBuilder::default()),
            #[cfg(feature = "mistral_rs_backend")]
            "MistralRs" => TestBackendConfig::MistralRs(MistralRsBackendBuilder::default()),

            _ => todo!(),
        }
    }

    pub fn default_llama_cpp() -> TestBackendConfig {
        TestBackendConfig::LlamaCpp(LlamaCppBackendBuilder::default())
    }

    #[cfg(feature = "mistral_rs_backend")]
    pub fn default_mistral_rs() -> TestBackendConfig {
        TestBackendConfig::MistralRs(MistralRsBackendBuilder::default())
    }
}

pub async fn default_tiny_llm() -> crate::Result<LlmClient> {
    TestBackendConfig::default_from_str(DEFAULT_BACKEND)
        .to_llm_client_with_preset(&TINY_LLM_PRESET)
        .await
}

pub async fn default_medium_llm() -> crate::Result<LlmClient> {
    TestBackendConfig::default_from_str(DEFAULT_BACKEND)
        .to_llm_client_with_preset(&MEDIUM_LLM_PRESET)
        .await
}

pub async fn default_large_llm() -> crate::Result<LlmClient> {
    TestBackendConfig::default_from_str(DEFAULT_BACKEND)
        .to_llm_client_with_preset(&LARGE_LLM_PRESET)
        .await
}

pub async fn default_max_llm() -> crate::Result<LlmClient> {
    TestBackendConfig::default_from_str(DEFAULT_BACKEND)
        .to_llm_client_with_preset(&MAX_LLM_PRESET)
        .await
}

#[cfg(feature = "llama_cpp_backend")]
pub async fn llama_cpp_from_preset(preset: &GgufPreset) -> crate::Result<LlmClient> {
    let mut builder = LlmClient::llama_cpp();
    builder.llm_loader.gguf_preset_loader.llm_preset = preset.clone();
    builder.init().await
}

#[cfg(feature = "llama_cpp_backend")]
pub async fn llama_cpp_tiny_llm() -> crate::Result<LlmClient> {
    llama_cpp_from_preset(&TINY_LLM_PRESET).await
}

#[cfg(feature = "llama_cpp_backend")]
pub async fn llama_cpp_medium_llm() -> crate::Result<LlmClient> {
    llama_cpp_from_preset(&MEDIUM_LLM_PRESET).await
}

#[cfg(feature = "llama_cpp_backend")]
pub async fn llama_cpp_large_llm() -> crate::Result<LlmClient> {
    llama_cpp_from_preset(&LARGE_LLM_PRESET).await
}

#[cfg(feature = "llama_cpp_backend")]
pub async fn llama_cpp_max_llm() -> crate::Result<LlmClient> {
    llama_cpp_from_preset(&MAX_LLM_PRESET).await
}

#[cfg(feature = "mistral_rs_backend")]
pub async fn mistral_rs_from_preset(preset: &GgufPreset) -> crate::Result<LlmClient> {
    let mut builder = LlmClient::mistral_rs();
    builder.llm_loader.gguf_preset_loader.llm_preset = preset.clone();
    builder.init().await
}

#[cfg(feature = "mistral_rs_backend")]
pub async fn mistral_rs_tiny_llm() -> crate::Result<LlmClient> {
    mistral_rs_from_preset(&TINY_LLM_PRESET).await
}

#[cfg(feature = "mistral_rs_backend")]
pub async fn mistral_rs_medium_llm() -> crate::Result<LlmClient> {
    mistral_rs_from_preset(&MEDIUM_LLM_PRESET).await
}

#[cfg(feature = "mistral_rs_backend")]
pub async fn mistral_rs_large_llm() -> crate::Result<LlmClient> {
    mistral_rs_from_preset(&LARGE_LLM_PRESET).await
}

#[cfg(feature = "mistral_rs_backend")]
pub async fn mistral_rs_max_llm() -> crate::Result<LlmClient> {
    mistral_rs_from_preset(&MAX_LLM_PRESET).await
}
