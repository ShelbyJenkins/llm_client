// #![allow(unused_imports)]
// #![allow(dead_code)]

#[allow(unused_imports)]
pub use anyhow::{anyhow, bail, Result};
use llm_client::prelude::*;
use llm_utils::models::local_model::gguf::preset::LlmPreset;
use serde::{Deserialize, Serialize};
use std::{fs::File, io::Read, path::PathBuf};
#[allow(unused_imports)]
pub use tracing::{debug, error, info, span, trace, warn, Level};
use url::Url;
pub mod speed_bench;
mod test_loader;
mod test_types;
pub use test_loader::*;
pub use test_types::*;

const PRINT_PRIMITIVE_RESULT: bool = false;
const PRINT_WORKFLOW_RESULT: bool = false;
const PRINT_PROMPT: bool = false;
const DEFAULT_BACKEND: &str = "llama_cpp";

const TINY_LLM_PRESET: LlmPreset = LlmPreset::Llama3_2_1bInstruct;
const MEDIUM_LLM_PRESET: LlmPreset = LlmPreset::Llama3_2_3bInstruct;
const LARGE_LLM_PRESET: LlmPreset = LlmPreset::Llama3_1_8bInstruct;
const MAX_LLM_PRESET: LlmPreset = LlmPreset::MistralNemoInstruct2407;

pub fn print_results<T: std::fmt::Debug>(
    prompt: &LlmPrompt,
    workflow_result: &Option<impl std::fmt::Display>,
    primitive_result: &Option<T>,
) {
    if PRINT_PROMPT {
        println!("{}", prompt);
    }
    if PRINT_WORKFLOW_RESULT {
        if let Some(result) = workflow_result {
            println!("{}", result);
        }
    }
    if PRINT_PRIMITIVE_RESULT {
        println!("{:?}", primitive_result);
    }
}

#[cfg(any(feature = "llama_cpp_backend", feature = "mistral_rs_backend"))]
pub async fn default_tiny_llm() -> crate::Result<LlmClient> {
    match DEFAULT_BACKEND {
        #[cfg(feature = "llama_cpp_backend")]
        "llama_cpp" => llama_cpp_tiny_llm().await,
        #[cfg(feature = "mistral_rs_backend")]
        "mistral_rs" => mistral_rs_tiny_llm().await,
        _ => bail!("Invalid default backend"),
    }
}

#[cfg(any(feature = "llama_cpp_backend", feature = "mistral_rs_backend"))]
pub async fn default_medium_llm() -> crate::Result<LlmClient> {
    match DEFAULT_BACKEND {
        #[cfg(feature = "llama_cpp_backend")]
        "llama_cpp" => llama_cpp_medium_llm().await,
        #[cfg(feature = "mistral_rs_backend")]
        "mistral_rs" => mistral_rs_medium_llm().await,
        _ => bail!("Invalid default backend"),
    }
}

#[cfg(any(feature = "llama_cpp_backend", feature = "mistral_rs_backend"))]
pub async fn default_large_llm() -> crate::Result<LlmClient> {
    match DEFAULT_BACKEND {
        #[cfg(feature = "llama_cpp_backend")]
        "llama_cpp" => llama_cpp_large_llm().await,
        #[cfg(feature = "mistral_rs_backend")]
        "mistral_rs" => mistral_rs_large_llm().await,
        _ => bail!("Invalid default backend"),
    }
}

#[cfg(any(feature = "llama_cpp_backend", feature = "mistral_rs_backend"))]
pub async fn default_max_llm() -> crate::Result<LlmClient> {
    match DEFAULT_BACKEND {
        #[cfg(feature = "llama_cpp_backend")]
        "llama_cpp" => llama_cpp_max_llm().await,
        #[cfg(feature = "mistral_rs_backend")]
        "mistral_rs" => mistral_rs_max_llm().await,
        _ => bail!("Invalid default backend"),
    }
}

#[cfg(feature = "llama_cpp_backend")]
pub async fn llama_cpp_tiny_llm() -> crate::Result<LlmClient> {
    let mut builder = LlmClient::llama_cpp();
    builder.llm_loader.gguf_preset_loader.llm_preset = TINY_LLM_PRESET;
    builder.init().await
}

#[cfg(feature = "llama_cpp_backend")]
pub async fn llama_cpp_medium_llm() -> crate::Result<LlmClient> {
    let mut builder = LlmClient::llama_cpp();
    builder.llm_loader.gguf_preset_loader.llm_preset = MEDIUM_LLM_PRESET;
    builder.init().await
}

#[cfg(feature = "llama_cpp_backend")]
pub async fn llama_cpp_large_llm() -> crate::Result<LlmClient> {
    let mut builder = LlmClient::llama_cpp();
    builder.llm_loader.gguf_preset_loader.llm_preset = LARGE_LLM_PRESET;
    builder.init().await
}

#[cfg(feature = "llama_cpp_backend")]
pub async fn llama_cpp_max_llm() -> crate::Result<LlmClient> {
    let mut builder = LlmClient::llama_cpp();
    builder.llm_loader.gguf_preset_loader.llm_preset = MAX_LLM_PRESET;
    builder.init().await
}

#[cfg(feature = "mistral_rs_backend")]
pub async fn mistral_rs_tiny_llm() -> crate::Result<LlmClient> {
    let mut builder = LlmClient::mistral_rs();
    builder.llm_loader.gguf_preset_loader.llm_preset = TINY_LLM_PRESET;
    builder.init().await
}

#[cfg(feature = "mistral_rs_backend")]
pub async fn mistral_rs_medium_llm() -> crate::Result<LlmClient> {
    let mut builder = LlmClient::mistral_rs();
    builder.llm_loader.gguf_preset_loader.llm_preset = MEDIUM_LLM_PRESET;
    builder.init().await
}

#[cfg(feature = "mistral_rs_backend")]
pub async fn mistral_rs_large_llm() -> crate::Result<LlmClient> {
    let mut builder = LlmClient::mistral_rs();
    builder.llm_loader.gguf_preset_loader.llm_preset = LARGE_LLM_PRESET;
    builder.init().await
}

#[cfg(feature = "mistral_rs_backend")]
pub async fn mistral_rs_max_llm() -> crate::Result<LlmClient> {
    let mut builder = LlmClient::mistral_rs();
    builder.llm_loader.gguf_preset_loader.llm_preset = MAX_LLM_PRESET;
    builder.init().await
}
