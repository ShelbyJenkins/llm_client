#![allow(dead_code)]

pub use anyhow::Result;
pub use llm_client::prelude::*;
pub use serial_test::serial;
pub mod test_loader;
pub use test_loader::TestSetsLoader;
pub mod test_types;
pub use serde::{Deserialize, Serialize};
pub use test_types::*;
pub use url::Url;

pub const PRINT_PRIMITIVE_RESULT: bool = false;
pub const PRINT_WORKFLOW_RESULT: bool = true;
pub const PRINT_PROMPT: bool = false;

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

/// Sets the levels of tests to be run
pub fn primitive_tests(optional: bool) -> TestSetsLoader {
    TestSetsLoader::default()
        .optional(optional)
        .test_level_zero()
}

pub async fn get_tiny_llm() -> Result<LlmClient> {
    LlmClient::llama_cpp()
        .phi3_5_mini_instruct()
        .available_vram(44)
        .init()
        .await
}

pub async fn get_medium_llm() -> Result<LlmClient> {
    LlmClient::llama_cpp()
        .llama3_1_8b_instruct()
        .available_vram(44)
        .init()
        .await
}

pub async fn get_large_llm() -> Result<LlmClient> {
    LlmClient::llama_cpp()
        .mistral_nemo_instruct2407()
        .available_vram(44)
        .init()
        .await
}

pub async fn get_max_llm() -> Result<LlmClient> {
    LlmClient::llama_cpp()
        .mixtral8x7b_instruct_v0_1()
        .available_vram(44)
        .init()
        .await
}
