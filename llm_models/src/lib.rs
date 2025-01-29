//! # llm_models: Load and download LLM models, metadata, and tokenizers
//!
//! [![API Documentation](https://docs.rs/llm_models/badge.svg)](https://docs.rs/llm_models)
//!
//! The llm_models crate is a workspace member of the [llm_client](https://github.com/ShelbyJenkins/llm_client) project.
//!
//! ## Features
//!
//! * GGUFs from local storage or Hugging Face
//!     * Parses model metadata from GGUF file
//!     * Includes limited support for tokenizer from GGUF file
//!     * Also supports loading Metadata and Tokenizer from their respective files
//! * API models from OpenAI, Anthropic, and Perplexity
//! * Tokenizer abstraction for Hugging Face's Tokenizer and Tiktoken
//!
//! ## LocalLlmModel
//!
//! Everything you need for GGUF models. The `GgufLoader` wraps the loaders for convenience.
//! All loaders return a `LocalLlmModel` which contains the tokenizer, metadata, chat template,
//! and anything that can be extracted from the GGUF.
//!
//! ### GgufPresetLoader
//!
//! * Presets for popular models like Llama 3, Phi, Mistral/Mixtral, and more
//! * Loads the best quantized model by calculating the largest quant that will fit in your VRAM
//!
//! ```rust
//! use llm_models::*;
//! let model: LocalLlmModel = GgufLoader::default()
//!     .llama3_1_8b_instruct()
//!     .preset_with_available_vram_gb(48) // Load the largest quant that will fit in your vram
//!     .load().unwrap();
//! ```
//!
//! ### GgufHfLoader
//!
//! GGUF models from Hugging Face.
//!
//! ```rust
//! use llm_models::*;
//! let model: LocalLlmModel = GgufLoader::default()
//!     .hf_quant_file_url("https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/blob/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf")
//!     .load().unwrap();
//! ```
//!
//! ### GgufLocalLoader
//!
//! GGUF models from local storage.
//!
//! ```rust
//! use llm_models::*;
//! let model: LocalLlmModel = GgufLoader::default()
//!     .local_quant_file_path("/root/.cache/huggingface/hub/models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF/blobs/9da71c45c90a821809821244d4971e5e5dfad7eb091f0b8ff0546392393b6283")
//!     .load().unwrap();
//! ```
//!
//! ## ApiLlmModel
//!
//! * Supports OpenAI, Anthropic, Perplexity, and adding your own API models
//! * Supports prompting, tokenization, and price estimation
//!
//! ```rust
//! use llm_models::*;
//! let model = ApiLlmModel::gpt_4_o();
//! assert_eq!(model.model_base.model_id, "gpt-4o");
//! assert_eq!(model.model_base.model_ctx_size, 128000);
//! assert_eq!(model.model_base.inference_ctx_size, 4096);
//! assert_eq!(model.cost_per_m_in_tokens, 5.00);
//! assert_eq!(model.cost_per_m_out_tokens, 15.00);
//! assert_eq!(model.tokens_per_message, 3);
//! assert_eq!(model.tokens_per_name, Some(1));
//! ```
//!
//! ## LlmTokenizer
//!
//! * Simple abstract API for encoding and decoding allows for abstract LLM consumption across multiple architectures
//! * Uses Hugging Face's Tokenizer library for local models and Tiktoken-rs for OpenAI and Anthropic
//!   ([Anthropic doesn't have a publicly available tokenizer](https://github.com/javirandor/anthropic-tokenizer))
//!
//! ```rust
//! use llm_models::*;
//! // Get a Tiktoken tokenizer
//! let tok = LlmTokenizer::new_tiktoken("gpt-4o");
//!
//! // From local path
//! let tok = LlmTokenizer::new_from_tokenizer_json("path/to/tokenizer.json");
//!
//! // From repo (requires Hugging Face token)
//! // let tok = LlmTokenizer::new_from_hf_repo(hf_token, "meta-llama/Meta-Llama-3-8B-Instruct");
//! ```
//!
//! ## Setter Traits
//!
//! * All setter traits are public, so you can integrate into your own projects if you wish
//! * Examples include: `OpenAiModelTrait`, `GgufLoaderTrait`, `AnthropicModelTrait`, and `HfTokenTrait` for loading models

#![feature(f16)]

// Public modules
pub mod api_model;
pub mod local_model;
pub mod tokenizer;

// Internal imports
#[allow(unused_imports)]
use anyhow::{anyhow, bail, Error, Result};
#[allow(unused_imports)]
use tracing::{debug, error, info, span, trace, warn, Level};

// Public exports
pub use api_model::{
    anthropic::AnthropicModelTrait, openai::OpenAiModelTrait, perplexity::PerplexityModelTrait,
    ApiLlmModel,
};
pub use local_model::{
    chat_template::LlmChatTemplate,
    gguf::{
        loaders::preset::GgufPresetLoader, preset::GgufPresetTrait, GgufLoader, GgufLoaderTrait,
    },
    hf_loader::HfTokenTrait,
    metadata::LocalLlmMetadata,
    LocalLlmModel,
};
pub use tokenizer::LlmTokenizer;

#[derive(Clone)]
pub struct LlmModelBase {
    pub model_id: String,
    pub model_ctx_size: u64,
    pub inference_ctx_size: u64,
    pub tokenizer: std::sync::Arc<LlmTokenizer>,
}
