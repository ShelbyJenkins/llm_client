// Public generated modules
pub mod models;
pub mod providers;

// Public exports
pub use self::{
    models::ApiLlmPreset,
    providers::{AnthropicModelTrait, ApiLlmProvider, OpenAiModelTrait, PerplexityModelTrait},
};

// Internal imports
use super::LlmModelBase;

// Feature-specific internal imports
#[cfg(feature = "model-tokenizers")]
use crate::tokenizer::LlmTokenizer;

#[derive(Clone)]
pub struct ApiLlmModel {
    pub model_base: LlmModelBase,
    pub provider: ApiLlmProvider,
    pub cost_per_m_in_tokens: usize,
    pub cost_per_m_out_tokens: usize, // to usize
    pub tokens_per_message: usize,
    pub tokens_per_name: Option<isize>,
}

impl Default for ApiLlmModel {
    fn default() -> Self {
        ApiLlmModel::from_preset(ApiLlmPreset::CLAUDE_3_5_SONNET)
    }
}

impl ApiLlmModel {
    pub fn from_preset(preset: ApiLlmPreset) -> Self {
        Self {
            model_base: LlmModelBase {
                #[cfg(feature = "model-tokenizers")]
                tokenizer: preset.provider.model_tokenizer(preset.model_id),
                model_id: preset.model_id.to_string(),
                friendly_name: preset.friendly_name.to_string(),
                model_ctx_size: preset.model_ctx_size,
                inference_ctx_size: preset.inference_ctx_size,
            },
            provider: preset.provider,
            cost_per_m_in_tokens: preset.cost_per_m_in_tokens,
            cost_per_m_out_tokens: preset.cost_per_m_out_tokens,
            tokens_per_message: preset.tokens_per_message,
            tokens_per_name: preset.tokens_per_name,
        }
    }
}

impl ApiLlmProvider {
    #[cfg(feature = "model-tokenizers")]
    fn model_tokenizer(&self, model_id: &str) -> std::sync::Arc<LlmTokenizer> {
        match self {
            ApiLlmProvider::Anthropic => {
                println!("Anthropic does not have a publically available tokenizer. However, since Anthropic does not support logit bias, we don't have a use for an actual tokenizer. So we can use TikToken to count tokens. See this for more information: https://github.com/javirandor/anthropic-tokenizer");
                std::sync::Arc::new(
                    LlmTokenizer::new_tiktoken("gpt-4")
                        .unwrap_or_else(|_| panic!("Failed to load tokenizer for gpt-4")),
                )
            }
            ApiLlmProvider::OpenAi => std::sync::Arc::new(
                LlmTokenizer::new_tiktoken(model_id)
                    .unwrap_or_else(|_| panic!("Failed to load tokenizer for {model_id}")),
            ),
            ApiLlmProvider::Perplexity => std::sync::Arc::new(
                LlmTokenizer::new_tiktoken("gpt-4")
                    .unwrap_or_else(|_| panic!("Failed to load tokenizer for gpt-4")),
            ),
        }
    }
}
