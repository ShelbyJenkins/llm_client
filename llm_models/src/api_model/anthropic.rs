use super::ApiLlmModel;
use crate::{tokenizer::LlmTokenizer, LlmModelBase};
use std::sync::Arc;

impl ApiLlmModel {
    pub fn anthropic_model_from_model_id(model_id: &str) -> ApiLlmModel {
        if model_id.starts_with("claude-3-opus") {
            Self::claude_3_opus()
        } else if model_id.starts_with("claude-3-sonnet") {
            Self::claude_3_sonnet()
        } else if model_id.starts_with("claude-3-haiku") {
            Self::claude_3_haiku()
        } else if model_id.starts_with("claude-3-5-sonnet") {
            Self::claude_3_5_sonnet()
        } else {
            panic!("Model ID ({model_id}) not found for ApiLlmModel")
        }
    }

    pub fn claude_3_opus() -> ApiLlmModel {
        let model_id = "claude-3-opus-20240229".to_string();
        let tokenizer = model_tokenizer(&model_id);
        ApiLlmModel {
            model_base: LlmModelBase {
                model_id,
                model_ctx_size: 200000,
                inference_ctx_size: 4096,
                tokenizer,
            },
            cost_per_m_in_tokens: 15.00,
            cost_per_m_out_tokens: 75.00,
            tokens_per_message: 3,
            tokens_per_name: None,
        }
    }

    pub fn claude_3_sonnet() -> ApiLlmModel {
        let model_id = "claude-3-sonnet-20240229".to_string();
        let tokenizer = model_tokenizer(&model_id);
        ApiLlmModel {
            model_base: LlmModelBase {
                model_id,
                model_ctx_size: 200000,
                inference_ctx_size: 4096,
                tokenizer,
            },
            cost_per_m_in_tokens: 3.00,
            cost_per_m_out_tokens: 15.00,
            tokens_per_message: 3,
            tokens_per_name: None,
        }
    }

    pub fn claude_3_haiku() -> ApiLlmModel {
        let model_id = "claude-3-haiku-20240307".to_string();
        let tokenizer = model_tokenizer(&model_id);
        ApiLlmModel {
            model_base: LlmModelBase {
                model_id,
                model_ctx_size: 200000,
                inference_ctx_size: 4096,
                tokenizer,
            },
            cost_per_m_in_tokens: 0.75,
            cost_per_m_out_tokens: 1.25,
            tokens_per_message: 3,
            tokens_per_name: None,
        }
    }

    pub fn claude_3_5_sonnet() -> ApiLlmModel {
        let model_id = "claude-3-5-sonnet-20240620".to_string();
        let tokenizer = model_tokenizer(&model_id);
        ApiLlmModel {
            model_base: LlmModelBase {
                model_id,
                model_ctx_size: 200000,
                inference_ctx_size: 8192,
                tokenizer,
            },
            cost_per_m_in_tokens: 3.00,
            cost_per_m_out_tokens: 15.00,
            tokens_per_message: 3,
            tokens_per_name: None,
        }
    }
}

pub fn model_tokenizer(_model_id: &str) -> Arc<LlmTokenizer> {
    println!("Anthropic does not have a publically available tokenizer. See this for more information: https://github.com/javirandor/anthropic-tokenizer");
    println!("However, since Anthropic does not support logit bias, we don't have a use for an actual tokenizer. So we can use TikToken to count tokens.");
    Arc::new(
        LlmTokenizer::new_tiktoken("gpt-4")
            .unwrap_or_else(|_| panic!("Failed to load tokenizer for gpt-4")),
    )
}

pub trait AnthropicModelTrait: Sized {
    fn model(&mut self) -> &mut ApiLlmModel;

    /// Set the model using the model_id string.
    fn model_id_str(mut self, model_id: &str) -> Self
    where
        Self: Sized,
    {
        *self.model() = ApiLlmModel::anthropic_model_from_model_id(model_id);
        self
    }

    /// Use the Claude 3 Opus model for the Anthropic client.
    fn claude_3_opus(mut self) -> Self
    where
        Self: Sized,
    {
        *self.model() = ApiLlmModel::claude_3_opus();
        self
    }

    /// Use the Claude 3 Sonnet model for the Anthropic client.
    fn claude_3_sonnet(mut self) -> Self
    where
        Self: Sized,
    {
        *self.model() = ApiLlmModel::claude_3_sonnet();
        self
    }

    /// Use the Claude 3 Haiku model for the Anthropic client.
    fn claude_3_haiku(mut self) -> Self
    where
        Self: Sized,
    {
        *self.model() = ApiLlmModel::claude_3_haiku();
        self
    }

    /// Use the Claude 3.5 Sonnet model for the Anthropic client.
    fn claude_3_5_sonnet(mut self) -> Self
    where
        Self: Sized,
    {
        *self.model() = ApiLlmModel::claude_3_5_sonnet();
        self
    }
}
