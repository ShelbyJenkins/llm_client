use super::ApiLlmModel;
use crate::{tokenizer::LlmTokenizer, LlmModelBase};
use std::sync::Arc;

impl ApiLlmModel {
    pub fn openai_model_from_model_id(model_id: &str) -> Self {
        match model_id {
            model_id if model_id.starts_with("gpt-4") => Self::gpt_4(),
            model_id if model_id.starts_with("gpt-3.5-turbo") => Self::gpt_3_5_turbo(),
            model_id if model_id.starts_with("gpt-4-32k") => Self::gpt_4_32k(),
            model_id if model_id.starts_with("gpt-4-turbo") => Self::gpt_4_turbo(),
            model_id if model_id.starts_with("gpt-4o") => Self::gpt_4_o(),
            model_id if model_id.starts_with("gpt-4o-mini") => Self::gpt_4_o_mini(),
            model_id if model_id.starts_with("o1") => Self::o1(),
            model_id if model_id.starts_with("o1-mini") => Self::o1_mini(),
            _ => panic!("Model ID ({model_id}) not found for ApiLlmModel"),
        }
    }

    pub fn gpt_4() -> Self {
        let model_id = "gpt-4".to_string();
        let tokenizer = model_tokenizer(&model_id);
        Self {
            model_base: LlmModelBase {
                model_id,
                model_ctx_size: 8192,
                inference_ctx_size: 4096,
                tokenizer,
            },
            cost_per_m_in_tokens: 30.00,
            cost_per_m_out_tokens: 60.00,
            tokens_per_message: 3,
            tokens_per_name: Some(1),
        }
    }

    pub fn gpt_3_5_turbo() -> Self {
        let model_id = "gpt-3.5-turbo".to_string();
        let tokenizer = model_tokenizer(&model_id);
        Self {
            model_base: LlmModelBase {
                model_id,
                model_ctx_size: 16385,
                inference_ctx_size: 4096,
                tokenizer,
            },
            cost_per_m_in_tokens: 0.50,
            cost_per_m_out_tokens: 1.50,
            tokens_per_message: 4,
            tokens_per_name: Some(-1),
        }
    }

    pub fn gpt_4_32k() -> Self {
        let model_id = "gpt-4-32k".to_string();
        let tokenizer = model_tokenizer(&model_id);
        Self {
            model_base: LlmModelBase {
                model_id,
                model_ctx_size: 32768,
                inference_ctx_size: 4096,
                tokenizer,
            },
            cost_per_m_in_tokens: 60.00,
            cost_per_m_out_tokens: 120.00,
            tokens_per_message: 3,
            tokens_per_name: Some(1),
        }
    }

    pub fn gpt_4_turbo() -> Self {
        let model_id = "gpt-4-turbo".to_string();
        let tokenizer = model_tokenizer(&model_id);
        Self {
            model_base: LlmModelBase {
                model_id,
                model_ctx_size: 128000,
                inference_ctx_size: 4096,
                tokenizer,
            },
            cost_per_m_in_tokens: 10.00,
            cost_per_m_out_tokens: 30.00,
            tokens_per_message: 3,
            tokens_per_name: Some(1),
        }
    }

    pub fn gpt_4_o_mini() -> Self {
        let model_id = "gpt-4o-mini".to_string();
        let tokenizer = model_tokenizer(&model_id);
        Self {
            model_base: LlmModelBase {
                model_id,
                model_ctx_size: 128000,
                inference_ctx_size: 16384,
                tokenizer,
            },
            cost_per_m_in_tokens: 0.150,
            cost_per_m_out_tokens: 0.60,
            tokens_per_message: 3,
            tokens_per_name: Some(1),
        }
    }

    pub fn gpt_4_o() -> Self {
        let model_id = "gpt-4o".to_string();
        let tokenizer = model_tokenizer(&model_id);
        Self {
            model_base: LlmModelBase {
                model_id,
                model_ctx_size: 128000,
                inference_ctx_size: 4096,
                tokenizer,
            },
            cost_per_m_in_tokens: 5.00,
            cost_per_m_out_tokens: 15.00,
            tokens_per_message: 3,
            tokens_per_name: Some(1),
        }
    }

    pub fn o1() -> Self {
        let model_id = "o1".to_string();
        let tokenizer = model_tokenizer(&model_id);
        Self {
            model_base: LlmModelBase {
                model_id,
                model_ctx_size: 200000,
                inference_ctx_size: 100000,
                tokenizer,
            },
            cost_per_m_in_tokens: 15.00,
            cost_per_m_out_tokens: 60.00,
            tokens_per_message: 4,
            tokens_per_name: Some(-1),
        }
    }

    pub fn o1_mini() -> Self {
        let model_id = "o1-mini".to_string();
        let tokenizer = model_tokenizer(&model_id);
        Self {
            model_base: LlmModelBase {
                model_id,
                model_ctx_size: 128000,
                inference_ctx_size: 65536,
                tokenizer,
            },
            cost_per_m_in_tokens: 3.00,
            cost_per_m_out_tokens: 12.00,
            tokens_per_message: 4,
            tokens_per_name: Some(-1),
        }
    }
}

fn model_tokenizer(model_id: &str) -> Arc<LlmTokenizer> {
    Arc::new(
        LlmTokenizer::new_tiktoken(model_id)
            .unwrap_or_else(|_| panic!("Failed to load tokenizer for {model_id}")),
    )
}

pub trait OpenAiModelTrait {
    fn model(&mut self) -> &mut ApiLlmModel;

    /// Set the model using the model_id string.
    fn model_id_str(mut self, model_id: &str) -> Self
    where
        Self: Sized,
    {
        *self.model() = ApiLlmModel::openai_model_from_model_id(model_id);
        self
    }

    /// Use gpt-4 as the model for the OpenAI client.
    fn gpt_4(mut self) -> Self
    where
        Self: Sized,
    {
        *self.model() = ApiLlmModel::gpt_4();
        self
    }

    /// Use gpt-4-32k as the model for the OpenAI client. Limited support for this model from OpenAI.
    fn gpt_4_32k(mut self) -> Self
    where
        Self: Sized,
    {
        *self.model() = ApiLlmModel::gpt_4_32k();
        self
    }

    /// Use gpt-3.5-turbo as the model for the OpenAI client.
    fn gpt_3_5_turbo(mut self) -> Self
    where
        Self: Sized,
    {
        *self.model() = ApiLlmModel::gpt_3_5_turbo();
        self
    }

    /// Use gpt-4-turbo as the model for the OpenAI client.
    fn gpt_4_turbo(mut self) -> Self
    where
        Self: Sized,
    {
        *self.model() = ApiLlmModel::gpt_4_turbo();
        self
    }

    /// Use gpt-4-o as the model for the OpenAI client.
    fn gpt_4_o(mut self) -> Self
    where
        Self: Sized,
    {
        *self.model() = ApiLlmModel::gpt_4_o();
        self
    }

    /// Use gpt-4o-mini as the model for the OpenAI client.
    fn gpt_4_o_mini(mut self) -> Self
    where
        Self: Sized,
    {
        *self.model() = ApiLlmModel::gpt_4_o_mini();
        self
    }

    /// Use o1 as the model for the OpenAI client.
    fn o1(mut self) -> Self
    where
        Self: Sized,
    {
        *self.model() = ApiLlmModel::o1();
        self
    }

    /// Use o1-mini as the model for the OpenAI client.
    fn o1_mini(mut self) -> Self
    where
        Self: Sized,
    {
        *self.model() = ApiLlmModel::o1_mini();
        self
    }
}
