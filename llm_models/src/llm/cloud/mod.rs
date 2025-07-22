// Public generated modules
pub mod models;
pub mod providers;

use super::{LlmModelBase, LlmModelId};
pub use providers::CloudProviderLlmId;

use providers::{AnthropicLlmId::*, MistralAiLlmId::*, OpenAiLlmId::*, PerplexityLlmId::*};
use std::borrow::Cow;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct CloudLlm {
    pub model_base: LlmModelBase,
    pub provider_llm_id: CloudProviderLlmId,
    pub cost_per_m_in_tokens: u64,
    pub cost_per_m_out_tokens: u64,
    pub tokens_per_message: u64,
    pub tokens_per_name: Option<i64>,
}

impl CloudLlm {
    pub fn friendly_name(&self) -> &str {
        &self.model_base.friendly_name
    }
    pub fn model_id(&self) -> &str {
        &self.model_base.model_id
    }
    pub fn model_ctx_size(&self) -> u64 {
        self.model_base.model_ctx_size
    }
    pub fn inference_ctx_size(&self) -> u64 {
        self.model_base.inference_ctx_size
    }
    pub fn matches_id(&self, id: &LlmModelId) -> bool {
        match id {
            LlmModelId::Cloud(id) => self.provider_llm_id == *id,
            _ => false,
        }
    }
}
