use super::LlmModelBase;

pub mod anthropic;
pub mod openai;
pub mod perplexity;

#[derive(Clone)]
pub struct ApiLlmModel {
    pub model_base: LlmModelBase,
    pub cost_per_m_in_tokens: f32,
    pub cost_per_m_out_tokens: f32,
    pub tokens_per_message: u32,
    pub tokens_per_name: Option<i32>,
}

impl Default for ApiLlmModel {
    fn default() -> Self {
        Self::gpt_4_o_mini()
    }
}
