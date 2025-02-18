use super::*;
#[derive(Debug, Clone)]
pub struct ApiLlmPreset {
    pub provider: ApiLlmProvider,
    pub model_id: &'static str,
    pub friendly_name: &'static str,
    pub model_ctx_size: usize,
    pub inference_ctx_size: usize,
    pub cost_per_m_in_tokens: usize,
    pub cost_per_m_out_tokens: usize,
    pub tokens_per_message: usize,
    pub tokens_per_name: Option<isize>,
}
impl ApiLlmModel {
    pub fn model_from_model_id(
        &self,
        model_id: &str,
    ) -> Result<ApiLlmModel, crate::Error> {
        let providers = ApiLlmProvider::all_providers();
        for provider in providers {
            if let Ok(preset) = provider.preset_from_model_id(model_id) {
                return Ok(Self::from_preset(preset));
            }
        }
        crate::bail!("Model not found for model_id: {}", model_id);
    }
}
impl ApiLlmPreset {
    pub fn all_models() -> Vec<Self> {
        vec![
            Self::CLAUDE_3_OPUS, Self::CLAUDE_3_SONNET, Self::CLAUDE_3_HAIKU,
            Self::CLAUDE_3_5_SONNET, Self::CLAUDE_3_5_HAIKU, Self::GPT_4,
            Self::GPT_3_5_TURBO, Self::GPT_4_32K, Self::GPT_4_TURBO, Self::GPT_4O,
            Self::GPT_4O_MINI, Self::O1, Self::O1_MINI, Self::O3_MINI,
            Self::SONAR_REASONING_PRO, Self::SONAR_REASONING, Self::SONAR_PRO,
            Self::SONAR
        ]
    }
    pub const CLAUDE_3_OPUS: ApiLlmPreset = ApiLlmPreset {
        provider: ApiLlmProvider::Anthropic,
        model_id: "claude-3-opus-latest",
        friendly_name: "Claude 3 Opus",
        model_ctx_size: 200000usize,
        inference_ctx_size: 4096usize,
        cost_per_m_in_tokens: 1500usize,
        cost_per_m_out_tokens: 7500usize,
        tokens_per_message: 3usize,
        tokens_per_name: None,
    };
    pub const CLAUDE_3_SONNET: ApiLlmPreset = ApiLlmPreset {
        provider: ApiLlmProvider::Anthropic,
        model_id: "claude-3-sonnet-20240229",
        friendly_name: "Claude 3 Sonnet",
        model_ctx_size: 200000usize,
        inference_ctx_size: 4096usize,
        cost_per_m_in_tokens: 300usize,
        cost_per_m_out_tokens: 1500usize,
        tokens_per_message: 3usize,
        tokens_per_name: None,
    };
    pub const CLAUDE_3_HAIKU: ApiLlmPreset = ApiLlmPreset {
        provider: ApiLlmProvider::Anthropic,
        model_id: "claude-3-haiku-20240307",
        friendly_name: "Claude 3 Haiku",
        model_ctx_size: 200000usize,
        inference_ctx_size: 4096usize,
        cost_per_m_in_tokens: 75usize,
        cost_per_m_out_tokens: 125usize,
        tokens_per_message: 3usize,
        tokens_per_name: None,
    };
    pub const CLAUDE_3_5_SONNET: ApiLlmPreset = ApiLlmPreset {
        provider: ApiLlmProvider::Anthropic,
        model_id: "claude-3-5-sonnet-latest",
        friendly_name: "Claude 3.5 Sonnet",
        model_ctx_size: 200000usize,
        inference_ctx_size: 8192usize,
        cost_per_m_in_tokens: 300usize,
        cost_per_m_out_tokens: 1500usize,
        tokens_per_message: 3usize,
        tokens_per_name: None,
    };
    pub const CLAUDE_3_5_HAIKU: ApiLlmPreset = ApiLlmPreset {
        provider: ApiLlmProvider::Anthropic,
        model_id: "claude-3-5-haiku-latest",
        friendly_name: "Claude 3.5 Haiku",
        model_ctx_size: 200000usize,
        inference_ctx_size: 8192usize,
        cost_per_m_in_tokens: 80usize,
        cost_per_m_out_tokens: 400usize,
        tokens_per_message: 3usize,
        tokens_per_name: None,
    };
    pub const GPT_4: ApiLlmPreset = ApiLlmPreset {
        provider: ApiLlmProvider::OpenAi,
        model_id: "gpt-4",
        friendly_name: "GPT-4",
        model_ctx_size: 8192usize,
        inference_ctx_size: 4096usize,
        cost_per_m_in_tokens: 3000usize,
        cost_per_m_out_tokens: 6000usize,
        tokens_per_message: 3usize,
        tokens_per_name: Some(1isize),
    };
    pub const GPT_3_5_TURBO: ApiLlmPreset = ApiLlmPreset {
        provider: ApiLlmProvider::OpenAi,
        model_id: "gpt-3.5-turbo",
        friendly_name: "GPT-3.5 Turbo",
        model_ctx_size: 16385usize,
        inference_ctx_size: 4096usize,
        cost_per_m_in_tokens: 50usize,
        cost_per_m_out_tokens: 150usize,
        tokens_per_message: 4usize,
        tokens_per_name: Some(-1isize),
    };
    pub const GPT_4_32K: ApiLlmPreset = ApiLlmPreset {
        provider: ApiLlmProvider::OpenAi,
        model_id: "gpt-4-32k",
        friendly_name: "GPT-4 32K",
        model_ctx_size: 32768usize,
        inference_ctx_size: 4096usize,
        cost_per_m_in_tokens: 6000usize,
        cost_per_m_out_tokens: 12000usize,
        tokens_per_message: 3usize,
        tokens_per_name: Some(1isize),
    };
    pub const GPT_4_TURBO: ApiLlmPreset = ApiLlmPreset {
        provider: ApiLlmProvider::OpenAi,
        model_id: "gpt-4-turbo",
        friendly_name: "GPT-4 Turbo",
        model_ctx_size: 128000usize,
        inference_ctx_size: 4096usize,
        cost_per_m_in_tokens: 1000usize,
        cost_per_m_out_tokens: 3000usize,
        tokens_per_message: 3usize,
        tokens_per_name: Some(1isize),
    };
    pub const GPT_4O: ApiLlmPreset = ApiLlmPreset {
        provider: ApiLlmProvider::OpenAi,
        model_id: "gpt-4o",
        friendly_name: "GPT-4o",
        model_ctx_size: 128000usize,
        inference_ctx_size: 4096usize,
        cost_per_m_in_tokens: 250usize,
        cost_per_m_out_tokens: 1000usize,
        tokens_per_message: 3usize,
        tokens_per_name: Some(1isize),
    };
    pub const GPT_4O_MINI: ApiLlmPreset = ApiLlmPreset {
        provider: ApiLlmProvider::OpenAi,
        model_id: "gpt-4o-mini",
        friendly_name: "GPT-4o Mini",
        model_ctx_size: 128000usize,
        inference_ctx_size: 16384usize,
        cost_per_m_in_tokens: 15usize,
        cost_per_m_out_tokens: 60usize,
        tokens_per_message: 3usize,
        tokens_per_name: Some(1isize),
    };
    pub const O1: ApiLlmPreset = ApiLlmPreset {
        provider: ApiLlmProvider::OpenAi,
        model_id: "o1",
        friendly_name: "O1",
        model_ctx_size: 200000usize,
        inference_ctx_size: 100000usize,
        cost_per_m_in_tokens: 1500usize,
        cost_per_m_out_tokens: 6000usize,
        tokens_per_message: 4usize,
        tokens_per_name: Some(-1isize),
    };
    pub const O1_MINI: ApiLlmPreset = ApiLlmPreset {
        provider: ApiLlmProvider::OpenAi,
        model_id: "o1-mini",
        friendly_name: "o1 Mini",
        model_ctx_size: 128000usize,
        inference_ctx_size: 65536usize,
        cost_per_m_in_tokens: 110usize,
        cost_per_m_out_tokens: 440usize,
        tokens_per_message: 4usize,
        tokens_per_name: Some(-1isize),
    };
    pub const O3_MINI: ApiLlmPreset = ApiLlmPreset {
        provider: ApiLlmProvider::OpenAi,
        model_id: "o3-mini",
        friendly_name: "o3 Mini",
        model_ctx_size: 200000usize,
        inference_ctx_size: 100000usize,
        cost_per_m_in_tokens: 110usize,
        cost_per_m_out_tokens: 440usize,
        tokens_per_message: 4usize,
        tokens_per_name: Some(-1isize),
    };
    pub const SONAR_REASONING_PRO: ApiLlmPreset = ApiLlmPreset {
        provider: ApiLlmProvider::Perplexity,
        model_id: "sonar-reasoning-pro",
        friendly_name: "Sonar Reasoning Pro",
        model_ctx_size: 127000usize,
        inference_ctx_size: 8000usize,
        cost_per_m_in_tokens: 200usize,
        cost_per_m_out_tokens: 8000usize,
        tokens_per_message: 3usize,
        tokens_per_name: None,
    };
    pub const SONAR_REASONING: ApiLlmPreset = ApiLlmPreset {
        provider: ApiLlmProvider::Perplexity,
        model_id: "sonar-reasoning",
        friendly_name: "Sonar Reasoning",
        model_ctx_size: 127000usize,
        inference_ctx_size: 8000usize,
        cost_per_m_in_tokens: 100usize,
        cost_per_m_out_tokens: 500usize,
        tokens_per_message: 3usize,
        tokens_per_name: None,
    };
    pub const SONAR_PRO: ApiLlmPreset = ApiLlmPreset {
        provider: ApiLlmProvider::Perplexity,
        model_id: "sonar-pro",
        friendly_name: "Sonar Pro",
        model_ctx_size: 200000usize,
        inference_ctx_size: 8000usize,
        cost_per_m_in_tokens: 300usize,
        cost_per_m_out_tokens: 1500usize,
        tokens_per_message: 3usize,
        tokens_per_name: None,
    };
    pub const SONAR: ApiLlmPreset = ApiLlmPreset {
        provider: ApiLlmProvider::Perplexity,
        model_id: "sonar",
        friendly_name: "Sonar",
        model_ctx_size: 127000usize,
        inference_ctx_size: 8000usize,
        cost_per_m_in_tokens: 100usize,
        cost_per_m_out_tokens: 100usize,
        tokens_per_message: 3usize,
        tokens_per_name: None,
    };
}
