use dioxus::fullstack::prelude::*;

pub struct LlmGuiApiModel {
    pub model_base: LlmGuiModelBase,
    // pub provider: ApiLlmProvider,
    pub cost_per_m_in_tokens: f32,
    pub cost_per_m_out_tokens: f32,
}
