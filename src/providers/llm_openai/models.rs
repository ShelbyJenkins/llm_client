#[derive(Clone, Debug)]
pub enum OpenAiDef {
    Gpt4,
    Gpt432k,
    Gpt35Turbo,
    Gpt35Turbo16k,
    EmbeddingAda002,
}

const SAFETY_TOKENS: u16 = 10;

impl OpenAiDef {
    pub fn get_default_model_params(model_definition: &OpenAiDef) -> crate::LlmModelParams {
        let frequency_penalty = OpenAiDef::frequency_penalty(None);
        let presence_penalty = OpenAiDef::presence_penalty(None);
        let temperature = OpenAiDef::temperature(None);
        let top_p = OpenAiDef::top_p(None);

        match model_definition {
            OpenAiDef::Gpt4 => crate::LlmModelParams {
                model_id: "gpt-4".to_string(),
                max_tokens_for_model: 8192,
                cost_per_k: 0.06,
                tokens_per_message: 3,
                tokens_per_name: 1,
                frequency_penalty,
                presence_penalty,
                temperature,
                top_p,
                safety_tokens: SAFETY_TOKENS,
            },
            OpenAiDef::Gpt432k => crate::LlmModelParams {
                model_id: "gpt-4-32k".to_string(),
                max_tokens_for_model: 32768,
                cost_per_k: 0.06,
                tokens_per_message: 3,
                tokens_per_name: 1,
                frequency_penalty,
                presence_penalty,
                temperature,
                top_p,
                safety_tokens: SAFETY_TOKENS,
            },
            OpenAiDef::Gpt35Turbo => crate::LlmModelParams {
                model_id: "gpt-3.5-turbo".to_string(),
                max_tokens_for_model: 4096,
                cost_per_k: 0.03,
                tokens_per_message: 4,
                tokens_per_name: -1,
                frequency_penalty,
                presence_penalty,
                temperature,
                top_p,
                safety_tokens: SAFETY_TOKENS,
            },
            OpenAiDef::Gpt35Turbo16k => crate::LlmModelParams {
                model_id: "gpt-3.5-turbo-16k".to_string(),
                max_tokens_for_model: 16384,
                cost_per_k: 0.03,
                tokens_per_message: 3,
                tokens_per_name: 1,
                frequency_penalty,
                presence_penalty,
                temperature,
                top_p,
                safety_tokens: SAFETY_TOKENS,
            },
            OpenAiDef::EmbeddingAda002 => crate::LlmModelParams {
                model_id: "text-embedding-ada-002".to_string(),
                max_tokens_for_model: 8191,
                cost_per_k: 0.0004,
                tokens_per_message: 0,
                tokens_per_name: 0,
                frequency_penalty,
                presence_penalty,
                temperature,
                top_p,
                safety_tokens: SAFETY_TOKENS,
            },
        }
    }

    pub fn frequency_penalty(frequency_penalty: Option<f32>) -> f32 {
        match frequency_penalty {
            Some(value) if (-2.0..=2.0).contains(&value) => value,
            _ => 0.0,
        }
    }
    pub fn presence_penalty(presence_penalty: Option<f32>) -> f32 {
        match presence_penalty {
            Some(value) if (-2.0..=2.0).contains(&value) => value,
            _ => 0.0,
        }
    }
    pub fn temperature(temperature: Option<f32>) -> f32 {
        match temperature {
            Some(value) if (0.0..=2.0).contains(&value) => value,
            _ => 1.0,
        }
    }
    pub fn top_p(top_p: Option<f32>) -> f32 {
        match top_p {
            Some(value) if (0.0..=2.0).contains(&value) => value,
            _ => 1.0,
        }
    }
    pub fn max_tokens_for_model(
        model_definition: &OpenAiDef,
        max_tokens_for_model: Option<u16>,
    ) -> u16 {
        let default_params = OpenAiDef::get_default_model_params(model_definition);
        if let Some(value) = max_tokens_for_model {
            if value > default_params.max_tokens_for_model {
                return default_params.max_tokens_for_model;
            }
            value
        } else {
            default_params.max_tokens_for_model
        }
    }
}
