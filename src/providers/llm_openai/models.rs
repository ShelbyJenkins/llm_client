#[derive(Clone, Debug)]
pub enum OpenAiLlmModels {
    Gpt4,
    Gpt432k,
    Gpt35Turbo,
    Gpt35Turbo16k,
}

const SAFETY_TOKENS: u16 = 10;

impl OpenAiLlmModels {
    pub fn get_default_model_params(model_definition: &OpenAiLlmModels) -> crate::LlmModelParams {
        let frequency_penalty = OpenAiLlmModels::frequency_penalty(None);
        let presence_penalty = OpenAiLlmModels::presence_penalty(None);
        let temperature = OpenAiLlmModels::temperature(None);
        let top_p = OpenAiLlmModels::top_p(None);

        match model_definition {
            OpenAiLlmModels::Gpt4 => crate::LlmModelParams {
                model_id: "gpt-4".to_string(),
                model_filename: None,
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
            OpenAiLlmModels::Gpt432k => crate::LlmModelParams {
                model_id: "gpt-4-32k".to_string(),
                model_filename: None,
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
            OpenAiLlmModels::Gpt35Turbo => crate::LlmModelParams {
                model_id: "gpt-3.5-turbo".to_string(),
                model_filename: None,
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
            OpenAiLlmModels::Gpt35Turbo16k => crate::LlmModelParams {
                model_id: "gpt-3.5-turbo-16k".to_string(),
                model_filename: None,
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
        model_definition: &OpenAiLlmModels,
        max_tokens_for_model: Option<u16>,
    ) -> u16 {
        let default_params = OpenAiLlmModels::get_default_model_params(&model_definition);
        if let Some(value) = max_tokens_for_model {
            if value > default_params.max_tokens_for_model {
                return default_params.max_tokens_for_model;
            }
            return value;
        } else {
            return default_params.max_tokens_for_model;
        }
    }
}
