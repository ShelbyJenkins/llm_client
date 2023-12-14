#[derive(Clone, Debug)]
pub enum LlamaLlmModels {
    Mistral7BInstruct(String),
    Mistral7BChat(String),
    Mixtral8X7BInstruct(String),
    SOLAR107BInstructv1(String),
}

struct CustomParams {
    max_tokens_for_model: u16,
    url: String,
}
const SAFETY_TOKENS: u16 = 10;

impl LlamaLlmModels {
    pub fn get_default_model_params(model_definition: &LlamaLlmModels) -> crate::LlmModelParams {
        let custom_params = match model_definition {
            LlamaLlmModels::Mistral7BInstruct(url) => CustomParams {
                url: url.to_string(),
                max_tokens_for_model: 32768,
            },
            LlamaLlmModels::Mistral7BChat(url) => CustomParams {
                url: url.to_string(),
                max_tokens_for_model: 32768,
            },
            LlamaLlmModels::Mixtral8X7BInstruct(url) => CustomParams {
                url: url.to_string(),
                max_tokens_for_model: 32768,
            },
            LlamaLlmModels::SOLAR107BInstructv1(url) => CustomParams {
                url: url.to_string(),
                max_tokens_for_model: 32768,
            },
        };

        let (model_id, model_filename) = process_url(&custom_params.url);

        crate::LlmModelParams {
            model_id,
            model_filename: Some(model_filename),
            max_tokens_for_model: custom_params.max_tokens_for_model,
            cost_per_k: 0.00,
            tokens_per_message: 0,
            tokens_per_name: 0,
            frequency_penalty: LlamaLlmModels::frequency_penalty(None),
            presence_penalty: LlamaLlmModels::presence_penalty(None),
            temperature: LlamaLlmModels::temperature(None),
            top_p: LlamaLlmModels::top_p(None),
            safety_tokens: SAFETY_TOKENS,
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
        model_definition: &LlamaLlmModels,
        max_tokens_for_model: Option<u16>,
    ) -> u16 {
        let default_params = LlamaLlmModels::get_default_model_params(model_definition);
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

fn process_url(url: &str) -> (String, String) {
    if !url.starts_with("https://huggingface.co") {
        panic!("URL does not start with https://huggingface.co\n Format should be like: https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF/blob/main/zephyr-7b-alpha.Q8_0.gguf");
    } else if !url.ends_with(".gguf") {
        panic!("URL does not end with .gguf\n Format should be like: https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF/blob/main/zephyr-7b-alpha.Q8_0.gguf");
    } else {
        let parts: Vec<&str> = url.split('/').collect();
        if parts.len() < 5 {
            panic!("URL does not have enough parts\n Format should be like: https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF/blob/main/zephyr-7b-alpha.Q8_0.gguf");
        }
        let model_id = format!("{}/{}", parts[3], parts[4]);
        let model_filename = parts.last().unwrap_or(&"").to_string();
        (model_id, model_filename)
    }
}
