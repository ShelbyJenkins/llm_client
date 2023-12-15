#[derive(Clone, Debug)]

pub enum LlamaPromptFormat {
    Mistral7BInstruct,
    Mistral7BChat,
    Mixtral8X7BInstruct,
    SOLAR107BInstructv1,
}
#[derive(Debug, Clone)]
pub struct LlamaLlmModel {
    pub model_id: String,
    pub model_filename: String,
    pub prompt_format: LlamaPromptFormat,
    pub max_tokens_for_model: u16,
}

const SAFETY_TOKENS: u16 = 10;
const MAX_TOKENS_FOR_MODEL: u16 = 9001;

impl LlamaLlmModel {
    pub fn new(
        model_url: &str,
        prompt_format: LlamaPromptFormat,
        max_tokens_for_model: Option<u16>,
    ) -> Self {
        let max_tokens_for_model = max_tokens_for_model.unwrap_or(MAX_TOKENS_FOR_MODEL);

        let (model_id, model_filename) = convert_url_to_hf_format(&model_url);

        LlamaLlmModel {
            model_id,
            model_filename,
            max_tokens_for_model,
            prompt_format,
        }
    }
    pub fn get_default_model_params(model_definition: &LlamaLlmModel) -> crate::LlmModelParams {
        crate::LlmModelParams {
            model_id: model_definition.model_id.clone(),
            max_tokens_for_model: model_definition.max_tokens_for_model,
            cost_per_k: 0.00,
            tokens_per_message: 0,
            tokens_per_name: 0,
            frequency_penalty: LlamaLlmModel::frequency_penalty(None),
            presence_penalty: LlamaLlmModel::presence_penalty(None),
            temperature: LlamaLlmModel::temperature(None),
            top_p: LlamaLlmModel::top_p(None),
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
        model_definition: &LlamaLlmModel,
        max_tokens_for_model: Option<u16>,
    ) -> u16 {
        max_tokens_for_model.unwrap_or(MAX_TOKENS_FOR_MODEL)
    }
}

pub fn convert_url_to_hf_format(url: &str) -> (String, String) {
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
