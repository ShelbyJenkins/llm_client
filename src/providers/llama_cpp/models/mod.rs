pub const DEFAULT_THREADS: u16 = 2;
pub const DEFAULT_CTX_SIZE: u16 = 9001;
pub const DEFAULT_N_GPU_LAYERS: u16 = 6;

pub const TEST_LLM_URL_1_CHAT: &str =
    "https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF/blob/main/zephyr-7b-alpha.Q5_K_M.gguf";
pub const TEST_LLM_PROMPT_TEMPLATE_1_CHAT: LlamaPromptFormat = LlamaPromptFormat::Mistral7BChat;

pub const TEST_LLM_URL_2_INSTRUCT: &str =
    "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q8_0.gguf";
pub const TEST_LLM_PROMPT_TEMPLATE_2_INSTRUCT: LlamaPromptFormat =
    LlamaPromptFormat::Mistral7BInstruct;

lazy_static! {
    #[derive(Clone, Debug, Copy)]
    pub static ref TEST_LLM_1_CHAT: LlamaDef = LlamaDef::new(
        TEST_LLM_URL_1_CHAT,
        TEST_LLM_PROMPT_TEMPLATE_1_CHAT,
        Some(DEFAULT_CTX_SIZE),
        Some(DEFAULT_THREADS),
        Some(DEFAULT_N_GPU_LAYERS),
        None,
        Some(true),
    );
    #[derive(Clone, Debug)]
    pub static ref TEST_LLM_2_INSTRUCT: LlamaDef = LlamaDef::new(
        TEST_LLM_URL_2_INSTRUCT,
        TEST_LLM_PROMPT_TEMPLATE_2_INSTRUCT,
        Some(DEFAULT_CTX_SIZE),
        Some(DEFAULT_THREADS),
        Some(DEFAULT_N_GPU_LAYERS),
        None,
        Some(true),
    );
}

const SAFETY_TOKENS: u16 = 10;

#[derive(Clone, Debug)]
pub enum LlamaPromptFormat {
    Mistral7BInstruct,
    Mistral7BChat,
    Mixtral8X7BInstruct,
    SOLAR107BInstructv1,
    None,
}
#[derive(Debug, Clone)]
pub struct LlamaDef {
    pub model_id: String,
    pub model_filename: String,
    pub prompt_format: LlamaPromptFormat,
    pub max_tokens_for_model: u16,
    pub threads: u16,
    pub n_gpu_layers: u16,
    pub embedding_enabled: bool,
    pub logging_enabled: bool,
}

impl LlamaDef {
    pub fn new(
        model_url: &str,
        prompt_format: LlamaPromptFormat,
        ctx_size: Option<u16>,
        threads: Option<u16>,
        n_gpu_layers: Option<u16>,
        embedding_enabled: Option<bool>,
        logging_enabled: Option<bool>,
    ) -> Self {
        let (model_id, model_filename) = convert_url_to_hf_format(model_url);

        LlamaDef {
            model_id,
            model_filename,
            prompt_format,
            max_tokens_for_model: ctx_size.unwrap_or(DEFAULT_CTX_SIZE),
            threads: threads.unwrap_or(DEFAULT_THREADS),
            n_gpu_layers: n_gpu_layers.unwrap_or(DEFAULT_N_GPU_LAYERS),
            embedding_enabled: embedding_enabled.unwrap_or(false),
            logging_enabled: logging_enabled.unwrap_or(true),
        }
    }

    pub fn get_default_model_params(model_definition: &LlamaDef) -> crate::LlmModelParams {
        crate::LlmModelParams {
            model_id: model_definition.model_id.clone(),
            max_tokens_for_model: model_definition.max_tokens_for_model,
            cost_per_k: 0.00,
            tokens_per_message: 0,
            tokens_per_name: 0,
            frequency_penalty: LlamaDef::frequency_penalty(None),
            presence_penalty: LlamaDef::presence_penalty(None),
            temperature: LlamaDef::temperature(None),
            top_p: LlamaDef::top_p(None),
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
        _model_definition: &LlamaDef,
        max_tokens_for_model: Option<u16>,
    ) -> u16 {
        max_tokens_for_model.unwrap_or(DEFAULT_CTX_SIZE)
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
