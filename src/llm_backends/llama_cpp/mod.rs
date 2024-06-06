pub mod api;
pub mod server;
use crate::{
    logging,
    response::parse_text_generation_response,
    LlmBackend,
    LlmClient,
    RequestConfig,
};
use anyhow::{anyhow, Result};
use api::{
    client::LlamaClient,
    config::LlamaConfig,
    types::{
        LlamaCompletionsRequestArgs,
        LlamaCreateDetokenizeRequestArgs,
        LlamaCreateTokenizeRequestArgs,
    },
};
use llm_utils::models::{
    open_source::{GGUFModelBuilder, LlmPreset, PresetModelBuilder},
    OsLlm,
};
use server::ServerProcess;
use std::collections::HashMap;

const LLAMA_PATH: &str = "src/llm_backends/llama_cpp/llama_cpp";
pub const DEFAULT_N_GPU_LAYERS: u16 = 20;

pub struct LlamaBackend {
    // Load from preset (default)
    pub open_source_model_type: Option<LlmPreset>,
    pub available_vram: Option<u32>,
    pub ctx_size: u32,
    // Load from hugging face
    pub hf_quant_file_url: Option<String>,
    pub hf_config_repo_id: Option<String>,
    // Load from local
    pub local_quant_file_path: Option<String>,
    pub local_config_path: Option<String>,
    // Load from instantiated model
    pub model: Option<OsLlm>,
    pub hf_token: Option<String>,
    pub threads: u16,
    pub n_gpu_layers: u16,
    pub logging_enabled: bool,
    server_process: Option<ServerProcess>,
    client: Option<LlamaClient<LlamaConfig>>,
    tracing_guard: Option<tracing::subscriber::DefaultGuard>,
}

impl Default for LlamaBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl LlamaBackend {
    pub fn new() -> Self {
        Self {
            open_source_model_type: None,
            available_vram: None,
            hf_quant_file_url: None,
            hf_config_repo_id: None,
            local_quant_file_path: None,
            local_config_path: None,
            model: None,
            hf_token: None,
            ctx_size: 2048,
            threads: 1,
            n_gpu_layers: DEFAULT_N_GPU_LAYERS,
            logging_enabled: true,
            server_process: None,
            client: None,
            tracing_guard: None,
        }
    }
    async fn setup(&mut self) -> Result<()> {
        if self.client.is_none() {
            self.client = Some(LlamaClient::new());
        }

        let model = if let Some(model) = &self.model {
            model
        } else if let Some(hf_quant_file_url) = &self.hf_quant_file_url {
            let mut builder = GGUFModelBuilder::new();
            builder.hf_quant_file_url(hf_quant_file_url);
            if let Some(hf_config_repo_id) = &self.hf_config_repo_id {
                builder.hf_config_repo_id(hf_config_repo_id);
            } else {
                return Err(anyhow!(
                    "hf_config_repo_id must be set when using hf_quant_file_url"
                ));
            }
            if let Some(hf_token) = &self.hf_token {
                builder.hf_token(hf_token);
            }
            let model = builder.load()?;
            self.model = Some(model);
            self.model.as_ref().unwrap()
        } else if let Some(local_quant_file_path) = &self.local_quant_file_path {
            let mut builder = GGUFModelBuilder::new();
            builder.local_quant_file_path(local_quant_file_path);
            if let Some(local_config_path) = &self.local_config_path {
                builder.local_config_path(local_config_path);
            } else {
                return Err(anyhow!(
                    "local_config_path must be set when using local_quant_file_path"
                ));
            }
            if let Some(hf_token) = &self.hf_token {
                builder.hf_token(hf_token);
            }
            let model = builder.load()?;
            self.model = Some(model);
            self.model.as_ref().unwrap()
        } else {
            let mut builder = PresetModelBuilder::new();
            if let Some(hf_token) = &self.hf_token {
                builder.hf_token(hf_token);
            }
            if let Some(open_source_model_type) = &self.open_source_model_type {
                builder.open_source_model_type = open_source_model_type.clone();
            }
            if let Some(available_vram) = self.available_vram {
                builder.quantization_from_vram_gb = available_vram;
            }
            builder.use_ctx_size = self.ctx_size;
            let model = builder.load()?;
            self.hf_quant_file_url = Some(model.model_url.clone());
            // Since the model is guaranteed to be constrained to the vram size, we max n_gpu_layers.
            self.n_gpu_layers = 9999;
            self.model = Some(model);
            self.model.as_ref().unwrap()
        };

        if model.tokenizer.is_none() {
            panic!("Tokenizer did not load correctly.")
        }

        if self.ctx_size > model.model_config_json.max_position_embeddings as u32 {
            eprintln!("Given value for ctx_size {} is greater than the model's max {}. Using the models max.", self.ctx_size, model.model_config_json.max_position_embeddings);
            self.ctx_size = model.model_config_json.max_position_embeddings as u32;
        };

        if self.server_process.is_none() {
            self.server_process = self.start_server().await?;
        }
        Ok(())
    }

    /// Initializes the LlamaBackend and returns the LlmClient for usage.
    pub async fn init(mut self) -> Result<LlmClient> {
        if self.logging_enabled {
            self.tracing_guard = Some(logging::create_logger("llama_backend"));
        }
        self.setup().await?;
        Ok(LlmClient::new(LlmBackend::Llama(self)))
    }

    /// If set, will attemplt too load the largest quantized model that fits into the available VRAM.
    pub fn available_vram(mut self, available_vram: u32) -> Self {
        self.available_vram = Some(available_vram);
        self
    }

    /// Used for setting the context limits of the model, and also for calculating vram usage.
    pub fn ctx_size(mut self, ctx_size: u32) -> Self {
        self.ctx_size = ctx_size;
        self
    }

    /// Set the open source model type to use by passinng in the LlmPreset enum. Used in src/benchmark.
    pub fn open_source_model_type(mut self, open_source_model_type: LlmPreset) -> Self {
        self.open_source_model_type = Some(open_source_model_type);
        self
    }

    /// Use the Llama3_70bInstruct model.
    pub fn llama_3_70b_instruct(mut self) -> Self {
        self.open_source_model_type = Some(LlmPreset::Llama3_70bInstruct);
        self
    }

    /// Use the Llama3_8bInstruct model.
    pub fn llama_3_8b_instruct(mut self) -> Self {
        self.open_source_model_type = Some(LlmPreset::Llama3_8bInstruct);
        self
    }

    /// Use the Mistral7bInstruct model.
    pub fn mistral_7b_instruct(mut self) -> Self {
        self.open_source_model_type = Some(LlmPreset::Mistral7bInstructV0_3);
        self
    }

    /// Use the Mistral8bInstruct model.
    pub fn mixtral_8x7b_instruct(mut self) -> Self {
        self.open_source_model_type = Some(LlmPreset::Mixtral8x7bInstructV0_1);
        self
    }

    /// Use the Phi3Medium4kInstruct model.
    pub fn phi_3_medium_4k_instruct(mut self) -> Self {
        self.open_source_model_type = Some(LlmPreset::Phi3Medium4kInstruct);
        self
    }

    /// Use the Phi3Mini4kInstruct model.
    pub fn phi_3_mini_4k_instruct(mut self) -> Self {
        self.open_source_model_type = Some(LlmPreset::Phi3Mini4kInstruct);
        self
    }

    /// Directly use an instantiated model from llm_utils::models::open_source::OsLlm.
    pub fn model(mut self, model: OsLlm) -> Self {
        self.model = Some(model);
        self
    }

    /// Use a model from a quantized file URL. May require setting ctx_size and n_gpu_layers manually.
    /// Requires setting hf_config_repo_id to load the tokenizer.json from the original model.
    pub fn hf_quant_file_url(mut self, hf_quant_file_url: &str, hf_config_repo_id: &str) -> Self {
        self.hf_quant_file_url = Some(hf_quant_file_url.to_string());
        self.hf_config_repo_id = Some(hf_config_repo_id.to_string());
        self
    }

    /// Use a model from a local quantized file path. May require setting ctx_size and n_gpu_layers manually.
    /// Requires setting local_config_path to load the tokenizer.json from the original model.
    pub fn local_quant_file_path(
        mut self,
        local_quant_file_path: &str,
        local_config_path: &str,
    ) -> Self {
        self.local_quant_file_path = Some(local_quant_file_path.to_string());
        self.local_config_path = Some(local_config_path.to_string());
        self
    }

    /// The number of CPU threads to use. If loading purely in vram, this can be set to 1.
    pub fn threads(mut self, threads: u16) -> Self {
        self.threads = threads;
        self
    }

    /// If using the `available_vram` method, will automatically be set to max.
    pub fn n_gpu_layers(mut self, n_gpu_layers: u16) -> Self {
        self.n_gpu_layers = n_gpu_layers;
        self
    }

    /// Set the Hugging Face API token to use for downloading models. If not set here, will try to load from .env.
    pub fn hf_token(mut self, hf_token: &str) -> Self {
        self.hf_token = Some(hf_token.to_string());
        self
    }

    /// If set to false, will disable logging. By defaults logs to the logs dir.
    pub fn logging_enabled(mut self, logging_enabled: bool) -> Self {
        self.logging_enabled = logging_enabled;
        self
    }

    fn client(&self) -> &LlamaClient<LlamaConfig> {
        self.client.as_ref().unwrap()
    }

    /// A function to create text completions from a given prompt. Called by various agents, and not meant to be called directly.
    pub async fn text_generation_request(
        &self,
        req_config: &RequestConfig,
        logit_bias: Option<&Vec<Vec<serde_json::Value>>>,
        grammar: Option<&String>,
    ) -> Result<String> {
        let prompt = req_config.chat_template_prompt.as_ref().unwrap();
        let mut request_builder = LlamaCompletionsRequestArgs::default()
            .prompt(prompt)
            .n_predict(req_config.actual_request_tokens.unwrap())
            .frequency_penalty(req_config.frequency_penalty)
            .presence_penalty(req_config.presence_penalty)
            .temperature(req_config.temperature)
            .top_p(req_config.top_p)
            .clone();

        if let Some(logit_bias) = logit_bias {
            request_builder.logit_bias(logit_bias.to_owned());
        }
        if let Some(grammar) = grammar {
            request_builder.grammar(grammar);
        }

        let request: api::types::LlamaCompletionsRequest = request_builder.build()?;
        if self.logging_enabled {
            tracing::info!(?request);
        }

        match self.client().completions().create(request).await {
            Err(e) => {
                let error =
                    anyhow::format_err!("LlamaBackend text_generation_request error: {}", e);

                if self.logging_enabled {
                    tracing::info!(?error);
                }
                Err(error)
            }
            Ok(completion) => match parse_text_generation_response(&completion.content) {
                Err(e) => {
                    let error =
                        anyhow::format_err!("LlamaBackend text_generation_request error: {}", e);
                    if self.logging_enabled {
                        tracing::info!(?error);
                    }
                    Err(error)
                }
                Ok(content) => {
                    if self.logging_enabled {
                        tracing::info!(?completion);
                    }
                    Ok(content)
                }
            },
        }
    }

    /// A function to create decisions from a given prompt. Called by various agents, and not meant to be called directly.
    pub async fn decision_request(
        &self,
        req_config: &RequestConfig,
        logit_bias: Option<&Vec<Vec<serde_json::Value>>>,
        stop_words: Option<Vec<String>>,
        grammar: Option<&str>,
    ) -> Result<String> {
        let mut request_builder = LlamaCompletionsRequestArgs::default();
        request_builder
            .prompt(req_config.chat_template_prompt.as_ref().unwrap())
            .frequency_penalty(req_config.frequency_penalty)
            .presence_penalty(req_config.presence_penalty)
            .temperature(req_config.temperature)
            .top_p(req_config.top_p)
            .n_predict(req_config.actual_request_tokens.unwrap());
        if let Some(logit_bias) = logit_bias {
            request_builder.logit_bias(logit_bias.to_owned());
        };
        if let Some(stop_words) = &stop_words {
            request_builder.stop(stop_words.to_owned());
        };
        if let Some(grammar) = grammar {
            request_builder.grammar(grammar.to_owned());
        };
        let request = request_builder.build()?;
        match self.client().completions().create(request).await {
            Err(e) => {
                let error = anyhow::format_err!("LlamaBackend decision_request error: {}", e);
                if self.logging_enabled {
                    tracing::info!(?error);
                }
                Err(error)
            }
            Ok(completion) => match parse_text_generation_response(&completion.content) {
                Err(e) => {
                    let error =
                        anyhow::format_err!("LlamaBackend text_generation_request error: {}", e);
                    if self.logging_enabled {
                        tracing::info!(?error);
                    }
                    Err(error)
                }
                Ok(content) => {
                    if self.logging_enabled {
                        tracing::info!(?completion);
                    }
                    Ok(content)
                }
            },
        }
    }

    /// Use llama.cpp as a tokenizer. Hopefully we can migrate off this.
    pub async fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let request = LlamaCreateTokenizeRequestArgs::default()
            .content(text)
            .build()?;
        Ok(self.client().tokenize().create(request).await?.tokens)
    }

    pub async fn try_into_single_token(&self, try_into_single_token: &str) -> Result<u32> {
        let request = LlamaCreateTokenizeRequestArgs::default()
            .content(try_into_single_token)
            .build()?;
        let response_tokens = self.client().tokenize().create(request).await?.tokens;
        // Llama.cpp /tokenize sometimes returns an extra whitespace token '/s' for a single character,
        // so we need to filter out the correct one.
        if response_tokens.len() == 1 {
            return Ok(response_tokens[0]);
        }
        if response_tokens.len() > 1 {
            for token in &response_tokens {
                let detokenize_response = self.detokenize_token(*token).await?;
                if detokenize_response == try_into_single_token {
                    return Ok(*token);
                }
                let strings_maybe: Vec<String> = detokenize_response
                    .trim()
                    .split_ascii_whitespace()
                    .map(|s| s.trim().to_string())
                    .collect();
                for maybe in strings_maybe {
                    if maybe == try_into_single_token {
                        return Ok(*token);
                    }
                }
            }
            return Err(anyhow!(
                "More than one token found in text: {}",
                try_into_single_token
            ));
        }
        Err(anyhow!(
            "No token found for try_into_single_token: {:?}",
            try_into_single_token,
        ))
    }

    pub async fn count_tokens(&self, text: &String) -> Result<u16> {
        let request = LlamaCreateTokenizeRequestArgs::default()
            .content(text)
            .build()?;
        let response = self.client().tokenize().create(request).await?;

        Ok(response.tokens.len() as u16)
    }

    pub async fn detokenize(&self, tokens: Vec<u32>) -> Result<String> {
        let request = LlamaCreateDetokenizeRequestArgs::default()
            .tokens(tokens.clone())
            .build()?;
        let response = self.client().detokenize().create(request).await?;
        Ok(response.content)
    }

    /// Detokenizes a single token into text. Used for checking returned tokens from try_into_single_token.
    ///
    /// # Arguments
    ///
    /// * `tokens` - A token to detokenize.
    ///
    /// # Returns
    ///
    /// A `Result` containing the detokenized string on success, or an error failure.
    pub async fn detokenize_token(&self, token: u32) -> Result<String> {
        let request = LlamaCreateDetokenizeRequestArgs::default()
            .tokens([token])
            .build()?;
        let response = self.client().detokenize().create(request).await?;
        Ok(response.content)
    }

    pub async fn try_from_single_token_id(&self, try_from_single_token_id: u32) -> Result<String> {
        let detokenize_response = self.detokenize_token(try_from_single_token_id).await?;
        println!("detokenize_response: {}", detokenize_response);
        let mut strings_maybe: Vec<String> = detokenize_response
            .split_ascii_whitespace()
            .map(|s| s.to_string())
            .collect();
        match strings_maybe.len() {
            0 => Err(anyhow!(
                "token_id is empty for try_from_single_token_id: {}",
                try_from_single_token_id
            )),
            1 => Ok(strings_maybe.remove(0)),
            n => Err(anyhow!(
                "Found more than one token ({n} total) in try_from_single_token_id: {}",
                try_from_single_token_id
            )),
        }
    }

    pub async fn validate_logit_bias_token_ids(
        &self,
        logit_bias: &std::collections::HashMap<u32, f32>,
    ) -> Result<()> {
        for token_id in logit_bias.keys() {
            self.try_from_single_token_id(*token_id).await?;
        }
        Ok(())
    }

    pub async fn logit_bias_from_chars(
        &self,
        logit_bias: &HashMap<char, f32>,
    ) -> Result<HashMap<u32, f32>> {
        let mut token_logit_bias: HashMap<u32, f32> = HashMap::new();
        for (char, bias) in logit_bias {
            let token_id = self.try_into_single_token(&char.to_string()).await?;
            token_logit_bias.insert(token_id, *bias);
        }
        Ok(token_logit_bias)
    }

    pub async fn logit_bias_from_words(
        &self,
        logit_bias: &HashMap<String, f32>,
    ) -> Result<HashMap<u32, f32>> {
        let mut token_logit_bias: HashMap<u32, f32> = HashMap::new();
        for (word_maybe, bias) in logit_bias {
            let mut words_maybe: Vec<String> = word_maybe
                .trim()
                .split_ascii_whitespace()
                .map(|s| s.trim().to_string())
                .collect();
            let word = if words_maybe.is_empty() {
                return Err(anyhow!(
                    "logit_bias contains an empty word. Given word: {}",
                    word_maybe
                ));
            } else if words_maybe.len() > 1 {
                return Err(anyhow!(
                    "logit_bias contains a word seperated by whitespace. Given word: {}",
                    word_maybe
                ));
            } else {
                words_maybe.remove(0)
            };
            let token_ids = self.tokenize(&word).await?;
            for id in token_ids {
                // if id == tokenizer.white_space_token_id {
                //     panic!(
                //         "logit_bias contains a whitespace token. Given word: {}",
                //         word
                //     )
                // }
                // token_logit_bias.insert(id, *bias);
                token_logit_bias.insert(id, *bias);
            }
        }
        Ok(token_logit_bias)
    }

    pub async fn logit_bias_from_texts(
        &self,
        logit_bias: &HashMap<String, f32>,
    ) -> Result<HashMap<u32, f32>> {
        let mut token_logit_bias: HashMap<u32, f32> = HashMap::new();
        for (text, bias) in logit_bias {
            let token_ids = self.tokenize(text).await?;
            for id in token_ids {
                // if id == tokenizer.white_space_token_id {
                //     continue;
                // }
                // token_logit_bias.insert(id, *bias);
                token_logit_bias.insert(id, *bias);
            }
        }
        Ok(token_logit_bias)
    }
}
