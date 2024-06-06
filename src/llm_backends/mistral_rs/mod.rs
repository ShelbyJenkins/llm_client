use crate::{
    logging,
    response::parse_text_generation_response,
    LlmBackend,
    LlmClient,
    RequestConfig,
};
use anyhow::{anyhow, Result};
use llm_utils::models::{
    open_source::{GGUFModelBuilder, LlmPreset, PresetModelBuilder},
    OsLlm,
};
use mistralrs::{
    Constraint,
    Device,
    DeviceMapMetadata,
    GGUFLoaderBuilder,
    GGUFSpecificConfig,
    LocalModelPaths,
    MistralRs,
    MistralRsBuilder,
    ModelPaths,
    NormalRequest,
    Request,
    RequestMessage,
    Response,
    SamplingParams,
    SchedulerMethod,
};
use std::{path::PathBuf, sync::Arc};
use tokio::sync::mpsc::channel;

pub const DEFAULT_N_GPU_LAYERS: u16 = 20;

pub struct MistraRsBackend {
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
    client: Option<Arc<MistralRs>>,
    tracing_guard: Option<tracing::subscriber::DefaultGuard>,
}

impl Default for MistraRsBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl MistraRsBackend {
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
            client: None,
            tracing_guard: None,
        }
    }
    async fn setup(&mut self) -> Result<()> {
        if self.client.is_some() {
            return Ok(());
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

        // Create the MistralRs, which is a runner
        self.client = Some(self.load_from_gguf_local()?);

        Ok(())
    }

    fn load_from_gguf_local(&self) -> Result<Arc<MistralRs>> {
        std::env::set_var("MISTRALRS_DEBUG", "1");
        let loader = GGUFLoaderBuilder::new(
            GGUFSpecificConfig { repeat_last_n: 64 },
            self.model
                .as_ref()
                .unwrap()
                .chat_template
                .chat_template_path
                .clone(),
            None,
            "".to_string(),
            "".to_string(),
        )
        .build();

        let local_model_paths: Vec<PathBuf> = self
            .model
            .as_ref()
            .unwrap()
            .local_model_paths
            .iter()
            .map(PathBuf::from)
            .collect();

        let paths: Box<dyn ModelPaths> = Box::new(LocalModelPaths::new(
            self.model
                .as_ref()
                .unwrap()
                .tokenizer
                .as_ref()
                .unwrap()
                .tokenizer_path
                .clone()
                .unwrap(),
            PathBuf::new(),
            PathBuf::new(),
            local_model_paths,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ));

        let pipeline = loader.load_model_from_path(
            &paths,
            None,
            &Device::cuda_if_available(0)?,
            false,
            DeviceMapMetadata::from_num_device_layers(self.n_gpu_layers.into()),
            None,
        )?;

        Ok(
            MistralRsBuilder::new(pipeline, SchedulerMethod::Fixed(5.try_into().unwrap()))
                .with_log("logs/mistralrs_gguf_tensors.txt".to_string())
                .build(),
        )
    }

    pub async fn init(mut self) -> Result<LlmClient> {
        if self.logging_enabled {
            self.tracing_guard = Some(logging::create_logger("mistralrs_backend"));
        }
        self.setup().await?;
        Ok(LlmClient::new(LlmBackend::MistralRs(self)))
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

    fn client(&self) -> &Arc<MistralRs> {
        self.client.as_ref().unwrap()
    }

    pub async fn text_generation_request(
        &self,
        req_config: &RequestConfig,
        grammar: Option<&String>,
    ) -> Result<String> {
        let prompt = req_config.chat_template_prompt.as_ref().unwrap();

        let sampling_params = SamplingParams {
            temperature: Some(req_config.temperature.into()),
            top_p: Some(req_config.top_p.into()),
            frequency_penalty: Some(req_config.frequency_penalty),
            presence_penalty: Some(req_config.presence_penalty),
            max_len: req_config.actual_request_tokens.map(|val| val as usize),
            logits_bias: req_config.logit_bias.clone(),
            ..Default::default()
        };

        let constraint = if let Some(grammar) = grammar {
            Constraint::Regex(grammar.to_owned())
        } else {
            Constraint::None
        };

        let (tx, mut rx) = channel(10_000);
        let request = Request::Normal(NormalRequest {
            messages: RequestMessage::Completion {
                text: prompt.to_owned(),
                echo_prompt: false,
                best_of: 1,
            },
            sampling_params,
            response: tx,
            return_logprobs: false,
            is_streaming: false,
            id: 0,
            constraint,
            suffix: None,
            adapters: None,
        });
        if self.logging_enabled {
            tracing::info!(?request);
        }
        self.client().get_sender().send(request).await?;

        match rx.recv().await {
            None => {
                let error = anyhow::format_err!(
                    "MistraRsBackend text_generation_request error: request returned `None`"
                );

                if self.logging_enabled {
                    tracing::info!(?error);
                }
                Err(error)
            }
            Some(response) => match response {
                Response::InternalError(e) | Response::ValidationError(e) => {
                    let error =
                        anyhow::format_err!("MistraRsBackend text_generation_request error: {}", e);
                    if self.logging_enabled {
                        tracing::info!(?error);
                    }
                    Err(error)
                }
                Response::Chunk(_) | Response::Done(_) => {
                    let error = anyhow::format_err!(
                            "MistraRsBackend text_generation_request error: response is a chat response"
                        );
                    if self.logging_enabled {
                        tracing::info!(?error);
                    }
                    Err(error)
                }
                Response::ModelError(e, _) | Response::CompletionModelError(e, _) => {
                    let error =
                        anyhow::format_err!("MistraRsBackend text_generation_request error: {}", e);
                    if self.logging_enabled {
                        tracing::info!(?error);
                    }
                    Err(error)
                }
                Response::CompletionDone(completion) => {
                    if self.logging_enabled {
                        tracing::info!(?completion);
                    }
                    if completion.choices.is_empty() {
                        let error = anyhow::format_err!(
                                "MistraRsBackend text_generation_request error: completion.content.is_empty()"
                            );
                        Err(error)
                    } else {
                        match parse_text_generation_response(&completion.choices[0].text) {
                            Err(e) => {
                                let error = anyhow::format_err!(
                                    "MistraRsBackend text_generation_request error: {}",
                                    e
                                );
                                if self.logging_enabled {
                                    tracing::info!(?error);
                                }
                                Err(error)
                            }
                            Ok(content) => Ok(content),
                        }
                    }
                }
            },
        }
    }
}
