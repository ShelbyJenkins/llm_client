use crate::{logging, LlmBackend, LlmClient, RequestConfig};
use anyhow::Result;
use candle_core::Device;
use llm_utils::{
    models::{
        gguf::{GGUFModel, GGUFModelBuilder},
        OpenSourceModelType,
    },
    tokenizer::LlmUtilsTokenizer,
};
use mistralrs::{
    Constraint,
    DeviceMapMetadata,
    GGUFLoaderBuilder,
    GGUFSpecificConfig,
    Loader,
    MistralRs,
    MistralRsBuilder,
    NormalLoaderBuilder,
    NormalLoaderType,
    NormalRequest,
    NormalSpecificConfig,
    Request,
    RequestMessage,
    Response,
    SamplingParams,
    SchedulerMethod,
    TokenSource,
};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::sync::mpsc::channel;

pub const DEFAULT_N_GPU_LAYERS: u16 = 20;

pub struct MistraRsBackend {
    // Option 1
    pub open_source_model_type: Option<OpenSourceModelType>,
    pub available_vram: Option<u32>,
    pub ctx_size: u32,
    // Option 2
    pub model_url: Option<String>,
    // Option 3
    pub model: Option<GGUFModel>,
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
        MistraRsBackend {
            open_source_model_type: None,
            available_vram: None,
            model_url: None,
            model: None,
            hf_token: None,
            ctx_size: 4444,
            threads: 1,
            n_gpu_layers: DEFAULT_N_GPU_LAYERS,
            logging_enabled: true,
            client: None,
            tracing_guard: None,
        }
    }

    pub async fn setup(&mut self) -> Result<()> {
        if self.client.is_some() {
            return Ok(());
        }
        let model = if let Some(model) = &self.model {
            model
        } else if let Some(model_url) = &self.model_url {
            let model = GGUFModelBuilder::new(self.hf_token.clone())
                .from_quant_file_url(model_url)
                .load()
                .await?;
            self.model = Some(model);
            self.model.as_ref().unwrap()
        } else {
            let mut builder = GGUFModelBuilder::new(self.hf_token.clone());
            if let Some(open_source_model_type) = &self.open_source_model_type {
                builder.open_source_model_type = open_source_model_type.clone();
            }
            if let Some(available_vram) = self.available_vram {
                builder.quantization_from_vram_gb = available_vram;
            }
            builder.use_ctx_size = self.ctx_size;
            let model = builder.load().await?;
            // Since the model is guaranteed to be constrained to the vram size, we max n_gpu_layers.
            self.n_gpu_layers = 9999;
            self.model = Some(model);
            self.model.as_ref().unwrap()
        };

        if self.ctx_size > model.metadata.context_length {
            eprintln!("Given value for ctx_size {} is greater than the model's max {}. Using the models max.", self.ctx_size, model.metadata.context_length);
            self.ctx_size = model.metadata.context_length;
        };

        // Select a Mistral model
        let loader = GGUFLoaderBuilder::new(
            GGUFSpecificConfig::default(),
            Some("".to_string()),
            None,
            None,
            "mistralai/Mistral-7B-Instruct-v0.3".to_string(),
            model.local_model_path,
        )
        .build();

        let pipeline = loader.load_model_from_path(
            // Issue here
            None,
            &Device::cuda_if_available(0)?,
            false,
            DeviceMapMetadata::from_num_device_layers(self.n_gpu_layers as usize),
            None,
        )?;
        // Create the MistralRs, which is a runner
        self.client = Some(
            MistralRsBuilder::new(pipeline, SchedulerMethod::Fixed(5.try_into().unwrap())).build(),
        );
        Ok(())
    }

    pub fn init(mut self) -> Result<LlmClient> {
        if self.logging_enabled {
            self.tracing_guard = Some(logging::create_logger("anthropic_backend"));
        }
        self.setup()?;
        Ok(LlmClient::new(LlmBackend::MistralRs(self)))
    }

    pub fn available_vram(mut self, available_vram: u32) -> Self {
        self.available_vram = Some(available_vram);
        self
    }

    pub fn ctx_size(mut self, ctx_size: u32) -> Self {
        self.ctx_size = ctx_size;
        self
    }

    pub fn mistral_7b_instruct(mut self) -> Self {
        self.open_source_model_type = Some(OpenSourceModelType::Mistral7bInstructV0_3);
        self
    }

    pub fn mixtral_8x7b_instruct(mut self) -> Self {
        self.open_source_model_type = Some(OpenSourceModelType::Mixtral8x7bInstruct);
        self
    }

    pub fn mixtral_8x22b_instruct(mut self) -> Self {
        self.open_source_model_type = Some(OpenSourceModelType::Mixtral8x22bInstruct);
        self
    }

    pub fn llama_3_70b_instruct(mut self) -> Self {
        self.open_source_model_type = Some(OpenSourceModelType::Llama3_70bInstruct);
        self
    }

    pub fn llama_3_8b_instruct(mut self) -> Self {
        self.open_source_model_type = Some(OpenSourceModelType::Llama3_8bInstruct);
        self
    }

    pub fn model(mut self, model: GGUFModel) -> Self {
        self.model = Some(model);
        self
    }

    pub fn model_url(mut self, model_url: &str) -> Self {
        self.model_url = Some(model_url.to_string());
        self
    }

    pub fn logging_enabled(mut self, logging_enabled: bool) -> Self {
        self.logging_enabled = logging_enabled;
        self
    }

    fn client(&self) -> &Arc<MistralRs> {
        self.client.as_ref().unwrap()
    }

    pub async fn text_generation_request(&self, req_config: &RequestConfig) -> Result<String> {
        let prompt = req_config.chat_template_prompt.as_ref().unwrap();

        let (tx, mut rx) = channel(10_000);
        let request = Request::Normal(NormalRequest {
            messages: RequestMessage::Completion {
                text: prompt.to_owned(),
                echo_prompt: false,
                best_of: 1,
            },
            sampling_params: SamplingParams::default(),
            response: tx,
            return_logprobs: false,
            is_streaming: false,
            id: 0,
            constraint: Constraint::None,
            suffix: None,
            adapters: None,
        });
        self.client().get_sender().blocking_send(request)?;

        let res = rx.blocking_recv().unwrap();
        let mut completion: mistralrs::CompletionResponse = match res {
            Response::CompletionDone(completion) => completion,
            _ => unreachable!(),
        };

        Ok(completion.choices.remove(0).text)
    }
}
