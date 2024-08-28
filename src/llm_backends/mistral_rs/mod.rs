use super::LlmBackend;
use crate::{
    components::{base_request::BaseLlmRequest, response::LlmClientResponse},
    logging,
    LlmClient,
};
use anyhow::Result;
use llm_utils::models::open_source::*;
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
    pub model: OsLlm,
    client: Arc<MistralRs>,
    _tracing_guard: Option<tracing::subscriber::DefaultGuard>,
    pub ctx_size: u32,
}

impl MistraRsBackend {
    pub async fn llm_request(&self, base: &BaseLlmRequest) -> Result<LlmClientResponse> {
        let sampling_params = SamplingParams {
            temperature: Some(base.config.temperature.into()),
            top_p: Some(base.config.top_p.into()),
            frequency_penalty: Some(base.config.frequency_penalty),
            presence_penalty: Some(base.config.presence_penalty),
            max_len: base.config.actual_request_tokens.map(|val| val as usize),
            // logits_bias: req.logit_bias.clone(),
            ..Default::default()
        };

        let constraint = Constraint::None;

        let (tx, mut rx) = channel(10_000);
        let request = Request::Normal(NormalRequest {
            messages: RequestMessage::Completion {
                text: base.prompt.built_chat_template_prompt.clone().unwrap(),
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
        tracing::info!(?request);

        self.client.get_sender().send(request).await?;

        match rx.recv().await {
            None => Err(anyhow::format_err!(
                "MistraRsBackend completion_request error: request returned `None`"
            )),
            Some(response) => match response {
                Response::InternalError(e) | Response::ValidationError(e) => Err(
                    anyhow::format_err!("MistraRsBackend completion_request error: {}", e),
                ),
                Response::Chunk(_) | Response::Done(_) => Err(anyhow::format_err!(
                    "MistraRsBackend completion_request error: response is a chat response"
                )),
                Response::ModelError(e, _) | Response::CompletionModelError(e, _) => Err(
                    anyhow::format_err!("MistraRsBackend completion_request error: {}", e),
                ),
                Response::CompletionDone(completion) => {
                    tracing::info!(?completion);
                    if completion.choices.is_empty() {
                        Err( anyhow::format_err!(
                                "MistraRsBackend completion_request error: completion.content.is_empty()"
                            ))
                    } else {
                        Ok(LlmClientResponse::new_from_mistral(completion))
                    }
                }
            },
        }
    }
}

pub struct MistraRsBackendBuilder {
    pub logging_enabled: bool,
    pub ctx_size: u32,
    pub n_gpu_layers: u16,
    pub llm_loader: OsLlmLoader,
}

impl Default for MistraRsBackendBuilder {
    fn default() -> Self {
        MistraRsBackendBuilder {
            logging_enabled: true,
            ctx_size: 4096,
            n_gpu_layers: DEFAULT_N_GPU_LAYERS,
            llm_loader: OsLlmLoader::default(),
        }
    }
}

impl MistraRsBackendBuilder {
    pub fn new() -> Self {
        MistraRsBackendBuilder::default()
    }

    /// If set to false, will disable logging. By defaults logs to the logs dir.
    pub fn logging_enabled(mut self, logging_enabled: bool) -> Self {
        self.logging_enabled = logging_enabled;
        self
    }

    /// Used for setting the context limits of the model, and also for calculating vram usage.
    pub fn ctx_size(mut self, ctx_size: u32) -> Self {
        self.ctx_size = ctx_size;
        self
    }

    /// If using the `available_vram` method, will automatically be set to max.
    pub fn n_gpu_layers(mut self, n_gpu_layers: u16) -> Self {
        self.n_gpu_layers = n_gpu_layers;
        self
    }

    pub fn init(mut self) -> Result<LlmClient> {
        let _tracing_guard = if self.logging_enabled {
            Some(logging::create_logger("mistralrs_backend"))
        } else {
            None
        };
        let model = self.load_model()?;

        let client = self.load_from_gguf_local(&model)?;
        Ok(LlmClient::new(LlmBackend::MistralRs(MistraRsBackend {
            model,
            client,
            ctx_size: self.ctx_size,
            _tracing_guard,
        })))
    }

    pub fn load_model(&mut self) -> Result<OsLlm> {
        if let Some(preset_loader) = &mut self.llm_loader.preset_loader {
            if let Some(ctx_size) = preset_loader.use_ctx_size {
                self.ctx_size = ctx_size; // If the preset loader has a ctx_size set, we use that.
            } else {
                preset_loader.use_ctx_size = Some(self.ctx_size); // Otherwise we set the preset loader to use the ctx_size from server_config.
            }
            self.n_gpu_layers = 9999; // Since the model is guaranteed to be constrained to the vram size, we max n_gpu_layers.
        }
        let model = self.llm_loader.load()?;
        if self.ctx_size > model.model_config_json.max_position_embeddings as u32 {
            eprintln!("Given value for ctx_size {} is greater than the model's max {}. Using the models max.", self.ctx_size, model.model_config_json.max_position_embeddings);
            self.ctx_size = model.model_config_json.max_position_embeddings as u32;
        };
        Ok(model)
    }

    fn load_from_gguf_local(&self, model: &OsLlm) -> Result<Arc<MistralRs>> {
        std::env::set_var("MISTRALRS_DEBUG", "1");
        let loader = GGUFLoaderBuilder::new(
            GGUFSpecificConfig { repeat_last_n: 64 },
            model.chat_template.chat_template_path.clone(),
            None,
            "".to_string(),
            "".to_string(),
        )
        .build();

        let local_model_paths: Vec<PathBuf> =
            model.local_model_path.iter().map(PathBuf::from).collect();

        let paths: Box<dyn ModelPaths> = Box::new(LocalModelPaths::new(
            model
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
}

impl LlmPresetTrait for MistraRsBackendBuilder {
    fn preset_loader(&mut self) -> &mut LlmPresetLoader {
        if self.llm_loader.preset_loader.is_none() {
            self.llm_loader.preset_loader = Some(LlmPresetLoader::new());
        }
        self.llm_loader.preset_loader.as_mut().unwrap()
    }
}

impl LlmGgufTrait for MistraRsBackendBuilder {
    fn gguf_loader(&mut self) -> &mut LlmGgufLoader {
        if self.llm_loader.gguf_loader.is_none() {
            self.llm_loader.gguf_loader = Some(LlmGgufLoader::new());
        }
        self.llm_loader.gguf_loader.as_mut().unwrap()
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use serial_test::serial;

    #[tokio::test]
    #[serial]
    async fn test_builder() -> Result<()> {
        let llm_client = MistraRsBackendBuilder::new().available_vram(40).init()?;
        assert_eq!(llm_client.backend.model_id(), "Phi-3-mini-4k-instruct");
        Ok(())
    }
}
