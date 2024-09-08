use super::{devices::get_device_map, LlmBackend, MistraRsBackend};
use crate::{logging, LlmClient};
use llm_utils::models::open_source_model::{
    gguf::GgufLoader,
    GgufLoaderTrait,
    HfTokenTrait,
    LlmPresetLoader,
    LlmPresetTrait,
    OsLlm,
};
use mistralrs::{
    DefaultSchedulerMethod,
    Device,
    DeviceMapMetadata,
    GGUFLoaderBuilder,
    GGUFSpecificConfig,
    LocalModelPaths,
    MemoryGpuConfig,
    MistralRs,
    MistralRsBuilder,
    ModelDType,
    ModelPaths,
    PagedAttentionConfig,
    SchedulerConfig,
};

pub const DEFAULT_N_GPU_LAYERS: u16 = 20;

pub struct MistraRsBackendBuilder {
    pub logging_enabled: bool,
    pub ctx_size: u32,
    pub n_gpu_layers: u16,
    pub llm_loader: GgufLoader,
}

impl Default for MistraRsBackendBuilder {
    fn default() -> Self {
        MistraRsBackendBuilder {
            logging_enabled: true,
            ctx_size: 4096,
            n_gpu_layers: DEFAULT_N_GPU_LAYERS,
            llm_loader: GgufLoader::default(),
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

    pub async fn init(mut self) -> crate::Result<LlmClient> {
        let _tracing_guard = if self.logging_enabled {
            Some(logging::create_logger("mistralrs_backend"))
        } else {
            None
        };
        let model = self.load_model()?;
        let client: std::sync::Arc<MistralRs> = self.load_from_gguf_local(&model).await?;

        Ok(LlmClient {
            backend: std::rc::Rc::new(LlmBackend::MistralRs(MistraRsBackend {
                model,
                client,
                ctx_size: self.ctx_size,
                _tracing_guard,
            })),
        })
    }

    pub fn load_model(&mut self) -> crate::Result<OsLlm> {
        if self.llm_loader.local_quant_file_path.is_none()
            || self.llm_loader.hf_quant_file_url.is_none()
        {
            if let Some(use_ctx_size) = self.llm_loader.preset_loader.use_ctx_size {
                self.ctx_size = use_ctx_size; // If the preset loader has a ctx_size set, we use that.
            } else {
                self.llm_loader.preset_loader.use_ctx_size = Some(self.ctx_size);
                // Otherwise we set the preset loader to use the ctx_size from server_config.
            }
            self.n_gpu_layers = 9999; // Since the model is guaranteed to be constrained to the vram size, we max n_gpu_layers.
        };

        let model = self.llm_loader.load()?;
        if self.ctx_size > model.model_metadata.max_position_embeddings as u32 {
            eprintln!("Given value for ctx_size {} is greater than the model's max {}. Using the models max.", self.ctx_size, model.model_metadata.max_position_embeddings);
            self.ctx_size = model.model_metadata.max_position_embeddings as u32;
        };
        Ok(model)
    }

    async fn load_from_gguf_local(
        &self,
        model: &OsLlm,
    ) -> crate::Result<std::sync::Arc<MistralRs>> {
        std::env::set_var("MISTRALRS_DEBUG", "1");

        let loader = GGUFLoaderBuilder::new(
            None,
            None,
            "".to_string(),
            vec![],
            GGUFSpecificConfig {
                prompt_batchsize: None,
                topology: None,
            },
        )
        .build();

        let local_model_paths: Vec<std::path::PathBuf> = vec![model.local_model_path.clone()];

        let paths: Box<dyn ModelPaths> = Box::new(LocalModelPaths {
            tokenizer_filename: std::path::PathBuf::new(),
            config_filename: std::path::PathBuf::new(),
            template_filename: None,
            filenames: local_model_paths,
            xlora_adapter_filenames: None,
            xlora_adapter_configs: None,
            classifier_path: None,
            classifier_config: None,
            xlora_ordering: None,
            gen_conf: None,
            lora_preload_adapter_info: None,
            preprocessor_config: None,
            processor_config: None,
        });

        let device_map = get_device_map(model.model_metadata.num_hidden_layers);

        let pipeline = loader.load_model_from_path(
            &paths,
            &ModelDType::default(),
            &Device::cuda_if_available(0)?,
            false,
            device_map,
            None,
            None,
        )?;

        Ok(MistralRsBuilder::new(
            pipeline,
            SchedulerConfig::DefaultScheduler {
                method: DefaultSchedulerMethod::Fixed(16.try_into().unwrap()),
            },
        )
        .with_log("logs/mistralrs_gguf_tensors.txt".to_string())
        .build())
    }
}

impl LlmPresetTrait for MistraRsBackendBuilder {
    fn preset_loader(&mut self) -> &mut LlmPresetLoader {
        &mut self.llm_loader.preset_loader
    }
}

impl GgufLoaderTrait for MistraRsBackendBuilder {
    fn gguf_loader(&mut self) -> &mut GgufLoader {
        &mut self.llm_loader
    }
}

impl HfTokenTrait for MistraRsBackendBuilder {
    fn hf_token_mut(&mut self) -> &mut Option<String> {
        &mut self.llm_loader.hf_loader.hf_token
    }

    fn hf_token_env_var_mut(&mut self) -> &mut String {
        &mut self.llm_loader.hf_loader.hf_token_env_var
    }
}
