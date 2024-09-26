use devices::CudaDeviceMap;
use llm_utils::models::local_model::{
    gguf::GgufLoader,
    metadata::llm::DEFAULT_CONTEXT_LENGTH,
    LocalLlmModel,
};

#[cfg(feature = "llama_cpp_backend")]
pub mod llama_cpp;
#[cfg(feature = "mistral_rs_backend")]
pub mod mistral_rs;

pub mod devices;

#[derive(Clone, Debug)]
pub struct LocalLlmConfig {
    pub batch_size: u64,
    pub inference_ctx_size: u64,
    pub device_config: devices::DeviceConfig,
}

impl Default for LocalLlmConfig {
    fn default() -> Self {
        Self {
            batch_size: 512,
            inference_ctx_size: DEFAULT_CONTEXT_LENGTH,
            device_config: devices::DeviceConfig::default(),
        }
    }
}

impl LocalLlmConfig {
    pub fn load_model(&mut self, mut llm_loader: GgufLoader) -> crate::Result<LocalLlmModel> {
        let model = if llm_loader.gguf_local_loader.local_quant_file_path.is_none()
            || llm_loader.gguf_hf_loader.hf_quant_file_url.is_none()
        {
            self.load_preset_model(llm_loader)?
        } else {
            llm_loader.load()?
        };

        if self.inference_ctx_size > model.model_metadata.context_length() {
            eprintln!("Given value for ctx_size {} is greater than the model's max {}. Using the models max.", self.inference_ctx_size, model.model_metadata.context_length());
            self.inference_ctx_size = model.model_metadata.context_length();
        };

        self.device_config.layer_count = Some(model.model_metadata.layers.count_blocks());
        self.device_config.average_layer_size_bytes = Some(
            model
                .model_metadata
                .average_layer_size_bytes(self.inference_ctx_size, Some(self.batch_size))?,
        );
        self.device_config.local_model_path = model.local_model_path.to_string_lossy().to_string();

        Ok(model)
    }

    fn load_preset_model(&mut self, mut llm_loader: GgufLoader) -> crate::Result<LocalLlmModel> {
        if llm_loader
            .gguf_preset_loader
            .preset_with_quantization_level
            .is_some()
        {
            return llm_loader.load();
        };

        if let Some(preset_with_max_ctx_size) =
            llm_loader.gguf_preset_loader.preset_with_max_ctx_size
        {
            if self.inference_ctx_size > preset_with_max_ctx_size {
                crate::info!(
                        "Given value for ctx_size {} is greater than preset_with_max_ctx_size {preset_with_max_ctx_size}. Using preset_with_max_ctx_size.", self.inference_ctx_size
                    );
                self.inference_ctx_size = preset_with_max_ctx_size;
            };
        } else {
            llm_loader.gguf_preset_loader.preset_with_max_ctx_size = Some(self.inference_ctx_size);
        }
        match (
            self.device_config.use_gpu,
            self.device_config.available_vram_bytes(),
        ) {
            (true, Ok(vram_bytes)) => {
                llm_loader
                    .gguf_preset_loader
                    .preset_with_available_vram_bytes = Some(vram_bytes);
            }
            (true, Err(_)) | (false, _) => {
                // Either use_gpu is false or available_vram_bytes() returned an error
                llm_loader
                    .gguf_preset_loader
                    .preset_with_available_vram_bytes = Some(
                    self.device_config
                        .cpu_config
                        .set_memory_utilization_for_inference(0.5)?,
                );
            }
        }

        llm_loader.load()
    }
}

pub trait LlmLocalTrait {
    fn config(&mut self) -> &mut LocalLlmConfig;

    fn use_gpu(mut self, use_gpu: bool) -> Self
    where
        Self: Sized,
    {
        self.config().device_config.use_gpu = use_gpu;
        self
    }

    fn cpu_only(mut self) -> Self
    where
        Self: Sized,
    {
        self.config().device_config.use_gpu = false;
        self
    }

    /// The number of CPU threads to use. If loading purely in vram, this defaults to 1.
    fn threads(mut self, threads: i16) -> Self
    where
        Self: Sized,
    {
        self.config().device_config.cpu_config.threads = Some(threads);
        self
    }

    /// The number of CPU threads to use for batching and prompt processing.
    fn threads_batch(mut self, threads_batch: i16) -> Self
    where
        Self: Sized,
    {
        self.config().device_config.cpu_config.threads_batch = Some(threads_batch);
        self
    }

    /// Defaults to 512.
    fn batch_size(mut self, batch_size: u64) -> Self
    where
        Self: Sized,
    {
        self.config().batch_size = batch_size;
        self
    }

    /// Maximum token limit for inference output.
    ///
    /// This value represents the maximum number of tokens the model can generate
    /// as output. It's set when the model is loaded and cannot be changed after.
    /// If it's not set, a default value will be used.
    fn inference_ctx_size(mut self, inference_ctx_size: u64) -> Self
    where
        Self: Sized,
    {
        self.config().inference_ctx_size = inference_ctx_size;
        self
    }

    /// If you're using only the CPU, you can set the available RAM in GB.
    /// Otherwise defaults to some percentage of the total system RAM.
    fn available_ram_gb(mut self, available_ram_gb: f32) -> Self
    where
        Self: Sized,
    {
        self.config().device_config.cpu_config.use_ram_bytes =
            Some((available_ram_gb * 1_073_741_824f32) as u64);
        self
    }

    /// The CUDA devices to use for loading the model.
    /// If not set, the devices will be automatically detected.
    fn cuda_device_map(mut self, cuda_device_map: CudaDeviceMap) -> Self
    where
        Self: Sized,
    {
        self.config().device_config.cuda_map = Some(cuda_device_map);
        self
    }
}
