#[cfg(not(target_os = "macos"))]
use devices::CudaConfig;
#[cfg(target_os = "macos")]
use devices::MetalConfig;
use llm_utils::models::local_model::{
    gguf::GgufLoader, metadata::llm::DEFAULT_CONTEXT_LENGTH, LocalLlmModel,
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
        llm_loader
            .gguf_preset_loader
            .preset_with_available_vram_bytes = Some(self.device_config.available_memory_bytes()?);

        llm_loader.load()
    }
}

pub trait LlmLocalTrait {
    fn config(&mut self) -> &mut LocalLlmConfig;

    /// If enabled, any issues with the configuration will result in an error.
    /// Otherwise, fallbacks will be used.
    /// Useful if you have a specific configuration in mind and want to ensure it is used.
    ///
    /// # Arguments
    ///
    /// * `error_on_config_issue` - A boolean indicating whether to error on configuration issues.
    ///
    /// # Default
    ///
    /// Defaults to false.
    fn error_on_config_issue(mut self, error_on_config_issue: bool) -> Self
    where
        Self: Sized,
    {
        self.config().device_config.error_on_config_issue = error_on_config_issue;
        self
    }

    /// Enables or disables GPU usage for inference.
    ///
    /// # Arguments
    ///
    /// * `use_gpu` - A boolean indicating whether to use GPU (true) or not (false).
    ///
    /// # Notes
    ///
    /// On macOS, this setting affects Metal usage. On other platforms, it typically
    /// affects CUDA usage.
    ///
    /// # Default
    ///
    /// Defaults to true. If set to false, CPU inference will be used.
    fn use_gpu(mut self, use_gpu: bool) -> Self
    where
        Self: Sized,
    {
        self.config().device_config.use_gpu = use_gpu;
        self
    }

    #[cfg(target_os = "macos")]
    /// Enables or disables Metal usage for inference on macOS.
    ///
    /// # Arguments
    ///
    /// * `use_metal` - A boolean indicating whether to use Metal (true) or not (false).
    ///
    /// # Notes
    ///
    /// This method is only available on macOS and is equivalent to `use_gpu`.
    ///
    /// # Default
    ///
    /// Defaults to true on macOS.
    fn use_metal(mut self, use_metal: bool) -> Self
    where
        Self: Sized,
    {
        self.config().device_config.use_gpu = use_metal;
        self
    }

    /// Disables GPU usage and forces CPU-only inference.
    ///
    /// # Notes
    ///
    /// This is equivalent to calling `use_gpu(false)`.
    ///
    /// # Default
    ///
    /// Defaults to false.
    fn cpu_only(mut self) -> Self
    where
        Self: Sized,
    {
        self.config().device_config.use_gpu = false;
        self
    }

    /// Sets the number of CPU threads to use for inference.
    ///
    /// # Arguments
    ///
    /// * `threads` - The number of CPU threads to use.
    ///
    /// # Notes
    ///
    /// If loading purely in VRAM, this defaults to 1.
    fn threads(mut self, threads: i16) -> Self
    where
        Self: Sized,
    {
        self.config().device_config.cpu_config.threads = Some(threads);
        self
    }

    /// Sets the number of CPU threads to use for batching and prompt processing.
    ///
    /// # Arguments
    ///
    /// * `threads_batch` - The number of CPU threads to use for batching and prompt processing.
    ///
    /// # Default
    ///
    /// If not set, defaults to a percentage of the total system threads.
    fn threads_batch(mut self, threads_batch: i16) -> Self
    where
        Self: Sized,
    {
        self.config().device_config.cpu_config.threads_batch = Some(threads_batch);
        self
    }

    /// Sets the batch size for inference.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - The batch size to use.
    ///
    /// # Default
    ///
    /// If not set, defaults to 512.
    fn batch_size(mut self, batch_size: u64) -> Self
    where
        Self: Sized,
    {
        self.config().batch_size = batch_size;
        self
    }

    /// Sets the inference context size (maximum token limit for inference output).
    ///
    /// # Arguments
    ///
    /// * `inference_ctx_size` - The maximum number of tokens the model can generate as output.
    ///
    /// # Notes
    ///
    /// This value is set when the model is loaded and cannot be changed after.
    /// If not set, a default value will be used.
    fn inference_ctx_size(mut self, inference_ctx_size: u64) -> Self
    where
        Self: Sized,
    {
        self.config().inference_ctx_size = inference_ctx_size;
        self
    }

    /// Sets the amount of RAM to use for inference.
    ///
    /// # Arguments
    ///
    /// * `available_ram_gb` - The amount of RAM to use, in gigabytes.
    ///
    /// # Effects
    ///
    /// - On macOS: Affects all inference operations.
    /// - On Windows and Linux: Affects CPU inference only.
    ///
    /// # Default Behavior
    ///
    /// If this method is not called, the amount of RAM used will default to a percentage
    /// of the total system RAM. See `use_ram_percentage` for details on setting this percentage.
    ///
    /// # Notes
    ///
    /// The input value is converted to bytes internally. Precision may be affected for
    /// very large values due to floating-point to integer conversion.
    fn use_ram_gb(mut self, available_ram_gb: f32) -> Self
    where
        Self: Sized,
    {
        self.config().device_config.ram_config.use_ram_bytes =
            (available_ram_gb * 1_073_741_824f32) as u64;
        #[cfg(target_os = "macos")]
        {
            self.config().device_config.metal_config.use_ram_bytes =
                (available_ram_gb * 1_073_741_824f32) as u64;
        }
        self
    }

    /// Sets the percentage of total system RAM to use for inference.
    ///
    /// # Arguments
    ///
    /// * `use_ram_percentage` - The percentage of total system RAM to use, expressed as a float
    ///   between 0.0 and 1.0.
    ///
    /// # Effects
    ///
    /// - On macOS: Affects all inference operations.
    /// - On Windows and Linux: Affects CPU inference only.
    ///
    /// # Default Behavior
    ///
    /// If neither this method nor `use_ram_gb` is called, the system will use 70% (0.7) of
    /// the available RAM by default for windows and linux or 90% (0.9) for macOS.
    ///
    /// # Precedence
    ///
    /// This setting is only used if `use_ram_gb` has not been called. If `use_ram_gb` has been
    /// set, that value takes precedence over the percentage set here.
    ///
    /// # Notes
    ///
    /// It's recommended to set this value conservatively to avoid potential system instability
    /// or performance issues caused by memory pressure.
    fn use_ram_percentage(mut self, use_ram_percentage: f32) -> Self
    where
        Self: Sized,
    {
        self.config().device_config.ram_config.use_percentage = use_ram_percentage;
        #[cfg(target_os = "macos")]
        {
            self.config().device_config.metal_config.use_percentage = use_ram_percentage;
        }
        self
    }

    #[cfg(not(target_os = "macos"))]
    /// Sets the CUDA configuration for GPU inference.
    ///
    /// # Arguments
    ///
    /// * `cuda_config` - The CUDA configuration to use.
    ///
    /// # Notes
    ///
    /// This method is only available on non-macOS platforms.
    /// If not set, CUDA devices will be automatically detected.
    fn cuda_config(mut self, cuda_config: CudaConfig) -> Self
    where
        Self: Sized,
    {
        self.config().device_config.cuda_config = Some(cuda_config);
        self
    }

    #[cfg(target_os = "macos")]
    /// Sets the Metal configuration for GPU inference on macOS.
    ///
    /// # Arguments
    ///
    /// * `metal_config` - The Metal configuration to use.
    ///
    /// # Notes
    ///
    /// This method is only available on macOS.
    fn metal_config(mut self, metal_config: MetalConfig) -> Self
    where
        Self: Sized,
    {
        self.config().device_config.metal_config = Some(metal_config);
        self
    }
}
