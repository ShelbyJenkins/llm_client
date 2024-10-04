use cpu::CpuConfig;
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub use cuda::CudaConfig;
use gpu::GpuLayerAllocator;
#[cfg(target_os = "macos")]
pub use metal::MetalConfig;
use ram::RamConfig;
pub mod cpu;
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub mod cuda;
pub mod gpu;
#[cfg(target_os = "macos")]
pub mod metal;
pub mod ram;

/// Configuration for device-specific settings in LLM inference.
#[derive(Debug, Clone)]
pub struct DeviceConfig {
    /// CPU configuration for thread count.
    pub cpu_config: CpuConfig,

    /// RAM configuration for non-GPU inference on Windows and Unix.
    ///
    /// This setting is used when GPU acceleration is not available or not enabled.
    pub ram_config: RamConfig,

    /// Indicates whether to use any available GPUs for inference.
    ///
    /// If true, the system will attempt to use GPU acceleration.
    /// If false, inference will be performed on CPU only.
    pub use_gpu: bool,

    /// CUDA configuration for GPU inference on non-macOS platforms.
    ///
    /// This field is only available on platforms other than macOS.
    /// If None, default CUDA settings will be used when GPU is enabled.
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub cuda_config: Option<CudaConfig>,

    /// Metal configuration for GPU inference on macOS.
    ///
    /// This field is only available on macOS.
    /// If None, default Metal settings will be used when GPU is enabled.
    #[cfg(target_os = "macos")]
    pub metal_config: Option<MetalConfig>,

    /// Determines error handling behavior for configuration issues.
    ///
    /// If true, the system will return an error when encountering configuration issues.
    /// If false (default), issues will be logged and execution will continue if possible.
    ///
    /// This flag is useful for debugging purposes.
    pub error_on_config_issue: bool,

    /// The number of layers in the model.
    ///
    /// This is set at runtime.
    pub layer_count: Option<u64>,

    /// The average size of a layer in bytes.
    ///
    /// This is set at runtime.
    pub average_layer_size_bytes: Option<u64>,

    /// The file system path to the local model.
    ///
    /// This is set at runtime.
    pub local_model_path: String,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            cpu_config: CpuConfig::default(),
            ram_config: RamConfig::default(),
            use_gpu: true,
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            cuda_config: None,
            #[cfg(target_os = "macos")]
            metal_config: None,
            error_on_config_issue: false,
            layer_count: None,
            average_layer_size_bytes: None,
            local_model_path: Default::default(),
        }
    }
}

impl DeviceConfig {
    pub fn initialize(&mut self) -> crate::Result<()> {
        self.cpu_config.initialize(self.error_on_config_issue)?;
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            self.initialize_unix_windows()?;
        }

        #[cfg(target_os = "macos")]
        {
            self.initialize_mac()?;
        }

        #[cfg(not(any(unix, windows, target_os = "macos")))]
        {
            crate::bail!("Unsupported OS");
        }
        crate::info!("{}", self);
        Ok(())
    }

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn initialize_unix_windows(&mut self) -> crate::Result<()> {
        if self.use_gpu {
            if self.cuda_config.is_none() {
                let cuda_config = CudaConfig::default();
                self.cuda_config = Some(cuda_config);
            }
            if let Some(cuda_config) = &mut self.cuda_config {
                match cuda_config.initialize(self.error_on_config_issue) {
                    Ok(_) => (),
                    Err(e) => {
                        if self.error_on_config_issue {
                            crate::warn!("{}", cuda_config);
                            crate::bail!("Failed to initialize CUDA devices: {}", e);
                        } else {
                            crate::warn!("{}", cuda_config);
                            crate::warn!("Failed to initialize CUDA devices: {}", e);
                            crate::warn!("Falling back to CPU");
                            self.use_gpu = false;
                        }
                    }
                }
            }
        }
        if !self.use_gpu {
            self.cuda_config = None;
            self.ram_config.initialize(self.error_on_config_issue)?;
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn initialize_mac(&mut self) -> crate::Result<()> {
        if self.use_gpu {
            if self.metal_config.is_none() {
                let metal_config = MetalConfig::default();
                self.metal_config = Some(metal_config);
            }
            if let Some(metal_config) = &mut self.metal_config {
                match metal_config.initialize(self.error_on_config_issue) {
                    Ok(_) => {
                        crate::info!("Successfully initialized: {}", metal_config);
                    }
                    Err(e) => {
                        if self.error_on_config_issue {
                            crate::warn!("{}", metal_config);
                            crate::bail!("Failed to initialize Metal: {}", e);
                        } else {
                            crate::warn!("{}", metal_config);
                            crate::warn!("Failed to initialize Metal: {}", e);
                            crate::warn!("Falling back to CPU");
                            self.use_gpu = false;
                        }
                    }
                }
            }
        }
        if !self.use_gpu {
            self.metal_config = None;
            self.ram_config.initialize(self.error_on_config_issue)?;
        }
        Ok(())
    }

    pub fn available_memory_bytes(&self) -> crate::Result<u64> {
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        if let Some(cuda_config) = &self.cuda_config {
            Ok(cuda_config.total_vram_bytes)
        } else {
            Ok(self.ram_config.use_ram_bytes)
        }
        #[cfg(target_os = "macos")]
        if let Some(metal_config) = &self.metal_config {
            Ok(metal_config.use_ram_bytes)
        } else {
            Ok(self.ram_config.use_ram_bytes)
        }

        #[cfg(not(any(unix, windows, target_os = "macos")))]
        {
            crate::bail!("Unsupported OS");
        }
    }

    pub fn average_layer_size_bytes(&self) -> crate::Result<u64> {
        match self.average_layer_size_bytes {
            Some(size) => Ok(size),
            None => crate::bail!("Average layer size not set"),
        }
    }

    pub fn layer_count(&self) -> crate::Result<u64> {
        match self.layer_count {
            Some(count) => Ok(count),
            None => crate::bail!("Layer count not set"),
        }
    }

    pub fn main_gpu(&self) -> crate::Result<u32> {
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        if let Some(cuda_config) = &self.cuda_config {
            cuda_config.main_gpu(self.error_on_config_issue)
        } else {
            crate::bail!("No GPUs available")
        }
        #[cfg(target_os = "macos")]
        if self.metal_config.is_some() {
            Ok(1)
        } else {
            Ok(0)
        }
        #[cfg(not(any(unix, windows, target_os = "macos")))]
        {
            crate::bail!("Unsupported OS");
        }
    }

    pub fn gpu_count(&self) -> usize {
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        if let Some(cuda_config) = &self.cuda_config {
            cuda_config.device_count()
        } else {
            0
        }
        #[cfg(target_os = "macos")]
        if self.metal_config.is_some() {
            1
        } else {
            0
        }
        #[cfg(not(any(unix, windows, target_os = "macos")))]
        {
            crate::bail!("Unsupported OS");
        }
    }

    pub fn allocate_layers_to_gpus(
        &self,
        buffer_layer_per_gpu: u64,
        buffer_layer_main_gpu: u64,
    ) -> crate::Result<Vec<gpu::GpuDevice>> {
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        let mut gpu_devices: Vec<gpu::GpuDevice> = if let Some(cuda_config) = &self.cuda_config {
            cuda_config.to_generic_gpu_devices(self.error_on_config_issue)?
        } else {
            crate::bail!("No GPUs available")
        };
        #[cfg(target_os = "macos")]
        let mut gpu_devices: Vec<gpu::GpuDevice> = if let Some(metal_config) = &self.metal_config {
            vec![metal_config.to_generic_gpu_device()]
        } else {
            crate::bail!("No GPUs available")
        };
        #[cfg(not(any(unix, windows, target_os = "macos")))]
        {
            crate::bail!("Unsupported OS");
        }
        let allocator = GpuLayerAllocator::new(
            self.average_layer_size_bytes()?,
            self.layer_count()?,
            buffer_layer_per_gpu,
            buffer_layer_main_gpu,
        );
        allocator.allocate(&mut gpu_devices)?;
        Ok(gpu_devices)
    }
}

impl std::fmt::Display for DeviceConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        write!(f, "DeviceConfig:")?;
        crate::i_ln(f, format_args!("{}", self.cpu_config))?;
        crate::i_ln(f, format_args!("{}", self.ram_config))?;
        crate::i_ln(f, format_args!("use_gpu: {}", self.use_gpu))?;

        #[cfg(any(target_os = "linux", target_os = "windows"))]
        if let Some(cuda_config) = &self.cuda_config {
            crate::i_ln(f, format_args!("{}", cuda_config))?;
        }
        #[cfg(target_os = "macos")]
        if let Some(metal_config) = &self.metal_config {
            crate::i_ln(f, format_args!("{}", metal_config))?;
        }
        crate::i_ln(
            f,
            format_args!("error_on_config_issue: {}", self.error_on_config_issue),
        )?;
        if let Some(layer_count) = self.layer_count {
            crate::i_ln(f, format_args!("layer_count: {}", layer_count))?;
        }
        if let Some(average_layer_size_bytes) = self.average_layer_size_bytes {
            crate::i_ln(
                f,
                format_args!("average_layer_size_bytes: {}", average_layer_size_bytes),
            )?;
        }

        Ok(())
    }
}
