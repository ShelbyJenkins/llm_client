use cpu::CpuConfig;
pub use cuda::CudaDeviceMap;
use gpu::GpuLayerAllocator;
pub mod cpu;
pub mod cuda;
pub mod gpu;

#[derive(Debug, Clone)]
pub struct DeviceConfig {
    pub(crate) cpu_config: CpuConfig,
    /// Whether to use any available GPUs.
    pub use_gpu: bool,
    pub cuda_map: Option<CudaDeviceMap>,
    /// If the model will not fit in VRAM, return an error.
    pub error_on_gpu_offload: bool,
    pub error_on_gpu_error: bool,
    // Model details
    pub layer_count: Option<u64>,
    pub average_layer_size_bytes: Option<u64>,
    pub local_model_path: String,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            cpu_config: CpuConfig::default(),
            use_gpu: true,
            cuda_map: None,
            error_on_gpu_offload: true,
            error_on_gpu_error: true,
            layer_count: None,
            average_layer_size_bytes: None,
            local_model_path: Default::default(),
        }
    }
}

impl DeviceConfig {
    pub(crate) fn initialize(&mut self) -> crate::Result<()> {
        if self.use_gpu {
            if self.cuda_map.is_none() {
                let cuda_map = CudaDeviceMap::default();
                self.cuda_map = Some(cuda_map);
            }
            if let Some(cuda_map) = &mut self.cuda_map {
                cuda_map.error_on_gpu_error = self.error_on_gpu_error;
                match cuda_map.initialize() {
                    Ok(_) => (),
                    Err(e) => {
                        crate::warn!("Failed to populate CUDA devices: {}", e);
                        if self.error_on_gpu_error {
                            crate::bail!("Failed to populate CUDA devices: {}", e);
                        }
                        self.use_gpu = false;
                    }
                }
            }
        }
        self.cpu_config.initialize()?;
        Ok(())
    }

    pub fn available_vram_bytes(&self) -> crate::Result<u64> {
        if let Some(cuda_map) = &self.cuda_map {
            Ok(cuda_map.total_vram_bytes)
        } else {
            crate::bail!("No GPUs available")
        }
    }

    pub fn available_vram_gigabytes(&self) -> crate::Result<u32> {
        Ok(((self.available_vram_bytes()? as f64) / 1_073_741_824.0).floor() as u32)
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
        if let Some(cuda_map) = &self.cuda_map {
            cuda_map.main_gpu()
        } else {
            crate::bail!("No GPUs available")
        }
    }

    pub fn gpu_count(&self) -> usize {
        if let Some(cuda_map) = &self.cuda_map {
            cuda_map.device_count()
        } else {
            0
        }
    }

    pub fn allocate_layers_to_gpus(&self) -> crate::Result<Vec<gpu::GpuDevice>> {
        let mut gpu_devices: Vec<gpu::GpuDevice> = if let Some(cuda_map) = &self.cuda_map {
            cuda_map.to_generic_gpu_devices()?
        } else {
            crate::bail!("No GPUs available")
        };

        let allocator =
            GpuLayerAllocator::new(self.average_layer_size_bytes()?, self.layer_count()?);
        allocator.allocate(&mut gpu_devices)?;
        Ok(gpu_devices)
    }
}
