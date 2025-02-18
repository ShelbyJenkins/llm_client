use super::gpu::GpuDevice;
use nvml_wrapper::Nvml;

// See https://gist.github.com/jrruethe/8974d2c8b4ece242a071d1a1526aa763#file-vram-rb-L64
/// CUDA overhead in bytes (500MB) used for VRAM calculations
pub const CUDA_OVERHEAD: usize = 500 * 1024 * 1024;

/// Configuration for NVIDIA CUDA devices on Linux and Windows platforms.
///
/// Manages device detection, VRAM allocation, and multi-GPU configurations.
#[derive(Debug, Clone)]
pub struct CudaConfig {
    /// Main GPU device ordinal. If None, selects GPU with most VRAM
    pub main_gpu: Option<u32>,

    /// List of CUDA device ordinals to use. If empty, uses all available devices
    pub use_cuda_devices: Vec<u32>,

    /// Information about detected CUDA devices
    pub(crate) cuda_devices: Vec<CudaDevice>,

    /// Total available VRAM across all devices in bytes
    pub(crate) total_vram_bytes: usize,
}

impl Default for CudaConfig {
    fn default() -> Self {
        Self {
            main_gpu: None,
            use_cuda_devices: Vec::new(),
            cuda_devices: Vec::new(),
            total_vram_bytes: 0,
        }
    }
}

impl CudaConfig {
    /// Creates a new CudaConfig using specified device ordinals.
    ///
    /// # Arguments
    ///
    /// * `use_cuda_devices` - Vector of CUDA device ordinals to use
    pub fn new_from_cuda_devices(use_cuda_devices: Vec<u32>) -> Self {
        Self {
            use_cuda_devices,
            ..Default::default()
        }
    }

    /// Creates a new CudaConfig with a specified main device.
    ///
    /// # Arguments
    ///
    /// * `use_cuda_devices` - Vector of CUDA device ordinals to use
    /// * `main_gpu` - Ordinal of the device to use as main GPU
    pub fn new_with_main_device(use_cuda_devices: Vec<u32>, main_gpu: u32) -> Self {
        Self {
            main_gpu: Some(main_gpu),
            use_cuda_devices,
            ..Default::default()
        }
    }

    pub(crate) fn initialize(&mut self, error_on_config_issue: bool) -> crate::Result<()> {
        let nvml: Nvml = init_nvml_wrapper()?;
        if self.use_cuda_devices.is_empty() {
            self.cuda_devices = get_all_cuda_devices(Some(&nvml))?;
        } else {
            for ordinal in &self.use_cuda_devices {
                match CudaDevice::new(*ordinal, Some(&nvml)) {
                    Ok(cuda_device) => self.cuda_devices.push(cuda_device),
                    Err(e) => {
                        if error_on_config_issue {
                            crate::bail!(
                                "Failed to get device {} specified in cuda_devices: {}",
                                ordinal,
                                e
                            );
                        } else {
                            crate::warn!(
                                "Failed to get device {} specified in cuda_devices: {}",
                                ordinal,
                                e
                            );
                        }
                    }
                }
            }
        }
        if self.cuda_devices.is_empty() {
            crate::bail!("No CUDA devices found");
        }

        self.main_gpu = Some(self.main_gpu(error_on_config_issue));

        self.total_vram_bytes = self
            .cuda_devices
            .iter()
            .map(|d| (d.available_vram_bytes))
            .sum();
        Ok(())
    }

    pub(crate) fn device_count(&self) -> usize {
        self.cuda_devices.len()
    }

    pub(crate) fn main_gpu(&self, error_on_config_issue: bool) -> u32 {
        if let Some(main_gpu) = self.main_gpu {
            for device in &self.cuda_devices {
                if device.ordinal == main_gpu {
                    return main_gpu;
                }
            }
            if error_on_config_issue {
                panic!(
                    "Main GPU set by user {} not found in CUDA devices",
                    main_gpu
                );
            } else {
                crate::warn!(
                    "Main GPU set by user {} not found in CUDA devices. Using largest VRAM device.",
                    main_gpu
                );
            }
        };
        let main_gpu = self
            .cuda_devices
            .iter()
            .max_by_key(|d| d.available_vram_bytes)
            .expect("No devices found when setting main gpu")
            .ordinal;
        for device in &self.cuda_devices {
            if device.ordinal == main_gpu {
                return main_gpu;
            }
        }
        panic!("Main GPU {} not found in CUDA devices", main_gpu);
    }

    pub(crate) fn to_generic_gpu_devices(
        &self,
        error_on_config_issue: bool,
    ) -> crate::Result<Vec<GpuDevice>> {
        let mut gpu_devices: Vec<GpuDevice> = self
            .cuda_devices
            .iter()
            .map(|d| d.to_generic_gpu())
            .collect();
        let main_gpu = self.main_gpu(error_on_config_issue);
        for gpu in &mut gpu_devices {
            if gpu.ordinal == main_gpu {
                gpu.is_main_gpu = true;
            }
        }
        Ok(gpu_devices)
    }
}

pub fn get_all_cuda_devices(nvml: Option<&Nvml>) -> crate::Result<Vec<CudaDevice>> {
    let nvml = match nvml {
        Some(nvml) => nvml,
        None => &init_nvml_wrapper()?,
    };
    let device_count = nvml.device_count()?;
    let mut cuda_devices: Vec<CudaDevice> = Vec::new();
    let mut ordinal = 0;
    while cuda_devices.len() < device_count as usize {
        if let Ok(nvml_device) = CudaDevice::new(ordinal, Some(&nvml)) {
            cuda_devices.push(nvml_device);
        }
        if ordinal > 100 {
            crate::warn!(
                "nvml_wrapper reported {device_count} devices, but we were only able to get {}",
                cuda_devices.len()
            );
        }
        ordinal += 1;
    }
    if cuda_devices.len() == 0 {
        crate::bail!("No CUDA devices found");
    }
    Ok(cuda_devices)
}

/// Represents a single CUDA device with its capabilities.
#[derive(Debug, Clone)]
pub struct CudaDevice {
    /// Device ordinal (GPU index)
    pub ordinal: u32,

    /// Available VRAM in bytes (total VRAM minus CUDA_OVERHEAD)
    pub available_vram_bytes: usize,

    /// Device name (e.g., "NVIDIA GeForce RTX 3080")
    pub name: Option<String>,

    /// Power limit in watts
    pub power_limit: Option<u32>,

    /// CUDA compute capability major version
    pub driver_major: Option<i32>,

    /// CUDA compute capability minor version
    pub driver_minor: Option<i32>,
}

impl CudaDevice {
    /// Creates a new CudaDevice from a device ordinal.
    ///
    /// # Arguments
    ///
    /// * `ordinal` - CUDA device ordinal to initialize
    /// * `nvml` - Optional NVML instance for hardware queries
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Device cannot be queried
    /// - Device has 0 bytes of VRAM
    pub fn new(ordinal: u32, nvml: Option<&Nvml>) -> crate::Result<Self> {
        let nvml = match nvml {
            Some(nvml) => nvml,
            None => &init_nvml_wrapper()?,
        };
        if let Ok(nvml_device) = nvml.device_by_index(ordinal) {
            if let Ok(memory_info) = nvml_device.memory_info() {
                if memory_info.total != 0 {
                    let name = if let Ok(name) = nvml_device.name() {
                        Some(name)
                    } else {
                        None
                    };
                    let power_limit = if let Ok(power_limit) = nvml_device.enforced_power_limit() {
                        Some(power_limit)
                    } else {
                        None
                    };
                    let (driver_major, driver_minor) = if let Ok(cuda_compute_capability) =
                        nvml_device.cuda_compute_capability()
                    {
                        (
                            Some(cuda_compute_capability.major),
                            Some(cuda_compute_capability.minor),
                        )
                    } else {
                        (None, None)
                    };
                    let cuda_device = CudaDevice {
                        ordinal: ordinal,
                        available_vram_bytes: memory_info.total as usize - CUDA_OVERHEAD,
                        name,
                        power_limit,
                        driver_major,
                        driver_minor,
                    };

                    Ok(cuda_device)
                } else {
                    crate::bail!("Device {} has 0 bytes of VRAM. Skipping device.", ordinal);
                }
            } else {
                crate::bail!("Failed to get device {}", ordinal);
            }
        } else {
            crate::bail!("Failed to get device {}", ordinal);
        }
    }

    pub fn to_generic_gpu(&self) -> GpuDevice {
        GpuDevice {
            ordinal: self.ordinal,
            available_vram_bytes: self.available_vram_bytes,
            ..Default::default()
        }
    }
}

/// Initializes NVIDIA Management Library (NVML).
///
/// Attempts to load the NVML library from common paths on Linux, WSL, and Windows.
///
/// # Errors
///
/// Returns error if NVML initialization fails on all attempted paths.
pub fn init_nvml_wrapper() -> crate::Result<Nvml> {
    let library_names = vec![
        "libnvidia-ml.so",   // For Linux
        "libnvidia-ml.so.1", // For WSL
        "nvml.dll",          // For Windows
    ];
    for library_name in library_names {
        match Nvml::builder().lib_path(library_name.as_ref()).init() {
            Ok(nvml) => return Ok(nvml),
            Err(_) => {
                continue;
            }
        }
    }
    crate::bail!("Failed to initialize nvml_wrapper::Nvml")
}

impl std::fmt::Display for CudaConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, "CudaConfig:")?;
        crate::i_nlns(
            f,
            &[
                format_args!("Main GPU: {:?}", self.main_gpu),
                format_args!(
                    "Total vram size: {:.2} GB",
                    (self.total_vram_bytes as f64) / 1_073_741_824.0
                ),
            ],
        )?;
        for device in &self.cuda_devices {
            crate::i_ln(f, format_args!("{}", device))?;
        }
        Ok(())
    }
}

impl std::fmt::Display for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "CudaDevice:")?;
        crate::i_nlns(
            f,
            &[
                format_args!("Ordinal: {:?}", self.ordinal),
                format_args!(
                    "Available VRAM: {:.2} GB",
                    (self.available_vram_bytes as f64) / 1_073_741_824.0
                ),
                format_args!("Name: {:?}", self.name),
                format_args!("Power limit: {:?}", self.power_limit),
                format_args!(
                    "Driver version: {}.{}",
                    self.driver_major.unwrap_or(-1),
                    self.driver_minor.unwrap_or(-1)
                ),
            ],
        )
    }
}
