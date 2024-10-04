use objc2::rc::Retained;
use objc2_metal::{MTLCopyAllDevices, MTLDevice};

use super::gpu::GpuDevice;

#[derive(Debug, Clone)]
pub struct MetalConfig {
    pub max_working_set_size: u64,
    pub available_ram_bytes: u64,
    pub allocated_ram_bytes: u64,
    pub use_ram_bytes: u64,
    pub use_percentage: f32,
}

impl Default for MetalConfig {
    fn default() -> Self {
        Self {
            max_working_set_size: 0,
            available_ram_bytes: 0,
            allocated_ram_bytes: 0,
            use_ram_bytes: 0,
            use_percentage: 0.90,
        }
    }
}

impl MetalConfig {
    pub fn new_from_ram_gb(use_ram_gb: f32) -> Self {
        Self {
            use_ram_bytes: (use_ram_gb * 1_073_741_824.0) as u64,
            ..Default::default()
        }
    }
    pub fn new_from_percentage(use_percentage: f32) -> Self {
        Self {
            use_percentage,
            ..Default::default()
        }
    }

    pub(crate) fn initialize(&mut self, error_on_config_issue: bool) -> crate::Result<()> {
        self.initialize_metal_device()?;
        if self.use_ram_bytes == 0 {
            self.use_ram_bytes = self.percentage_of_total(error_on_config_issue)?;
        } else if self.use_ram_bytes >= self.available_ram_bytes {
            if error_on_config_issue {
                crate::bail!(
                    "use_ram_bytes {:.2} is greater than the available system RAM {:.2}",
                    (self.use_ram_bytes as f64) / 1_073_741_824.0,
                    (self.available_ram_bytes as f64) / 1_073_741_824.0
                );
            } else {
                crate::warn!(
                    "use_ram_bytes {:.2} is greater than the available system RAM {:.2}. Falling back to percentage of total RAM",
                    (self.use_ram_bytes as f64) / 1_073_741_824.0,
                    (self.available_ram_bytes as f64) / 1_073_741_824.0
                );
                self.use_ram_bytes = self.percentage_of_total(error_on_config_issue)?;
            }
        }
        Ok(())
    }

    fn initialize_metal_device(&mut self) -> crate::Result<()> {
        let devices = unsafe {
            match Retained::from_raw(MTLCopyAllDevices().as_ptr()) {
                Some(devices) => devices,
                None => {
                    crate::bail!("No Metal devices found");
                }
            }
        };

        let device = match devices.first() {
            Some(device) => device,
            None => {
                crate::bail!("No Metal devices found");
            }
        };
        let has_unified_memory = device.hasUnifiedMemory();
        match has_unified_memory {
            true => has_unified_memory,
            false => {
                crate::bail!("No has_unified_memory found");
            }
        };

        self.max_working_set_size = device.recommendedMaxWorkingSetSize();
        self.allocated_ram_bytes = device.currentAllocatedSize() as u64;
        self.available_ram_bytes = self.max_working_set_size - self.allocated_ram_bytes;
        Ok(())
    }

    fn percentage_of_total(&mut self, error_on_config_issue: bool) -> crate::Result<u64> {
        if self.use_percentage > 1.0 || self.use_percentage < 0.0 {
            if error_on_config_issue {
                crate::bail!(
                    "Percentage of total RAM must be between 0.0 and 1.0. use_percentage: {}",
                    self.use_percentage
                );
            } else {
                crate::warn!("Percentage of total RAM must be between 0.0 and 1.0. use_percentage: {}. Falling back to default value of 0.90", self.use_percentage);
                self.use_percentage = 0.90;
            }
        }

        Ok((self.available_ram_bytes as f32 * self.use_percentage) as u64)
    }

    pub(crate) fn to_generic_gpu_device(&self) -> GpuDevice {
        GpuDevice {
            ordinal: 0,
            available_vram_bytes: self.use_ram_bytes,
            allocated_bytes: 0,
            allocated_buffer_bytes: 0,
            allocated_layers: 0,
            is_main_gpu: true,
        }
    }
}

impl std::fmt::Display for MetalConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, "MetalConfig:")?;
        crate::i_nlns(
            f,
            &[
                format_args!(
                    "max_working_set_size: {:.2} GB",
                    (self.max_working_set_size as f64) / 1_073_741_824.0
                ),
                format_args!(
                    "available_ram_bytes: {:.2} GB",
                    (self.available_ram_bytes as f64) / 1_073_741_824.0
                ),
                format_args!(
                    "allocated_ram_bytes: {:.2} GB",
                    (self.allocated_ram_bytes as f64) / 1_073_741_824.0
                ),
                format_args!(
                    "Specified RAM for Inference: {:.2} GB",
                    (self.use_ram_bytes as f64) / 1_073_741_824.0
                ),
            ],
        )
    }
}
