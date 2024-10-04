use sysinfo;

#[derive(Debug, Clone)]
pub struct RamConfig {
    pub total_ram_bytes: u64,
    pub available_ram_bytes: u64,
    pub used_ram_bytes: u64,
    pub use_ram_bytes: u64,
    pub use_percentage: f32,
}

impl Default for RamConfig {
    fn default() -> Self {
        let mut sys = sysinfo::System::new_all();
        sys.refresh_all();
        Self {
            total_ram_bytes: sys.total_memory(),
            available_ram_bytes: sys.available_memory(),
            used_ram_bytes: sys.used_memory(),
            use_ram_bytes: 0,
            use_percentage: 0.70,
        }
    }
}

impl RamConfig {
    pub(crate) fn initialize(&mut self, error_on_config_issue: bool) -> crate::Result<()> {
        if self.use_ram_bytes == 0 {
            self.use_ram_bytes = self.percentage_of_total(error_on_config_issue)?;
        } else if self.use_ram_bytes >= self.likely_ram_bytes() {
            if error_on_config_issue {
                crate::bail!(
                    "use_ram_bytes {:.2} is greater than the available system RAM {:.2}",
                    (self.use_ram_bytes as f64) / 1_073_741_824.0,
                    (self.likely_ram_bytes() as f64) / 1_073_741_824.0
                );
            } else {
                crate::warn!(
                    "use_ram_bytes {:.2} is greater than the available system RAM {:.2}. Falling back to percentage of total RAM",
                    (self.use_ram_bytes as f64) / 1_073_741_824.0,
                    (self.likely_ram_bytes() as f64) / 1_073_741_824.0
                );
                self.use_ram_bytes = self.percentage_of_total(error_on_config_issue)?;
            }
        }
        Ok(())
    }

    pub(crate) fn likely_ram_bytes(&self) -> u64 {
        std::cmp::min(
            self.total_ram_bytes - self.used_ram_bytes,
            self.available_ram_bytes,
        )
    }

    fn percentage_of_total(&mut self, error_on_config_issue: bool) -> crate::Result<u64> {
        if self.use_percentage > 1.0 || self.use_percentage < 0.0 {
            if error_on_config_issue {
                crate::bail!(
                    "Percentage of total RAM must be between 0.0 and 1.0. use_percentage: {:.2}",
                    self.use_percentage
                );
            } else {
                crate::warn!("Percentage of total RAM must be between 0.0 and 1.0. use_percentage: {}. Falling back to default value of 0.70", self.use_percentage);
                self.use_percentage = 0.70;
            }
        }

        Ok((self.likely_ram_bytes() as f32 * self.use_percentage) as u64)
    }
}

impl std::fmt::Display for RamConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "RamConfig:")?;
        crate::i_nlns(
            f,
            &[
                format_args!(
                    "Total system RAM: {:.2} GB",
                    (self.total_ram_bytes as f64) / 1_073_741_824.0
                ),
                format_args!(
                    "Available system RAM: {:.2} GB",
                    (self.available_ram_bytes as f64) / 1_073_741_824.0
                ),
                format_args!(
                    "Specified RAM for Inference: {:.2} GB",
                    (self.use_ram_bytes as f64) / 1_073_741_824.0
                ),
            ],
        )
    }
}
