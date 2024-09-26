#[derive(Debug, Clone)]
pub(crate) struct CpuConfig {
    pub num_cpus: usize,
    pub system_ram_bytes: u64,
    pub use_ram_bytes: Option<u64>,
    pub threads: Option<i16>,
    pub threads_batch: Option<i16>,
}

impl Default for CpuConfig {
    fn default() -> Self {
        let mut sys = sysinfo::System::new_all();
        sys.refresh_all();

        let num_cpus = match sys.physical_core_count() {
            Some(cores) => cores,
            None => sys.cpus().len(), // Fallback to logical core count for VMs
        };
        let system_ram_bytes = std::cmp::min(
            sys.total_memory() - sys.used_memory(),
            sys.available_memory(),
        );

        Self {
            num_cpus,
            system_ram_bytes,
            use_ram_bytes: None,
            threads: None,
            threads_batch: None,
        }
    }
}

impl CpuConfig {
    pub(crate) fn initialize(&mut self) -> crate::Result<()> {
        self.threads = self.check_thread_count(self.threads);
        self.threads_batch = self.check_thread_count(self.threads_batch);
        if let Some(use_ram_bytes) = self.use_ram_bytes {
            if use_ram_bytes > self.system_ram_bytes {
                crate::warn!(
                    "Requested RAM {} is greater than the available system RAM {}. Using the available system RAM",
                    use_ram_bytes,
                    self.system_ram_bytes
                );
                self.use_ram_bytes = Some(self.system_ram_bytes);
            }
        }
        Ok(())
    }

    pub(crate) fn check_thread_count(&self, threads: Option<i16>) -> Option<i16> {
        if let Some(threads) = threads {
            if threads > self.num_cpus as i16 {
                crate::warn!(
                    "Requested threads {} is greater than the number of available physical CPU cores {}. Using the number of available physical CPU cores",
                    threads,
                    self.num_cpus
                );
                Some(self.num_cpus as i16)
            } else {
                Some(threads)
            }
        } else {
            None
        }
    }

    pub(crate) fn set_default_thread_count(&self, threads: Option<i16>, ratio: f32) -> i16 {
        let target = (self.num_cpus as f32 / ratio) as i16;
        let threads = if let Some(threads) = threads {
            if threads > self.num_cpus as i16 {
                crate::warn!(
                    "Requested threads {} is greater than the number of available physical CPU cores {}. Using the number of available physical CPU cores",
                    threads,
                    self.num_cpus
                );
                self.num_cpus as i16
            } else {
                threads
            }
        } else {
            self.num_cpus as i16
        };
        let threads = if threads > target { target } else { threads };
        crate::trace!("Setting thread count to {}", threads);
        threads
    }

    pub(crate) fn set_memory_utilization_for_inference(
        &self,
        percentage_of_total: f32,
    ) -> crate::Result<u64> {
        if percentage_of_total > 1.0 || percentage_of_total < 0.0 {
            crate::bail!("Percentage of total RAM must be between 0.0 and 1.0");
        }
        let use_ram_bytes = if let Some(use_ram_bytes) = self.use_ram_bytes {
            if use_ram_bytes > self.system_ram_bytes {
                crate::warn!(
                    "Requested RAM {} is greater than the available system RAM {}. Using the available system RAM",
                    use_ram_bytes,
                    self.system_ram_bytes
                );
                self.system_ram_bytes
            } else {
                return Ok(use_ram_bytes); // No changes, so lets use what the user set.
            }
        } else {
            self.system_ram_bytes
        };
        Ok((use_ram_bytes as f32 * percentage_of_total) as u64)
    }
}
