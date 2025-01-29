/// Configuration for managing CPU resources in LLM inference workloads.
///
/// This struct handles the detection and allocation of CPU cores for parallel processing tasks.
/// It provides functionality to:
/// - Detect available physical CPU cores (with fallback to logical cores for VMs)
/// - Configure manual thread count overrides
/// - Manage CPU resource utilization through percentage allocation
///
#[derive(Debug, Clone)]
pub struct CpuConfig {
    /// The number of physical CPU cores available on the system.
    /// Falls back to logical core count when running in a VM.
    pub num_cpus: usize,

    /// Optional manual override for the number of threads to use for general processing.
    /// If None, the thread count will be calculated based on `use_percentage`.
    pub threads: Option<i16>,

    /// Optional manual override for the number of threads to use for batch processing.
    /// If None, the thread count will be calculated based on `use_percentage`.
    pub threads_batch: Option<i16>,

    /// The percentage (0.0 to 1.0) of total CPU cores to use when no manual thread
    /// count is specified. This affects both general and batch processing when their
    /// respective thread counts are not manually specified.
    ///
    /// Defaults to 0.70 (70% of available cores).
    pub use_percentage: f32,
}

impl Default for CpuConfig {
    fn default() -> Self {
        let mut sys = sysinfo::System::new_all();
        sys.refresh_all();

        let num_cpus = match sys.physical_core_count() {
            Some(cores) => cores,
            None => sys.cpus().len(), // Fallback to logical core count for VMs
        };

        Self {
            num_cpus,
            threads: None,
            threads_batch: None,
            use_percentage: 0.70,
        }
    }
}

impl CpuConfig {
    pub(crate) fn initialize(&mut self, error_on_config_issue: bool) -> crate::Result<()> {
        if self.use_percentage > 1.0 || self.use_percentage < 0.0 {
            if error_on_config_issue {
                crate::bail!(
                    "Percentage of total CPU cores must be between 0.0 and 1.0. use_percentage: {}",
                    self.use_percentage
                );
            } else {
                crate::warn!("Percentage of total CPU must be between 0.0 and 1.0. use_percentage: {}. Falling back to default value of 0.70", self.use_percentage);
                self.use_percentage = 0.70;
            }
        }
        self.threads = self.check_thread_count(self.threads, error_on_config_issue)?;
        self.threads_batch = self.check_thread_count(self.threads_batch, error_on_config_issue)?;
        Ok(())
    }

    fn check_thread_count(
        &self,
        threads: Option<i16>,
        error_on_config_issue: bool,
    ) -> crate::Result<Option<i16>> {
        if let Some(threads) = threads {
            if threads > self.num_cpus as i16 {
                if error_on_config_issue {
                    crate::bail!(
                        "Requested threads {} is greater than the number of available physical CPU cores {}. Exiting",
                        threads,
                        self.num_cpus
                    );
                } else {
                    crate::warn!(
                    "Requested threads {} is greater than the number of available physical CPU cores {}. Using the number of available physical CPU cores",
                    threads,
                    self.num_cpus
                );
                    Ok(Some(self.num_cpus as i16))
                }
            } else {
                Ok(Some(threads))
            }
        } else {
            Ok(None)
        }
    }

    pub fn thread_count_or_default(&self) -> i16 {
        self.count_or_default(self.threads)
    }

    pub fn thread_count_batch_or_default(&self) -> i16 {
        self.count_or_default(self.threads_batch)
    }

    fn count_or_default(&self, threads: Option<i16>) -> i16 {
        if let Some(threads) = threads {
            threads
        } else {
            (self.num_cpus as f32 * self.use_percentage) as i16
        }
    }
}

impl std::fmt::Display for CpuConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, "CpuConfig:")?;
        crate::i_nlns(
            f,
            &[
                format_args!("num_cpus: {}", self.num_cpus),
                format_args!("threads: {:?}", self.threads),
                format_args!("threads_batch: {:?}", self.threads_batch),
                format_args!("use_percentage: {}", self.use_percentage),
            ],
        )
    }
}
