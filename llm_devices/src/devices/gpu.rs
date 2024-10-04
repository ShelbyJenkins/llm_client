#[derive(Debug, Default)]
pub struct GpuDevice {
    pub ordinal: u32,
    pub available_vram_bytes: u64,
    pub allocated_layer_bytes: u64,
    pub allocated_buffer_bytes: u64,
    pub allocated_layers: u64,
    pub is_main_gpu: bool,
}

impl GpuDevice {
    fn can_allocate(&self, layer_size: u64) -> bool {
        self.available_vram_bytes >= self.allocated_layer_bytes + layer_size
    }

    fn allocate_layer(&mut self, layer_size: u64) {
        self.allocated_layers += 1;
        self.allocated_layer_bytes += layer_size;
    }
}

pub struct GpuLayerAllocator {
    layer_size: u64,
    total_layers: u64,
    buffer_layer_per_gpu: u64,
    buffer_layer_main_gpu: u64,
}

impl GpuLayerAllocator {
    pub fn new(
        layer_size: u64,
        total_layers: u64,
        buffer_layer_per_gpu: u64,
        buffer_layer_main_gpu: u64,
    ) -> Self {
        GpuLayerAllocator {
            layer_size,
            total_layers,
            buffer_layer_per_gpu,
            buffer_layer_main_gpu,
        }
    }

    pub fn allocate(&self, gpus: &mut [GpuDevice]) -> crate::Result<()> {
        // Sort GPUs by available VRAM, descending
        gpus.sort_by_key(|gpu| std::cmp::Reverse(gpu.available_vram_bytes));

        // Calculate total available VRAM
        let total_available_vram: u64 = gpus.iter().map(|gpu| gpu.available_vram_bytes).sum();

        let mut buffer_layers = 0;
        // Allocate buffer layers
        for gpu in gpus.iter_mut() {
            for _ in 1..=self.buffer_layer_per_gpu {
                buffer_layers += 1;
                gpu.allocate_layer(self.layer_size);
                gpu.allocated_buffer_bytes += self.layer_size;
            }
            if gpu.is_main_gpu {
                for _ in 1..=self.buffer_layer_main_gpu {
                    buffer_layers += 1;
                    gpu.allocate_layer(self.layer_size);
                    gpu.allocated_buffer_bytes += self.layer_size;
                }
            }
        }

        let total_required_vram = (self.total_layers + buffer_layers) * self.layer_size;

        // Check if there's enough total VRAM
        if total_available_vram < total_required_vram {
            crate::bail!(
                "Insufficient total VRAM. Required: {}GB, Available: {}GB",
                total_required_vram / 1_073_741_824,
                total_available_vram / 1_073_741_824
            );
        }

        let mut allocation = vec![0; gpus.len()];
        let result = self.dfs_allocate(gpus, &mut allocation, 0, self.total_layers);
        Self::print_allocation(gpus);
        if !result {
            // Check why allocation failed
            let allocated_layers: u64 = gpus.iter().map(|gpu| gpu.allocated_layers).sum();
            let remaining_layers = self.total_layers - (allocated_layers - buffer_layers);

            if remaining_layers > 0 {
                crate::bail!(
                    "Failed to allocate all layers. {} layers remaining unallocated.",
                    remaining_layers
                );
            } else {
                crate::bail!("Allocation failed due to VRAM fragmentation across GPUs.");
            }
        }
        Ok(())
    }

    fn dfs_allocate(
        &self,
        gpus: &mut [GpuDevice],
        allocation: &mut Vec<u64>,
        gpu_index: usize,
        remaining_layers: u64,
    ) -> bool {
        if remaining_layers == 0 {
            return true;
        }

        // Try to allocate to each GPU in a round-robin fashion
        for i in 0..gpus.len() {
            let current_gpu_index = (gpu_index + i) % gpus.len();
            if gpus[current_gpu_index].can_allocate(self.layer_size) {
                gpus[current_gpu_index].allocate_layer(self.layer_size);
                allocation[current_gpu_index] += 1;

                if self.dfs_allocate(
                    gpus,
                    allocation,
                    (current_gpu_index + 1) % gpus.len(),
                    remaining_layers - 1,
                ) {
                    return true;
                }

                // If allocation failed, backtrack
                gpus[current_gpu_index].allocated_layers -= 1;
                gpus[current_gpu_index].allocated_layer_bytes -= self.layer_size;
                allocation[current_gpu_index] -= 1;
            }
        }

        false
    }

    fn print_allocation(gpus: &[GpuDevice]) {
        let message = std::fmt::format(format_args!(
            "\nGPU Allocation:\n{}",
            gpus.iter()
                .map(|gpu| format!("{}", gpu))
                .collect::<Vec<_>>()
                .join("\n")
        ));
        crate::info!("{}", message);
    }
}

impl std::fmt::Display for GpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        crate::i_nlns(
            f,
            &[
                format_args!("Ordinal: {}", self.ordinal),
                format_args!("Main GPU: {}", self.is_main_gpu),
                format_args!("Allocated Layers: {}", self.allocated_layers),
                format_args!(
                    "Layer Size: {:.2} GB",
                    self.allocated_layer_bytes as f64 / 1_073_741_824.0
                ),
                format_args!(
                    "Buffer Size: {:.2} GB",
                    self.allocated_buffer_bytes as f64 / 1_073_741_824.0
                ),
                format_args!(
                    "Total Size: {:.2} GB",
                    (self.allocated_layer_bytes + self.allocated_buffer_bytes) as f64
                        / 1_073_741_824.0
                ),
                format_args!(
                    "Available VRAM: {:.2} GB",
                    self.available_vram_bytes as f64 / 1_073_741_824.0
                ),
            ],
        )
    }
}
