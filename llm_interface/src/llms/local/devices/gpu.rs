pub struct GpuDevice {
    pub(crate) ordinal: u32,
    pub(crate) available_vram_bytes: u64,
    pub(crate) allocated_bytes: u64,
    pub(crate) allocated_buffer_bytes: u64,
    pub(crate) allocated_layers: u64,
    pub(crate) is_main_gpu: bool,
}

impl GpuDevice {
    fn can_allocate(&self, layer_size: u64) -> bool {
        self.available_vram_bytes >= self.allocated_bytes + layer_size
    }

    fn allocate_layer(&mut self, layer_size: u64) {
        self.allocated_layers += 1;
        self.allocated_bytes += layer_size;
    }
}

pub struct GpuLayerAllocator {
    layer_size: u64,
    total_layers: u64,
}

impl GpuLayerAllocator {
    pub fn new(layer_size: u64, total_layers: u64) -> Self {
        GpuLayerAllocator {
            layer_size,
            total_layers,
        }
    }

    pub fn allocate(&self, gpus: &mut [GpuDevice]) -> crate::Result<()> {
        // Sort GPUs by available VRAM, descending
        gpus.sort_by_key(|gpu| std::cmp::Reverse(gpu.available_vram_bytes));

        // Calculate total available VRAM
        let total_available_vram: u64 = gpus.iter().map(|gpu| gpu.available_vram_bytes).sum();

        // Calculate total required VRAM
        let buffer_layers = gpus.len() as u64 + 1; // One for each GPU plus an extra for the main GPU
        let total_required_vram = (self.total_layers + buffer_layers) * self.layer_size;

        // Check if there's enough total VRAM
        if total_available_vram < total_required_vram {
            return Err(anyhow::anyhow!(
                "Insufficient total VRAM. Required: {}GB, Available: {}GB",
                total_required_vram / 1_073_741_824,
                total_available_vram / 1_073_741_824
            ));
        }

        // Allocate buffer layers
        for gpu in gpus.iter_mut() {
            gpu.allocate_layer(self.layer_size);
            gpu.allocated_buffer_bytes = self.layer_size;
            if gpu.is_main_gpu {
                gpu.allocate_layer(self.layer_size); // Extra buffer for main GPU
                gpu.allocated_buffer_bytes += self.layer_size;
            }
        }

        let mut allocation = vec![0; gpus.len()];
        let result = self.dfs_allocate(gpus, &mut allocation, 0, self.total_layers);
        self.print_allocation(gpus);
        if !result {
            // Check why allocation failed
            let allocated_layers: u64 = gpus.iter().map(|gpu| gpu.allocated_layers).sum();
            let remaining_layers = self.total_layers - (allocated_layers - buffer_layers);

            if remaining_layers > 0 {
                return Err(anyhow::anyhow!(
                    "Failed to allocate all layers. {} layers remaining unallocated.",
                    remaining_layers
                ));
            } else {
                return Err(anyhow::anyhow!(
                    "Allocation failed due to VRAM fragmentation across GPUs."
                ));
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
                gpus[current_gpu_index].allocated_bytes -= self.layer_size;
                allocation[current_gpu_index] -= 1;
            }
        }

        false
    }

    fn print_allocation(&self, gpus: &[GpuDevice]) {
        for gpu in gpus.iter() {
            crate::trace!(
                "Ordinal {} | Main GPU: {:5} | Layers: {:3} | Allocated: {:5.2}GB | Buffer: {:5.2}GB | Available: {:5.2}GB",
                gpu.ordinal,
                gpu.is_main_gpu,
                gpu.allocated_layers,
                gpu.allocated_bytes as f64 / 1_073_741_824.0,
                gpu.allocated_buffer_bytes as f64 / 1_073_741_824.0,
                gpu.available_vram_bytes as f64 / 1_073_741_824.0
            );
        }
    }
}
