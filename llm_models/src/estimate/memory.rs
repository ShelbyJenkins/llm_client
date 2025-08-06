use bon::Builder;
use serde::{Deserialize, Serialize};

use crate::manifest::file_encoding_type::GgmlFileType;

#[derive(Builder, Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeMemorySpec {
    /// Maximum context length in tokens to use with the model.
    pub inference_ctx_size: Option<u64>,

    /// Logical maximum batch size (number of tokens processed in one forward pass).
    /// This affects how many tokens are evaluated simultaneously. A higher batch size
    /// can improve throughput but uses more memory (default 2048).  
    #[builder(default = 2048)]
    pub batch_size: u64,

    /// Bits per KV element (`bits_kv`). *Default = 16 (fp16)*.
    #[builder(default = GgmlFileType::F16)]
    pub kv_cache_type: GgmlFileType,

    /// Per‑device descriptions (order is irrelevant).
    #[builder(default)]
    pub device_specs: Vec<DeviceTypeSpec>,

    /// If **true** the KV‑cache is sharded once across GPUs;  
    /// if **false** each GPU holds a full copy. *Default = false*.
    #[builder(default)]
    pub shard_kv: bool,
}

impl Default for RuntimeMemorySpec {
    fn default() -> Self {
        RuntimeMemorySpec::builder().build()
    }
}

impl RuntimeMemorySpec {
    pub fn available_memory_bytes_compute(&self) -> u64 {
        self.device_specs
            .iter()
            .filter_map(|d| match d {
                DeviceTypeSpec::Compute(spec) => Some(spec.available_memory_bytes),
                _ => None,
            })
            .sum()
    }

    pub fn available_memory_bytes_moe_offload(&self) -> u64 {
        self.device_specs
            .iter()
            .filter_map(|d| match d {
                DeviceTypeSpec::MoeOffload(spec) => Some(spec.available_memory_bytes),
                _ => None,
            })
            .sum()
    }
}
/// Classifies a device:
/// * `Compute`  – regular GPU that runs attention / FFN kernels.
/// * `MoeOffload` – high‑memory device (GPU ‑or‑ CPU pinned mem) that
///                   only stores *expert weights*; no scratch nor KV.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceTypeSpec {
    Compute(DeviceMemSpec),
    MoeOffload(DeviceMemSpec),
}

/// One physical accelerator in the host.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceMemSpec {
    pub available_memory_bytes: u64,
    /// If `None` we fall back to the heuristic for scratch size.
    pub compute_buffer_bytes: Option<u64>,
}
