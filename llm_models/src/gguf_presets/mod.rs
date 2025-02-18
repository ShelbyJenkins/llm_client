// Public modules
pub mod loader;
pub mod models;
pub mod organizations;

// Internal imports
use crate::{
    local_models::{
        gguf::memory::{
            estimate_context_size, estimate_model_size_level, estimate_quantization_level,
        },
        hf_loader::HuggingFaceFileCacheStatus,
    },
    GgufLoader, LocalLlmModel,
};
use loader::DEFAULT_PRESET_CONTEXT_LENGTH;

// Public exports
pub use loader::GgufPresetLoader;
pub use models::GgufPresetTrait;
pub use organizations::LocalLlmOrganization;

const PATH_TO_GGUF_PRESETS_DIR: std::sync::LazyLock<std::path::PathBuf> =
    std::sync::LazyLock::new(|| {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("gguf_presets")
    });

#[derive(Debug, Clone)]
pub struct GgufPreset {
    pub organization: LocalLlmOrganization,
    pub model_id: &'static str,
    pub friendly_name: &'static str,
    pub model_repo_id: &'static str,
    pub gguf_repo_id: &'static str,
    pub number_of_parameters: f64,
    pub model_ctx_size: usize,
    pub tokenizer_path: Option<&'static str>,
    pub config: ConfigJson,
    pub quants: &'static [GgufPresetQuant],
}

impl Default for GgufPreset {
    fn default() -> Self {
        Self::LLAMA_3_2_3B_INSTRUCT
    }
}

impl GgufPreset {
    pub fn load(self) -> Result<LocalLlmModel, crate::Error> {
        let mut loader = GgufLoader {
            gguf_preset_loader: GgufPresetLoader {
                llm_preset: self,
                ..Default::default()
            },
            ..Default::default()
        };
        loader.load()
    }

    pub fn get_quants(&self, ctx_size: Option<usize>) -> Vec<GgufQuant> {
        let ctx_size = ctx_size.unwrap_or(DEFAULT_PRESET_CONTEXT_LENGTH);
        let ctx_memory_size_bytes = estimate_context_size(
            ctx_size,
            self.config.embedding_length,
            self.config.head_count,
            self.config.head_count_kv.unwrap(),
            self.config.block_count,
            None,
        );

        let mut quants = Vec::new();
        for quant in self.quants {
            let estimated_memory_usage_bytes =
                estimate_model_size_level(self.number_of_parameters(), quant.q_lvl) as usize
                    + ctx_memory_size_bytes;

            let hf_model_cach_status =
                HuggingFaceFileCacheStatus::new(&quant.fname, self.gguf_repo_id, quant.total_bytes)
                    .unwrap();

            quants.push(GgufQuant {
                q_lvl: quant.q_lvl,
                file_name: quant.fname.to_string(),
                downloaded: hf_model_cach_status.available,
                on_disk_file_size_bytes: hf_model_cach_status.on_disk_file_size_bytes,
                total_file_size_bytes: quant.total_bytes,
                estimated_memory_usage_bytes,
            });
        }
        quants
    }

    pub fn select_quant_for_available_memory(
        &self,
        ctx_size: Option<usize>,
        available_ram_bytes: u64,
    ) -> Result<u8, crate::Error> {
        let ctx_size = ctx_size.unwrap_or(DEFAULT_PRESET_CONTEXT_LENGTH);
        let ctx_memory_size_bytes = estimate_context_size(
            ctx_size,
            self.config.embedding_length,
            self.config.head_count,
            self.config.head_count_kv.unwrap(),
            self.config.block_count,
            None,
        );
        let initial_q_bits = estimate_quantization_level(
            self.number_of_parameters(),
            available_ram_bytes as usize,
            ctx_memory_size_bytes,
        )?;
        let mut q_bits = initial_q_bits;
        loop {
            if let Some(_) = self.quant_file_name_for_q_bit(q_bits) {
                return Ok(q_bits);
            }
            if q_bits == 1 {
                crate::bail!(
                    "No model file found from quantization levels: {initial_q_bits}-{q_bits}",
                );
            } else {
                q_bits -= 1;
            }
        }
    }

    pub fn quant_file_name_for_q_bit(&self, q_bits: u8) -> Option<&'static str> {
        self.quants
            .iter()
            .find(|quant| quant.q_lvl == q_bits)
            .map(|quant| quant.fname)
    }

    pub fn tokenizer_path(&self) -> Option<std::path::PathBuf> {
        if let Some(tokenizer_path) = self.tokenizer_path {
            Some(
                PATH_TO_GGUF_PRESETS_DIR
                    .join("tokenizers")
                    .join(tokenizer_path),
            )
        } else {
            None
        }
    }

    pub fn number_of_parameters(&self) -> f64 {
        self.number_of_parameters * 1_000_000_000.0 as f64
    }
}

#[derive(Debug, Clone)]
pub struct GgufPresetQuant {
    pub q_lvl: u8,
    pub fname: &'static str,
    pub total_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct GgufQuant {
    pub q_lvl: u8,
    pub file_name: String,
    pub downloaded: bool,
    pub on_disk_file_size_bytes: Option<usize>,
    pub total_file_size_bytes: usize,
    pub estimated_memory_usage_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct ConfigJson {
    pub context_length: usize,
    pub embedding_length: usize,
    pub feed_forward_length: Option<usize>,
    pub head_count: usize,
    pub head_count_kv: Option<usize>,
    pub block_count: usize,
    pub torch_dtype: &'static str,
    pub vocab_size: usize,
    pub architecture: &'static str,
    pub model_size_bytes: Option<usize>,
}
