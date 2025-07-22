// Public modules
pub mod general;
pub mod llm;
pub mod memory;
pub mod tokenizer;

// Internal imports
use general::GeneralMetadata;
use llm::Architecture;
use tokenizer::TokenizerMetadata;

use super::gguf::{gguf_file::GgufFile, gguf_layers::GgufLayers};

#[derive(Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct LocalLlmMetadata {
    pub general: GeneralMetadata,
    pub llm: Architecture,
    pub tokenizer: TokenizerMetadata,
    pub gguf_file: GgufFile,
    pub layers: GgufLayers,
}

impl LocalLlmMetadata {
    pub fn from_gguf_path(path: &std::path::Path) -> crate::Result<Self> {
        let mut reader = std::fs::File::open(path)?;
        let gguf: GgufFile = GgufFile::read(&mut reader)?;

        Ok(Self {
            general: GeneralMetadata::from_gguf_file(&gguf)?,
            llm: Architecture::from_gguf_file(&gguf)?,
            tokenizer: TokenizerMetadata::from_gguf_file(&gguf)?,
            layers: GgufLayers::from_tensors(&gguf.tensors),
            gguf_file: gguf,
        })
    }

    pub fn estimate_model_parameters(&self) -> u64 {
        let mut number_of_parameters = 0;
        for tensor in &self.gguf_file.tensors {
            number_of_parameters += tensor.parameters() as u64;
        }
        number_of_parameters
    }

    pub fn estimate_context_size_bytes(
        &self,
        ctx_size: Option<u64>,
        batch_size: Option<u64>,
    ) -> u64 {
        self.llm.estimate_context_size_bytes(ctx_size, batch_size)
    }

    pub fn estimate_model_memory_usage_bytes(
        &self,
        ctx_size: Option<u64>,
        batch_size: Option<u64>,
    ) -> u64 {
        let ctx_memory_size_bytes = self.estimate_context_size_bytes(ctx_size, batch_size);

        let estimated_memory_usage_bytes = memory::estimate_model_size_dtype(
            self.estimate_model_parameters(),
            &self.general.file_type.to_ggml_d_type(),
        );

        (estimated_memory_usage_bytes + ctx_memory_size_bytes as f64) as u64
    }

    pub fn average_layer_size_bytes(
        &self,
        ctx_size: Option<u64>,
        batch_size: Option<u64>,
    ) -> crate::Result<u64> {
        let total_layers_size = self.layers.total_size_blocks_bytes() as u64;
        let block_count = self.layers.count_blocks() as u64;
        let context_size = self.estimate_context_size_bytes(ctx_size, batch_size);
        let total_size = total_layers_size + context_size;
        Ok(total_size / block_count)
    }

    pub fn model_ctx_size(&self) -> u64 {
        self.llm.model_ctx_size()
    }
}

impl std::fmt::Debug for LocalLlmMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "LocalLlmMetadata:")?;
        writeln!(f, "   GeneralMetadata: {:?}", self.general)?;
        writeln!(f, "   Architecture: {:?}", self.llm)?;

        Ok(())
    }
}
