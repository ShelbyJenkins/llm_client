pub mod config_json;
pub mod general;
pub mod llm;
pub mod tokenizer;
use super::gguf::tools::{gguf_file::GgufFile, gguf_layers::GgufLayers};
use general::GeneralMetadata;
use llm::Architecture;
use tokenizer::TokenizerMetadata;

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
            general: GeneralMetadata::from_gguf(&gguf)?,
            llm: Architecture::from_gguf(&gguf)?,
            tokenizer: TokenizerMetadata::from_gguf(&gguf)?,
            layers: GgufLayers::from_tensors(&gguf.tensors),
            gguf_file: gguf,
        })
    }

    pub fn estimate_model_size(&self) -> crate::Result<u64> {
        Ok(self.gguf_file.size())
    }

    pub fn estimate_context_size(&self, ctx_size: u64, batch_size: Option<u64>) -> u64 {
        self.llm.estimate_context_size(ctx_size, batch_size)
    }

    pub fn average_layer_size_bytes(
        &self,
        ctx_size: u64,
        batch_size: Option<u64>,
    ) -> crate::Result<u64> {
        let total_layers_size = self.layers.total_size_blocks_bytes();
        let block_count = self.layers.count_blocks();
        let context_size = self.estimate_context_size(ctx_size, batch_size);
        let total_size = total_layers_size + context_size;
        Ok(total_size / block_count)
    }

    pub fn context_length(&self) -> u64 {
        self.llm.context_length()
    }
}

impl std::fmt::Debug for LocalLlmMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("LocalLlmMetadata");
        debug_struct.field("GeneralMetadata", &self.general);
        debug_struct.field("Architecture", &self.llm);
        debug_struct.field("TokenizerMetadata", &self.tokenizer);
        debug_struct.finish()
    }
}
