use llama::LlamaMetadata;

use crate::llm::local::gguf::gguf_file::GgufFile;

pub mod llama;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum Architecture {
    Llama(LlamaMetadata),
}

impl Architecture {
    pub fn from_gguf_file(gguf: &GgufFile) -> crate::Result<Self> {
        Ok(Self::Llama(LlamaMetadata::from_gguf_file(gguf)?))
        // let arch: String = gguf.get_value("general.architecture")?;
        // match arch.as_str() {
        //     "llama" | "phi3" | "qwen2" | "granite" | "stablelm" => {
        //         Ok(Self::Llama(LlamaMetadata::from_gguf(gguf)?))
        //     }
        //     _ => crate::bail!("Unknown architecture: {}", arch),
        // }
    }

    pub fn llama(&self) -> crate::Result<&LlamaMetadata> {
        match self {
            Self::Llama(llama) => Ok(llama),
        }
    }

    pub fn model_ctx_size(&self) -> u64 {
        match self {
            Self::Llama(llama) => llama.context_length,
        }
    }

    pub fn estimate_context_size_bytes(
        &self,
        ctx_size: Option<u64>,
        batch_size: Option<u64>,
    ) -> u64 {
        match self {
            Self::Llama(llama) => llama.estimate_context_size_bytes(ctx_size, batch_size),
        }
    }
}
