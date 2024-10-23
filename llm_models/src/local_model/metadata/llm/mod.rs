use llama::LlamaMetadata;
pub mod llama;

pub const DEFAULT_CONTEXT_LENGTH: u64 = 8192;

#[derive(Debug)]
pub enum Architecture {
    Llama(LlamaMetadata),
}

impl Architecture {
    pub fn from_gguf(
        gguf: &crate::local_model::gguf::tools::gguf_file::GgufFile,
    ) -> crate::Result<Self> {
        let arch: String = gguf.get_value("general.architecture")?;
        match arch.as_str() {
            "llama" | "phi3" | "qwen2" | "granite" | "stablelm" => {
                Ok(Self::Llama(LlamaMetadata::from_gguf(gguf)?))
            }
            _ => crate::bail!("Unknown architecture: {}", arch),
        }
    }

    pub fn estimate_context_size(&self, ctx_size: u64, batch_size: Option<u64>) -> u64 {
        match self {
            Self::Llama(llama) => llama.estimate_context_size(ctx_size, batch_size),
        }
    }

    pub fn llama(&self) -> crate::Result<&LlamaMetadata> {
        match self {
            Self::Llama(llama) => Ok(llama),
        }
    }

    pub fn context_length(&self) -> u64 {
        match self {
            Self::Llama(llama) => llama.context_length,
        }
    }
}
