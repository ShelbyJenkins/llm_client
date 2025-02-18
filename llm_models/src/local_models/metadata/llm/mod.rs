use llama::LlamaMetadata;
pub mod llama;

pub const DEFAULT_CONTEXT_LENGTH: usize = 8192;

#[derive(Debug)]
pub enum Architecture {
    Llama(LlamaMetadata),
}

impl Architecture {
    pub fn from_gguf(
        gguf: &crate::local_models::gguf::tools::gguf_file::GgufFile,
    ) -> crate::Result<Self> {
        Ok(Self::Llama(LlamaMetadata::from_gguf(gguf)?))
        // let arch: String = gguf.get_value("general.architecture")?;
        // match arch.as_str() {
        //     "llama" | "phi3" | "qwen2" | "granite" | "stablelm" => {
        //         Ok(Self::Llama(LlamaMetadata::from_gguf(gguf)?))
        //     }
        //     _ => crate::bail!("Unknown architecture: {}", arch),
        // }
    }

    pub fn estimate_context_size(&self, ctx_size: usize, batch_size: Option<usize>) -> usize {
        match self {
            Self::Llama(llama) => llama.estimate_context_size(ctx_size, batch_size),
        }
    }

    pub fn llama(&self) -> crate::Result<&LlamaMetadata> {
        match self {
            Self::Llama(llama) => Ok(llama),
        }
    }

    pub fn context_length(&self) -> usize {
        match self {
            Self::Llama(llama) => llama.context_length,
        }
    }
}
