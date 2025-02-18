#[derive(Clone)]
pub struct TokenizerMetadata {
    pub ggml: Option<GgmlTokenizerMetadata>,
    pub huggingface_json: Option<String>,
    pub rwkv_world: Option<String>,
    pub chat_template: Option<String>,
}

#[derive(Debug, Clone)]
pub struct GgmlTokenizerMetadata {
    pub model: GgmlTokenizerModel,
    pub tokens: Vec<String>,
    pub scores: Option<Vec<f32>>,
    pub merges: Option<Vec<String>>,
    pub added_tokens: Option<Vec<String>>,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub unknown_token_id: Option<u32>,
    pub separator_token_id: Option<u32>,
    pub padding_token_id: Option<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GgmlTokenizerModel {
    Llama,
    Replit,
    Gpt2,
    Rwkv,
}

impl GgmlTokenizerModel {
    pub fn from_str(s: &str) -> crate::Result<Self> {
        Ok(match s {
            "llama" => Self::Llama,
            "replit" => Self::Replit,
            "gpt2" => Self::Gpt2,
            "rwkv" => Self::Rwkv,
            _ => crate::bail!("Unknown GGML tokenizer model: {}", s),
        })
    }

    pub fn to_str(&self) -> &str {
        match self {
            Self::Llama => "llama",
            Self::Replit => "replit",
            Self::Gpt2 => "gpt2",
            Self::Rwkv => "rwkv",
        }
    }
}

impl TokenizerMetadata {
    pub fn from_gguf(
        gguf: &crate::local_models::gguf::tools::gguf_file::GgufFile,
    ) -> crate::Result<Self> {
        if gguf
            .get_value::<Option<String>>("tokenizer.ggml.model")?
            .is_some()
        {
            return Ok(Self {
                ggml: Some(GgmlTokenizerMetadata::from_gguf(gguf)?),
                huggingface_json: gguf.get_value("tokenizer.huggingface.json")?,
                rwkv_world: gguf.get_value("tokenizer.rwkv_world")?,
                chat_template: gguf.get_value("tokenizer.chat_template")?,
            });
        }
        Ok(Self {
            ggml: None,
            huggingface_json: gguf.get_value("tokenizer.huggingface.json")?,
            rwkv_world: gguf.get_value("tokenizer.rwkv_world")?,
            chat_template: gguf.get_value("tokenizer.chat_template")?,
        })
    }
}

impl GgmlTokenizerMetadata {
    pub fn from_gguf(
        gguf: &crate::local_models::gguf::tools::gguf_file::GgufFile,
    ) -> crate::Result<Self> {
        let model_string: String = gguf.get_value("tokenizer.ggml.model")?;

        Ok(Self {
            model: GgmlTokenizerModel::from_str(&model_string)?,
            tokens: gguf.get_value("tokenizer.ggml.tokens")?,
            scores: gguf.get_value("tokenizer.ggml.scores")?,
            merges: gguf.get_value("tokenizer.ggml.merges")?,
            added_tokens: gguf.get_value("tokenizer.ggml.added_tokens")?,
            bos_token_id: gguf.get_value("tokenizer.ggml.bos_token_id")?,
            eos_token_id: gguf.get_value("tokenizer.ggml.eos_token_id")?,
            unknown_token_id: gguf.get_value("tokenizer.ggml.unknown_token_id")?,
            separator_token_id: gguf.get_value("tokenizer.ggml.separator_token_id")?,
            padding_token_id: gguf.get_value("tokenizer.ggml.padding_token_id")?,
        })
    }
}

impl std::fmt::Debug for TokenizerMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("TokenizerMetadata");
        if let Some(ggml) = &self.ggml {
            debug_struct.field("GgmlTokenizerModel", &ggml.model.to_str());
            debug_struct.field("bos_token_id", &ggml.bos_token_id);
            debug_struct.field("eos_token_id", &ggml.eos_token_id);
            if let Some(unknown_token_id) = ggml.unknown_token_id {
                debug_struct.field("unknown_token_id", &unknown_token_id);
            }
            if let Some(separator_token_id) = ggml.separator_token_id {
                debug_struct.field("separator_token_id", &separator_token_id);
            }
        }
        debug_struct.finish()
    }
}
