use crate::local_model::gguf::memory::estimate_context_size;
use serde::Deserialize;
use std::{fs::File, io::BufReader};

#[derive(Debug, Deserialize, Clone)]
pub struct ConfigJson {
    #[serde(alias = "max_position_embeddings")]
    #[serde(alias = "n_ctx")]
    pub context_length: u64,
    #[serde(alias = "hidden_size")]
    #[serde(alias = "n_embd")]
    pub embedding_length: u64,
    #[serde(alias = "intermediate_size")]
    #[serde(alias = "n_inner")]
    pub feed_forward_length: Option<u64>,
    #[serde(alias = "num_attention_heads")]
    #[serde(alias = "n_head")]
    pub head_count: u64,
    #[serde(alias = "num_key_value_heads")]
    pub head_count_kv: Option<u64>,
    #[serde(alias = "num_hidden_layers")]
    #[serde(alias = "n_layers")]
    #[serde(alias = "n_layer")]
    #[serde(alias = "num_layers")]
    pub block_count: u64,
    pub torch_dtype: String,
    pub vocab_size: u32,
    #[serde(alias = "model_type")]
    pub architecture: String,
    pub model_size_bytes: Option<u64>,
}

impl ConfigJson {
    pub fn from_local_path(config_json_path: &std::path::PathBuf) -> crate::Result<Self> {
        let file = File::open(config_json_path)?;
        let reader = BufReader::new(file);
        let config: ConfigJson = serde_json::from_reader(reader)?;
        Ok(config)
    }

    // This is converted from https://github.com/pandora-s-git/LLMVRAMCalculator/blob/main/LLMVRAMCalculator/LLMVRAMCalculator.py
    pub fn estimate_context_size(&self, ctx_size: u64) -> u64 {
        estimate_context_size(
            ctx_size,
            self.embedding_length,
            self.head_count,
            self.head_count_kv.unwrap(),
            self.block_count,
            None,
        )
    }
}
