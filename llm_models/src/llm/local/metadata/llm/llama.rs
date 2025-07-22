use crate::llm::{
    base::DEFAULT_CONTEXT_LENGTH,
    local::{gguf::gguf_file::GgufFile, metadata::memory::estimate_context_size_bytes},
};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct LlamaMetadata {
    // Required fields
    pub context_length: u64,
    pub embedding_length: u64,
    pub block_count: u64,

    // Optional fields
    pub feed_forward_length: Option<u64>,
    pub vocab_size: Option<u32>,
    pub tensor_data_layout: Option<String>,
    pub expert_count: Option<u32>,
    pub expert_used_count: Option<u32>,

    // Other potentially useful fields from the previous LlmMetadata
    pub use_parallel_residual: Option<bool>,
    pub attention: AttentionMetadata,
    pub rope: RopeMetadata,
    pub ssm: SsmMetadata,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct AttentionMetadata {
    pub head_count: u64,
    pub head_count_kv: Option<u64>,
    pub max_alibi_bias: Option<f32>,
    pub clamp_kqv: Option<f32>,
    pub layer_norm_epsilon: Option<f32>,
    pub layer_norm_rms_epsilon: Option<f32>,
    pub key_length: Option<u32>,
    pub value_length: Option<u32>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct RopeMetadata {
    pub dimension_count: Option<u64>,
    pub freq_base: Option<f32>,
    pub scale: Option<f32>,
    pub scaling: RopeScalingMetadata,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct RopeScalingMetadata {
    pub r#type: Option<String>,
    pub factor: Option<f32>,
    pub original_context_length: Option<u32>,
    pub finetuned: Option<bool>,
    pub scale_linear: Option<f32>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct SsmMetadata {
    pub conv_kernel: Option<u32>,
    pub inner_size: Option<u32>,
    pub state_size: Option<u32>,
    pub time_step_rank: Option<u32>,
}

impl LlamaMetadata {
    pub fn from_gguf_file(gguf: &GgufFile) -> crate::Result<Self> {
        let arch: String = gguf.get_value("general.architecture")?;
        let path_prefixes: &[&str] = &[&arch];
        Ok(Self {
            context_length: gguf
                .get_pathed_value(path_prefixes, "context_length")
                .unwrap_or(DEFAULT_CONTEXT_LENGTH),
            embedding_length: gguf.get_pathed_value(path_prefixes, "embedding_length")?,
            block_count: gguf.get_pathed_value(path_prefixes, "block_count")?,
            feed_forward_length: gguf
                .get_pathed_value(path_prefixes, "feed_forward_length")
                .unwrap_or(None),
            vocab_size: gguf.get_pathed_value(path_prefixes, "vocab_size")?,
            tensor_data_layout: gguf.get_pathed_value(path_prefixes, "tensor_data_layout")?,
            expert_count: gguf.get_pathed_value(path_prefixes, "expert_count")?,
            expert_used_count: gguf.get_pathed_value(path_prefixes, "expert_used_count")?,
            use_parallel_residual: gguf.get_pathed_value(path_prefixes, "use_parallel_residual")?,
            attention: AttentionMetadata::from_gguf_file(gguf, path_prefixes)?,
            rope: RopeMetadata::from_gguf_file(gguf, path_prefixes)?,
            ssm: SsmMetadata::from_gguf_file(gguf, path_prefixes)?,
        })
    }

    // This is converted from https://github.com/pandora-s-git/LLMVRAMCalculator/blob/main/LLMVRAMCalculator/LLMVRAMCalculator.py
    pub fn estimate_context_size_bytes(
        &self,
        ctx_size: Option<u64>,
        batch_size: Option<u64>,
    ) -> u64 {
        estimate_context_size_bytes(
            ctx_size,
            self.embedding_length,
            self.attention.head_count,
            self.attention
                .head_count_kv
                .unwrap_or(self.attention.head_count),
            self.block_count,
            batch_size,
        )
    }
}

impl AttentionMetadata {
    pub fn from_gguf_file(gguf: &GgufFile, path_prefixes: &[&str]) -> crate::Result<Self> {
        Ok(Self {
            head_count: gguf.get_pathed_value(path_prefixes, "attention.head_count")?,
            head_count_kv: gguf.get_pathed_value(path_prefixes, "attention.head_count_kv")?,
            max_alibi_bias: gguf.get_pathed_value(path_prefixes, "attention.max_alibi_bias")?,
            clamp_kqv: gguf.get_pathed_value(path_prefixes, "attention.clamp_kqv")?,
            layer_norm_epsilon: gguf
                .get_pathed_value(path_prefixes, "attention.layer_norm_epsilon")?,
            layer_norm_rms_epsilon: gguf
                .get_pathed_value(path_prefixes, "attention.layer_norm_rms_epsilon")?,
            key_length: gguf.get_pathed_value(path_prefixes, "attention.key_length")?,
            value_length: gguf.get_pathed_value(path_prefixes, "attention.value_length")?,
        })
    }
}

impl RopeMetadata {
    pub fn from_gguf_file(gguf: &GgufFile, path_prefixes: &[&str]) -> crate::Result<Self> {
        Ok(Self {
            dimension_count: gguf.get_pathed_value(path_prefixes, "rope.dimension_count")?,
            freq_base: gguf.get_pathed_value(path_prefixes, "rope.freq_base")?,
            scale: gguf.get_pathed_value(path_prefixes, "rope.scale")?,
            scaling: RopeScalingMetadata::from_gguf_file(gguf, path_prefixes)?,
        })
    }
}

impl RopeScalingMetadata {
    pub fn from_gguf_file(gguf: &GgufFile, path_prefixes: &[&str]) -> crate::Result<Self> {
        Ok(Self {
            r#type: gguf.get_pathed_value(path_prefixes, "rope.scaling.type")?,
            factor: gguf.get_pathed_value(path_prefixes, "rope.scaling.factor")?,
            original_context_length: gguf
                .get_pathed_value(path_prefixes, "rope.scaling.original_context_length")?,
            finetuned: gguf.get_pathed_value(path_prefixes, "rope.scaling.finetuned")?,
            scale_linear: gguf.get_pathed_value(path_prefixes, "rope.scaling.scale_linear")?,
        })
    }
}

impl SsmMetadata {
    pub fn from_gguf_file(gguf: &GgufFile, path_prefixes: &[&str]) -> crate::Result<Self> {
        Ok(Self {
            conv_kernel: gguf.get_pathed_value(path_prefixes, "ssm.conv_kernel")?,
            inner_size: gguf.get_pathed_value(path_prefixes, "ssm.inner_size")?,
            state_size: gguf.get_pathed_value(path_prefixes, "ssm.state_size")?,
            time_step_rank: gguf.get_pathed_value(path_prefixes, "ssm.time_step_rank")?,
        })
    }
}
