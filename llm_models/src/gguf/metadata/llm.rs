use ggus::{GGuf, GGufMetaMapExt};
use serde::{Deserialize, Serialize};

/// Represents the model's architecture name as given in GGUF metadata
/// (`general.architecture`). The variant names correspond to known
/// architectures (all lowercase in metadata), for example:
/// - `Llama` for "llama" (Meta LLaMA series),
/// - `Mpt` for "mpt" (Mosaic MPT),
/// - `GptNeoX` for "gptneox" (EleutherAI GPT-NeoX),
/// - `GptJ` for "gptj",
/// - `Gpt2` for "gpt2",
/// - `Bloom` for "bloom",
/// - `Falcon` for "falcon" (TII Falcon),
/// - `Mamba` for "mamba" (state-space model architecture),
/// - `Rwkv` for "rwkv",
/// - `Other(String)` for any unrecognized architecture (stored as lowercase string).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[serde(untagged)]
#[non_exhaustive]
pub enum Architecture {
    Llama,
    Mpt,
    GptNeoX,
    GptJ,
    Gpt2,
    Bloom,
    Falcon,
    Mamba,
    Rwkv,
    /// Any value not (yet) in the spec.
    Other(String),
}

impl Architecture {
    /// Create an `Architecture` enum from a string (case-insensitive match to known architectures).
    pub fn from_str(raw: &str) -> Self {
        match raw.to_ascii_lowercase().as_str() {
            "llama" => Self::Llama,
            "mpt" => Self::Mpt,
            "gptneox" => Self::GptNeoX,
            "gptj" => Self::GptJ,
            "gpt2" => Self::Gpt2,
            "bloom" => Self::Bloom,
            "falcon" => Self::Falcon,
            "mamba" => Self::Mamba,
            "rwkv" => Self::Rwkv,
            other => Self::Other(other.to_owned()),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Llama => "llama",
            Self::Mpt => "mpt",
            Self::GptNeoX => "gptneox",
            Self::GptJ => "gptj",
            Self::Gpt2 => "gpt2",
            Self::Bloom => "bloom",
            Self::Falcon => "falcon",
            Self::Mamba => "mamba",
            Self::Rwkv => "rwkv",
            Self::Other(s) => s,
        }
    }
}

impl core::fmt::Display for Architecture {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Metadata for an LLM's overall architecture hyperparameters, as stored in GGUF.
/// Each field corresponds to a `[architecture].[field]` key in the metadata:
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Llm {
    /// The maximum context length (in tokens) that the model was trained on.
    /// This is typically the model's attention window size.
    /// **Also known as** `n_ctx`.
    pub context_length: Option<u64>,

    /// The size of the embedding layer (often equal to the model's hidden dimension).
    /// **Also known as** `n_embd` (number of embedding features).
    pub embedding_length: Option<u64>,

    /// The number of transformer blocks (layers) in the model's architecture (count of attention+FFN layers).
    /// Does not include input/token embedding or output layers.
    /// This is essentially the number of decoder layers in the transformer (**also referred to as** `n_layer` in many contexts).
    pub block_count: Option<u64>,

    /// The dimension of the feed-forward network hidden layer in each transformer block.
    /// In many architectures this is 4× the embedding size (if using an MLP expansion).
    /// **Also known as** `n_ff`.
    pub feed_forward_length: Option<u64>,

    /// The layout of tensor data in the model file. This indicates if tensors were permuted or transformed for efficiency during GGUF conversion.
    /// If not present, it defaults to `"reference"` (meaning the same ordering as the original model).
    pub tensor_data_layout: Option<String>,

    /// Number of experts in Mixture-of-Experts (MoE) models.
    /// This is the total count of expert sub-models (usually per MoE layer).
    /// Only present for architectures that use MoE (optional for others).
    /// *Alternative terminology:* sometimes called `num_experts`.
    pub expert_count: Option<u32>,

    /// Number of experts used per token inference in MoE models (i.e. how many experts are selected by the gating mechanism for each token).
    /// For example, 1 means each token is routed to the single best expert (Top-1 gating), 2 means Top-2 experts, etc.
    /// Only present for MoE architectures.
    /// *Also known as* the MoE top-k experts per token (e.g., Top-1 or Top-2).
    pub expert_used_count: Option<u32>,

    /// Whether the model uses **parallel residual** connections in its transformer blocks.
    /// If `true`, the attention output and feed-forward output are combined in parallel (with input) instead of sequentially.
    /// This is a feature of the GPT-NeoX architecture (EleutherAI GPT-NeoX-20B and variants).
    /// *In configuration files, this is often the `parallel_residual` setting.*
    pub use_parallel_residual: Option<bool>,

    /// Nested struct containing all attention-related metadata (number of heads, etc).
    pub attention: Attention,

    /// Nested struct for RoPE (Rotary Positional Embedding) settings.
    pub rope: Rope,

    /// Nested struct for SSM (State Space Model) settings – used only by architectures that incorporate state-space layers (e.g. Mamba).
    pub ssm: Ssm,
}

impl Llm {
    pub fn new(gguf: &GGuf) -> Self {
        Self {
            context_length: gguf.llm_context_length().ok().map(|v| v as u64),
            embedding_length: gguf.llm_embedding_length().ok().map(|v| v as u64),
            block_count: gguf.llm_block_count().ok().map(|v| v as u64),
            feed_forward_length: gguf.llm_feed_forward_length().ok().map(|v| v as u64),
            tensor_data_layout: gguf.llm_tensor_data_layout().ok().map(str::to_owned),
            expert_count: gguf.llm_expert_count().ok().map(|v| v as u32),
            expert_used_count: gguf.llm_expert_used_count().ok().map(|v| v as u32),
            use_parallel_residual: gguf.llm_use_parallel_residual().ok(),
            attention: Attention::new(gguf),
            rope: Rope::new(gguf),
            ssm: Ssm::new(gguf),
        }
    }
}

/// Metadata for the model's attention mechanism hyperparameters.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Attention {
    /// Number of attention heads in each transformer block.
    /// **Also known as** `n_head`.
    pub head_count: Option<u64>,

    /// Number of key/value heads in each transformer block, used for **Grouped-Query Attention (GQA)**.
    /// This effectively is the number of independent key/value sets.
    /// If this is present and less than `head_count`, the model uses GQA (heads are divided into groups sharing keys/values).
    /// If not present or equal to `head_count`, the model does not use GQA (each head has its own key and value).
    /// *In some literature, this may be referred to as the "number of KV groups" or `num_key_value_heads`.*
    pub head_count_kv: Option<u64>,

    /// The maximum ALiBi bias value used in the attention mechanism.
    /// ALiBi (Attention Linear Bias) is an alternate position bias method.
    /// This parameter typically appears in models like MPT.
    /// **Alternative name:** `alibi_bias_max` (as it is called in MPT metadata).
    pub max_alibi_bias: Option<f32>,

    /// Clamp value `C` for attention Q, K, V magnitudes. If set, the model clamps the values of the Q, K, V vectors to the range [-C, C] to stabilize training.
    /// Used in certain architectures such as MPT for gradient stability.
    /// **Alternative name:** `clip_kqv` (as it appears in MPT metadata).
    pub clamp_kqv: Option<f32>,

    /// Epsilon value for **layer normalization** in the attention sublayers.
    /// This is the small constant added to variance to avoid division-by-zero in LayerNorm.
    /// (Typical values are 1e-5 or 1e-6.)
    pub layer_norm_epsilon: Option<f32>,

    /// Epsilon value for **RMS normalization** (Root Mean Square LayerNorm) in the attention sublayers.
    /// This is used by architectures that employ RMSNorm instead of standard LayerNorm (e.g. LLaMA uses RMSNorm).
    /// *Often referred to as* `rms_norm_eps` in LLaMA-related code/configs.
    pub layer_norm_rms_epsilon: Option<f32>,

    /// The dimension of each attention **key** vector ($d_k$).
    /// If not specified in the metadata, it is assumed to be `embedding_length / head_count` (i.e., each head's key has the default dimension).
    pub key_length: Option<u32>,

    /// The dimension of each attention **value** vector ($d_v$).
    /// If not specified, it defaults to `embedding_length / head_count` (same as keys, for standard multi-head attention).
    pub value_length: Option<u32>,
}
impl Attention {
    pub fn new(gguf: &GGuf) -> Self {
        Self {
            head_count: gguf.llm_attention_head_count().ok().map(|v| v as u64),
            head_count_kv: gguf.llm_attention_head_count_kv().ok().map(|v| v as u64),
            max_alibi_bias: gguf.llm_attention_max_alibi_bias().ok(),
            clamp_kqv: gguf.llm_attention_clamp_kqv().ok(),
            layer_norm_epsilon: gguf.llm_attention_layer_norm_epsilon().ok(),
            layer_norm_rms_epsilon: gguf.llm_attention_layer_norm_rms_epsilon().ok(),
            key_length: gguf.llm_attention_key_length().ok().map(|v| v as u32),
            value_length: gguf.llm_attention_value_length().ok().map(|v| v as u32),
        }
    }
}

/// Metadata for RoPE (Rotary Position Embedding) parameters.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Rope {
    /// Number of dimensions (out of the model's embedding) that use Rotary Position Embedding.
    /// This is sometimes the full embedding size, or a fraction of it (e.g., half) depending on the model.
    /// *Also known as* the rotary dimension (often `rotary_dim` in some model implementations).
    pub dimension_count: Option<usize>,

    /// Base frequency for RoPE rotations. This defines the geometric progression of rotation frequencies.
    /// For example, LLaMA uses a base of 10000.0 for its rotary embeddings.
    pub freq_base: Option<f32>,

    /// RoPE scaling factor for extended context (if the model uses RoPE scaling).
    /// This field, if present, typically indicates a **linear** scaling of RoPE to allow a longer context window than the base model.
    /// (For instance, a value of 2.0 might imply doubling the effective context length.)
    /// **Note:** Newer models use the more detailed `rope.scaling.*` fields instead of this. This field corresponds to the older `[llm].rope.scale` key.
    pub scale: Option<f32>,

    /// Linear RoPE scaling factor from older GGUF versions or models.
    /// This is equivalent to `scale` above, provided for backward compatibility when a single linear scaling factor was used (key `[llm].rope.scale_linear`).
    /// New models should use the structured `rope.scaling` fields (captured in the `RopeScaling` struct).
    pub scale_linear: Option<f32>,

    /// Nested struct detailing the RoPE scaling method and parameters, if any.
    pub scaling: RopeScaling,
}

impl Rope {
    pub fn new(gguf: &GGuf) -> Self {
        let architecture = gguf.general_architecture().ok().map(str::to_owned);
        // Some older models used an architecture-specific rope.scale key
        let scale = if let Some(arch) = &architecture {
            gguf.get_f32(&format!("{arch}.rope.scale")).ok()
        } else {
            None
        };
        Self {
            dimension_count: gguf.llm_rope_dimension_count().ok(),
            freq_base: gguf.llm_rope_freq_base().ok(),
            scale,
            scale_linear: gguf.llm_rope_scale_linear().ok(),
            scaling: RopeScaling::new(gguf),
        }
    }
}

/// Describes the model's RoPE scaling strategy for extended context windows.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RopeScaling {
    /// The type of RoPE scaling used.
    /// Known values: `"none"` (no scaling, standard RoPE), `"linear"` (NTK Linear scaling), or `"yarn"` (YARN scaling).
    pub r#type: Option<String>,

    /// The RoPE scaling factor. Interpretation depends on the `type`:
    /// for `"linear"` scaling, this factor linearly expands the context (e.g., 2.0 for 2x context length);
    /// for `"yarn"`, it may be used as a multiplier in a different way (Yarn uses log-space scaling).
    pub factor: Option<f32>,

    /// The original context length that the base model was trained on, before applying scaling.
    /// Used as a reference in scaling formulas (for example, linear scaling might target a new context length = factor * original_length).
    pub original_context_length: Option<u32>,

    /// Indicates if the model was fine-tuned with the RoPE scaling in effect (`true`), or if the scaling is only being applied during inference on a model fine-tuned at the original context length (`false`).
    pub finetuned: Option<bool>,
}

impl RopeScaling {
    pub fn new(gguf: &GGuf) -> Self {
        Self {
            r#type: gguf.llm_rope_scaling_type().ok().map(str::to_owned),
            factor: gguf.llm_rope_scaling_factor().ok(),
            original_context_length: gguf
                .llm_rope_scaling_original_context_length()
                .ok()
                .map(|v| v as u32),
            finetuned: gguf.llm_rope_scaling_finetuned().ok(),
        }
    }
}

/// Metadata for State Space Model (SSM) layer parameters, used by architectures like Mamba.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Ssm {
    /// Size of the convolution kernel in the SSM layer (for the state update).
    /// This might correspond to how many past tokens the SSM state spans (sometimes called the "rolling state" length).
    pub conv_kernel: Option<u32>,

    /// The inner embedding size of the SSM states.
    /// This can be thought of as the dimension of the state vectors before projection to the output.
    pub inner_size: Option<u32>,

    /// The size of the recurrent state in the SSM.
    /// In other words, the dimensionality of the state vector that is carried across timesteps.
    pub state_size: Option<u32>,

    /// The rank of the time-step projection in the SSM.
    /// This might relate to how the time evolution is parameterized (a higher rank could allow more complex time dynamics).
    pub time_step_rank: Option<u32>,
}

impl Ssm {
    pub fn new(gguf: &GGuf) -> Self {
        Self {
            conv_kernel: gguf.llm_ssm_conv_kernel().ok().map(|v| v as u32),
            inner_size: gguf.llm_ssm_inner_size().ok().map(|v| v as u32),
            state_size: gguf.llm_ssm_state_size().ok().map(|v| v as u32),
            time_step_rank: gguf.llm_ssm_time_step_rank().ok().map(|v| v as u32),
        }
    }
}
