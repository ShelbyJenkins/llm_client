use std::{collections::HashMap, num::NonZeroU32};

use serde::{Deserialize, Serialize};

use crate::{
    fs::file_status::{CheckpointFiles, SourceLocator},
    manifest::{
        file_encoding_type::GgmlFileType,
        profile::{CheckpointCounts, ProfileError},
        profile_from_hf::HfProfileError,
    },
};

#[derive(Debug, thiserror::Error)]
pub enum ManifestModelError {
    // Duplicate checkpoint for single-file models.
    #[error("duplicate checkpoint for model '{model}': {checkpoint}")]
    DuplicateCheckpoint { model: String, checkpoint: String },

    #[error("duplicate shard index(es) for model '{model}': {duplicates:?}")]
    DuplicateShards {
        model: String,
        duplicates: Vec<NonZeroU32>,
    },

    // Incoming shard id has a 'total' that doesn't match the initial one.
    #[error("mismatched shard total for model '{model}': expected {expected}, found {found}")]
    MismatchedShardTotal {
        model: String,
        expected: NonZeroU32,
        found: NonZeroU32,
    },

    #[error(
        "missing shard index(es) for model '{model}': expected total {expected_total}, \
         but absent: {missing:?}"
    )]
    MissingShardIndexes {
        model: String,
        expected_total: u32,
        missing: Vec<NonZeroU32>,
    },

    #[error(transparent)]
    Profile(#[from] ProfileError),

    #[error(transparent)]
    HfModel(#[from] HfProfileError), // Could this be removed and HfModel wraps this error instead?
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ManifestModelFormat {
    Gguf,
    SafeTensors,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelManifest {
    pub format: ManifestModelFormat,
    pub base_name: String,
    pub source_locator: SourceLocator,
    pub checkpoints: HashMap<String /* model_name */, CheckpointManifest>,

    /// **Total number of trainable parameters** in the original
    /// (unquantised) model, *including* embeddings and layer norms.
    ///
    /// *Unit*: **parameters**
    /// *Example*: 7 574 000 000 for "Llama‑2‑7B".
    pub n_params: u64,

    /// **Number of transformer blocks** (decoder layers).
    ///
    /// Determined for GGUF by counting tensor prefixes such as
    /// `"blk.{i}.attn_q.weight"`.
    ///
    /// Aliases: `n_layer`.
    ///
    /// *Unit*: **layers**
    pub block_count: u64,

    /// **Maximum context length** (tokens) seen during training.
    ///
    /// Aliases: `max_position_embeddings`, `context_length`, or `n_ctx`.
    ///
    /// *Unit*: **tokens**
    pub model_ctx_size: Option<u64>,

    /// **Embedding / hidden dimension** of the model.
    ///
    /// Aliases: `hidden_size`, `embedding_length`.
    ///
    /// *Unit*: **elements**
    pub n_embd: Option<u64>,

    /// **Number of attention heads** per transformer layer.
    ///
    /// Aliases: `head_count`, `num_attention_heads`.
    ///
    /// `head_dim = n_embd / n_head`.
    pub n_head: Option<u64>,

    /// **Number of key/value heads** (KV groups).
    ///
    /// * If `Some(x)` **and `x < n_head`** → Grouped‑Query / Multi‑Query
    ///   Attention is used.
    /// * If `None` or equal to `n_head` → standard attention.
    ///
    /// Aliases: `head_count_kv`, `num_key_value_heads`.
    pub n_head_kv: Option<u64>,

    pub expert_count: Option<u64>,

    /// Router *top-k*: how many experts are selected **per token** according
    /// to the metadata key `llm_expert_used_count`.
    ///
    /// *Absent* (`None`) if the converter did not write the key.
    pub expert_used_count: Option<u64>,

    pub experts_block_count: Option<u64>,
}

impl ModelManifest {
    pub(crate) fn from_counts(
        format: ManifestModelFormat,
        base_name: String,
        source_locator: SourceLocator,
        counts: HashMap<String /* model_name */, CheckpointCounts>,
    ) -> Result<Self, ProfileError> {
        fn from_vec(v: Vec<u64>, label: &'static str) -> Result<u64, ProfileError> {
            let Some(last) = v.last() else {
                eprintln!("⚠️  {} list is empty - no value available", label);
                return Err(ProfileError::MissingProfileParameter(label));
            };

            if v.iter().all(|&x| x == *last) {
                return Ok(*last);
            }
            eprintln!("⚠️  {} mismatch values {:?}", label, v);
            Ok(*last)
        }
        fn from_opt_vec(v: Vec<Option<u64>>, label: &'static str) -> Option<u64> {
            if v.is_empty() {
                eprintln!("⚠️  {} list is empty - no value available", label);
                return None;
            }

            let non_none_v: Vec<u64> = v.into_iter().filter_map(|x| x).collect();

            if non_none_v.is_empty() {
                eprintln!(
                    "⚠️  {} list contains only `None` - no value available",
                    label
                );
                return None;
            }

            let first = non_none_v[0];
            if non_none_v.iter().all(|&x| x == first) {
                return Some(first);
            }
            eprintln!("⚠️  {} mismatch values {:?}", label, non_none_v);
            Some(first)
        }

        let mut checkpoints = HashMap::with_capacity(counts.len());
        let mut n_param_counts: Vec<u64> = Vec::with_capacity(counts.len());
        let mut block_counts: Vec<u64> = Vec::with_capacity(counts.len());
        let mut model_ctx_size_counts: Vec<Option<u64>> = Vec::with_capacity(counts.len());
        let mut n_embd_counts: Vec<Option<u64>> = Vec::with_capacity(counts.len());
        let mut n_head_counts: Vec<Option<u64>> = Vec::with_capacity(counts.len());
        let mut n_head_kv_counts: Vec<Option<u64>> = Vec::with_capacity(counts.len());
        let mut expert_count: Vec<Option<u64>> = Vec::with_capacity(counts.len());
        let mut expert_used_count: Vec<Option<u64>> = Vec::with_capacity(counts.len());
        let mut experts_block_count: Vec<Option<u64>> = Vec::with_capacity(counts.len());

        for (_, c) in counts.into_iter() {
            let block_count = c.blocks.len() as u64;
            if let Some(meta) = c.meta_block_count.filter(|&meta| meta != block_count) {
                eprintln!(
                    "⚠️  block count mismatch: expected {meta} from meta, found {}",
                    block_count
                );
            }

            let counted_expert_count = c.expert_count();
            let meta_expert_count = c.meta_expert_count;
            match (counted_expert_count, meta_expert_count) {
                (Some(counted), Some(meta)) if counted != meta => {
                    eprintln!(
                        "⚠️  expert count mismatch: expected {meta} from meta, found {counted}"
                    );
                }
                (None, None) => {}
                _ => {
                    eprintln!(
                        "⚠️  expert count mismatch: expected {:?} from meta, found {:?}",
                        meta_expert_count, counted_expert_count
                    );
                }
            }

            n_param_counts.push(c.params());
            block_counts.push(block_count);
            model_ctx_size_counts.push(c.meta_model_ctx_size);
            n_embd_counts.push(c.meta_n_embd);
            n_head_counts.push(c.meta_n_head);
            n_head_kv_counts.push(c.meta_n_head_kv);
            expert_count.push(counted_expert_count);
            expert_used_count.push(c.meta_expert_used_count);
            experts_block_count.push(c.experts_block_count());
            let checkpoint = CheckpointManifest::from_counts(c).unwrap();
            checkpoints.insert(checkpoint.model_name.clone(), checkpoint);
        }

        Ok(Self {
            format: format.clone(),
            base_name,
            source_locator,
            checkpoints,
            n_params: from_vec(n_param_counts, "n_params")?,
            block_count: from_vec(block_counts, "block_count")?,
            model_ctx_size: from_opt_vec(model_ctx_size_counts, "model_ctx_size"),
            n_embd: from_opt_vec(n_embd_counts, "n_embd"),
            n_head: from_opt_vec(n_head_counts, "n_head"),
            n_head_kv: from_opt_vec(n_head_kv_counts, "n_head_kv"),
            expert_count: from_opt_vec(expert_count, "expert_count"),
            expert_used_count: from_opt_vec(expert_used_count, "expert_used_count"),
            experts_block_count: from_opt_vec(experts_block_count, "experts_block_count"),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CheckpointManifest {
    pub model_name: String,
    pub files: CheckpointFiles,
    /// **Dominant tensor data‑type** used in the file.
    ///
    /// Aliases: `file_type`, `Encoding`, `ftype`.
    pub ggml_f_type: GgmlFileType,

    // /// **Total size in bytes** of the checkpoint file on disk.
    // ///
    // /// This is the size of the GGUF file, not the model size.
    // pub total_file_size_bytes: u64,
    //
    /// The total size in bytes of all tensor data in the GGUF model.
    ///
    /// This includes every tensor in the model and represents the overall memory footprint of the model's parameters.
    pub total_tensors_bytes: u64,

    /// The total size in bytes of all block-related tensor data.
    ///
    /// This sum includes only the tensors that belong to numbered blocks (layers), i.e., those with names prefixed by `"blk."`.
    pub total_blocks_bytes: u64, // tensor bytes that belong to blk.* layers

    /// Cumulative **byte size** of all expert weight tensors
    /// (`ffn_{up,down,gate}_exps.weight`).
    /// This is the exact amount of VRAM freed if those tensors are offloaded
    /// to CPU RAM.
    pub expert_blocks_bytes: Option<u64>,

    /// Majority quantisation scheme (`Q4_K_M`, `Q6_K_S`, …) used *inside* the
    /// expert tensors, determined by voting across their `GGmlType`s.
    pub experts_ggml_type: Option<GgmlFileType>,
}

impl CheckpointManifest {
    pub(crate) fn from_counts(c: CheckpointCounts) -> Result<Self, ProfileError> {
        debug_assert!(
            !c.blocks.is_empty(),
            "CheckpointCounts must have at least one block"
        );
        debug_assert!(
            !c.global_tensors.is_empty(),
            "CheckpointCounts must have at least one global tensor"
        );
        let total_tensors_bytes = c.total_tensors_bytes();
        debug_assert!(total_tensors_bytes > 0, "CheckpointCounts must have bytes");

        Ok(Self {
            model_name: c.parsed_model_name.to_owned(),
            files: c.files.clone(),
            ggml_f_type: c.ggml_f_type()?,
            total_tensors_bytes,
            total_blocks_bytes: c.total_blocks_bytes(),
            expert_blocks_bytes: c.expert_blocks_bytes(),
            experts_ggml_type: c.counted_expert_ggml_type()?,
        })
    }

    pub fn total_file_bytes(&self) -> u64 {
        self.files.total_file_size_bytes()
    }
}
