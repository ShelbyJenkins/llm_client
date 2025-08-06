use std::{
    collections::{BTreeMap, HashMap},
    num::NonZeroU32,
    vec,
};

use ggus::GGufMetaMapExt;

use crate::{
    fs::file_status::{CheckpointFile, CheckpointFiles, FileLocator, FileStatus},
    hf::{
        client::HfClient,
        id::{HfRFilename, HfRepoId},
    },
    manifest::{
        file_encoding_type::{GgmlFileType, GgmlFileTypeError},
        model::{ManifestModelError, ManifestModelFormat},
        shard_id::ShardId,
    },
};

#[derive(Debug, thiserror::Error)]
pub enum ProfileError {
    #[error("error loading metadata: {0}")]
    MetadataLoadError(String),

    #[error("tensor '{tensor}' is empty (zero {parameter})")]
    TensorEmptyOrMissing {
        tensor: String,
        parameter: &'static str,
    },

    #[error("no tensors found in GGUF file")]
    NoTensorsFound,

    #[error("no blocks found in GGUF file")]
    NoBlocksFound,

    #[error("invalid block index in tensor name '{block}': {source}")]
    InvalidBlockIndex {
        block: String,
        #[source]
        source: std::num::ParseIntError,
    },

    #[error("MOE block tensor '{tensor}' does not have 3 dimensions (expected experts)")]
    MoeBlockMissingExperts { tensor: String },

    #[error("MOE block tensor '{tensor}' is has an expert count of 0 (expected experts)")]
    MoeBlockZeroExperts { tensor: String },

    #[error("missing required model profile parameter: {0}")]
    MissingProfileParameter(&'static str),

    #[error(transparent)]
    GgmlFileTypeError(#[from] GgmlFileTypeError),
}

pub struct CheckpointCounts {
    pub global_tensors: Vec<Tensor>,
    pub blocks: HashMap<u64, Block>,
    pub files: CheckpointFiles,
    pub parsed_base_name: String,
    pub parsed_model_name: String,
    pub meta_ggml_f_type: Option<GgmlFileType>,
    pub meta_block_count: Option<u64>,
    pub meta_model_ctx_size: Option<u64>,
    pub meta_n_embd: Option<u64>,
    pub meta_n_head: Option<u64>,
    pub meta_n_head_kv: Option<u64>,
    pub meta_expert_count: Option<u64>,
    pub meta_expert_used_count: Option<u64>,
}

impl CheckpointCounts {
    pub async fn from_hf_repo(
        client: &HfClient,
        repo_id: &HfRepoId,
        format: &ManifestModelFormat,
        rfilename: &str,
        total_file_size_bytes: u64,
        base_name: &str,
        model_name: &str,
        shard_id: &ShardId,
    ) -> Result<Self, ManifestModelError> {
        match format {
            ManifestModelFormat::Gguf => {
                let buffer = client.load_gguf_header_buffer(repo_id, rfilename).await?;

                let file_locator =
                    FileLocator::HfRFilename(HfRFilename::try_new(rfilename).unwrap());
                let status = FileStatus::new(
                    client.cache_repo(&repo_id).get(&rfilename),
                    total_file_size_bytes,
                );

                Ok(CheckpointCounts::from_gguf(
                    status,
                    file_locator,
                    total_file_size_bytes,
                    base_name,
                    model_name,
                    shard_id,
                    buffer,
                )
                .unwrap())
            }
            ManifestModelFormat::SafeTensors => {
                todo!("SafeTensors profiling not implemented yet")
            }
        }
    }

    pub fn from_gguf(
        status: FileStatus,
        file_locator: FileLocator,
        total_file_size_bytes: u64,
        base_name: &str,
        model_name: &str,
        shard_id: &ShardId,
        buffer: Vec<u8>,
    ) -> Result<Self, ProfileError> {
        let gguf = ggus::GGuf::new(&buffer)
            .map_err(|e| ProfileError::MetadataLoadError(format!("Failed to parse GGUF: {e}")))?;

        let mut global_tensors = Vec::new();
        let mut blocks = HashMap::<u64, Block>::new();

        for (name, meta) in gguf.tensors.iter() {
            let info = meta.to_info();
            let shape = info.shape();
            let params = shape.iter().product::<u64>();
            let bytes = info.nbytes() as u64;
            let ggml_f_type_value = info.ty() as u32;

            if shape.is_empty() {
                return Err(ProfileError::TensorEmptyOrMissing {
                    tensor: name.to_string(),
                    parameter: "rank (zero-dimensional)",
                });
            }

            if params == 0 {
                return Err(ProfileError::TensorEmptyOrMissing {
                    tensor: name.to_string(),
                    parameter: "parameters",
                });
            }

            if bytes == 0 {
                return Err(ProfileError::TensorEmptyOrMissing {
                    tensor: name.to_string(),
                    parameter: "bytes",
                });
            }

            if let Some(blk_str) = name
                .strip_prefix("blk.")
                .and_then(|r| r.split_once('.').map(|t| t.0))
            {
                let block_id =
                    blk_str
                        .parse::<u64>()
                        .map_err(|e| ProfileError::InvalidBlockIndex {
                            block: blk_str.into(),
                            source: e,
                        })?;

                let acc = blocks.entry(block_id).or_insert_with(|| Block {
                    block_id,
                    tensors: Vec::new(),
                });
                let experts: Option<u64> = if name.ends_with("_exps.weight") {
                    let experts = (shape.len() == 3)
                        .then(|| *shape.last().expect("only evaluated for 3D tensors"))
                        .ok_or_else(|| ProfileError::MoeBlockMissingExperts {
                            tensor: name.to_string(),
                        })?;
                    if experts == 0 {
                        return Err(ProfileError::MoeBlockZeroExperts {
                            tensor: name.to_string(),
                        });
                    }
                    Some(experts)
                } else {
                    None
                };

                acc.tensors.push(Tensor {
                    params,
                    bytes,
                    ggml_f_type_value,
                    experts,
                });
            } else {
                global_tensors.push(Tensor {
                    params,
                    bytes,
                    ggml_f_type_value,
                    experts: None,
                });
            };
        }
        let general_filetype = gguf.general_filetype().ok();
        let meta_ggml_f_type = if let Some(ft) = general_filetype {
            Some(GgmlFileType::from_file_type(ft)?)
        } else {
            None
        };

        if (global_tensors.len() + blocks.len()) == 0 {
            return Err(ProfileError::NoTensorsFound);
        }
        if blocks.is_empty() {
            return Err(ProfileError::NoBlocksFound);
        }

        let files = if shard_id.is_single() {
            CheckpointFiles::Single(CheckpointFile::new(
                file_locator,
                total_file_size_bytes,
                status,
            ))
        } else {
            CheckpointFiles::Sharded {
                total: shard_id.total(),
                parts: BTreeMap::from([(
                    shard_id.index(),
                    CheckpointFile::new(file_locator, total_file_size_bytes, status),
                )]),
            }
        };
        Ok(Self {
            global_tensors,
            blocks,
            files,
            parsed_base_name: base_name.to_owned(),
            parsed_model_name: model_name.to_owned(),
            meta_ggml_f_type,
            meta_block_count: gguf.llm_block_count().ok().map(|v| v as u64),
            meta_model_ctx_size: gguf.llm_context_length().ok().map(|v| v as u64),
            meta_n_embd: gguf.llm_embedding_length().ok().map(|v| v as u64),
            meta_n_head: gguf.llm_attention_head_count().ok().map(|v| v as u64),
            meta_n_head_kv: gguf.llm_attention_head_count_kv().ok().map(|v| v as u64),
            meta_expert_count: gguf.llm_expert_count().ok().map(|v| v as u64),
            meta_expert_used_count: gguf.llm_expert_used_count().ok().map(|v| v as u64),
        })
    }

    pub fn push_file(
        &mut self,
        status: FileStatus,
        file_locator: FileLocator,
        total_file_size_bytes: u64,
        shard_id: &ShardId,
    ) -> Result<(), ManifestModelError> {
        if shard_id.is_single() {
            return Err(ManifestModelError::DuplicateCheckpoint {
                model: self.parsed_model_name.clone(),
                checkpoint: file_locator.as_str().to_string(),
            });
        }
        if let CheckpointFiles::Sharded { total, parts } = &mut self.files {
            if shard_id.total() != *total {
                return Err(ManifestModelError::MismatchedShardTotal {
                    model: self.parsed_model_name.clone(),
                    expected: *total,
                    found: shard_id.total(),
                });
            }

            let idx = shard_id.index();

            if parts.contains_key(&idx) {
                return Err(ManifestModelError::DuplicateShards {
                    model: self.parsed_model_name.clone(),
                    duplicates: vec![idx],
                });
            }

            parts.insert(
                idx,
                CheckpointFile::new(file_locator, total_file_size_bytes, status),
            );
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<(), ManifestModelError> {
        match &self.files {
            CheckpointFiles::Single(_) => Ok(()),

            CheckpointFiles::Sharded { total, parts } => {
                let expected_total = total.get();
                debug_assert!(expected_total == parts.len() as u32);

                // Collect every index that should exist but doesn't.
                let missing: Vec<NonZeroU32> = (1..=expected_total)
                    .filter_map(NonZeroU32::new)
                    .filter(|idx| !parts.contains_key(idx))
                    .collect();

                if !missing.is_empty() {
                    return Err(ManifestModelError::MissingShardIndexes {
                        model: self.parsed_model_name.clone(),
                        expected_total,
                        missing,
                    });
                }
                Ok(())
            }
        }
    }

    pub fn params(&self) -> u64 {
        self.global_tensors.iter().map(|t| t.params).sum::<u64>()
            + self.blocks.values().map(|b| b.params()).sum::<u64>()
    }

    pub fn total_tensors_bytes(&self) -> u64 {
        self.global_tensors.iter().map(|t| t.bytes).sum::<u64>()
            + self.blocks.values().map(|b| b.bytes()).sum::<u64>()
    }

    pub fn bits(&self) -> u128 {
        self.global_tensors
            .iter()
            .map(|t| (t.bytes as u128) * 8)
            .sum::<u128>()
            + self.blocks.values().map(|b| b.bits()).sum::<u128>()
    }

    pub fn total_blocks_bytes(&self) -> u64 {
        self.blocks.values().map(|b| b.bytes()).sum::<u64>()
    }

    pub fn expert_blocks(&self) -> Vec<&Block> {
        self.blocks
            .values()
            .filter(|acc| {
                !acc.tensors.is_empty() && acc.tensors.iter().any(|t| t.experts.is_some())
            })
            .collect()
    }

    pub fn expert_count(&self) -> Option<u64> {
        if self.expert_blocks().is_empty() {
            return None;
        }

        let experts = self
            .expert_blocks()
            .iter()
            .map(|blk| blk.tensors.iter().filter_map(|t| t.experts).sum::<u64>())
            .sum::<u64>();
        debug_assert!(experts > 0, "expert_blocks() must have at least one expert");
        Some(experts)
    }

    pub fn experts_block_count(&self) -> Option<u64> {
        if self.expert_blocks().is_empty() {
            return None;
        }

        let count = self.expert_blocks().len() as u64;
        Some(count)
    }

    pub fn expert_blocks_bytes(&self) -> Option<u64> {
        let b = self
            .expert_blocks()
            .iter()
            .map(|blk| blk.bytes())
            .sum::<u64>();
        if b == 0 { None } else { Some(b) }
    }

    pub fn ggml_f_type(&self) -> Result<GgmlFileType, GgmlFileTypeError> {
        // majority vote across tensor types
        let ty_from_counts = self.counted_by_ty_ggml_f_type()?;

        // heuristic based on bits-per-weight (may fail ⇒ None)
        let ty_from_bpw = self.counted_by_bpw_ggml_f_type().ok();

        match ty_from_bpw {
            Some(bpw) => {
                // Compare bpw against: ① vote result, ② meta tag (if present)
                let mismatch = std::iter::once(&ty_from_counts)
                    .chain(self.meta_ggml_f_type.as_ref())
                    .any(|&ft| ft != bpw);

                if mismatch {
                    eprintln!(
                        "⚠️  GGML-type mismatch: bpw = {:?}, vote = {:?}, meta = {:?}",
                        bpw, ty_from_counts, self.meta_ggml_f_type,
                    );
                }
            }
            None => {
                eprintln!("⚠️  Could not infer GGML type by bits-per-weight; using vote");
            }
        }

        Ok(ty_from_counts)
    }

    pub fn counted_by_ty_ggml_f_type(&self) -> Result<GgmlFileType, GgmlFileTypeError> {
        use std::collections::HashMap;

        // — accumulate votes in a single pass —
        let mut counts: HashMap<u32, usize> = HashMap::new();

        // block tensors
        for block in self.blocks.values() {
            for &ty in &block.ggml_f_type_values() {
                *counts.entry(ty).or_insert(0) += 1;
            }
        }

        // global tensors
        for tensor in &self.global_tensors {
            *counts.entry(tensor.ggml_f_type_value).or_insert(0) += 1;
        }

        Self::count_ggml_f_type(counts)
    }

    pub fn counted_by_bpw_ggml_f_type(&self) -> Result<GgmlFileType, GgmlFileTypeError> {
        let bits_per_weight_avg = (self.bits() as f64 / self.params() as f64) as f64;
        GgmlFileType::from_bits_per_weight(bits_per_weight_avg)
    }

    pub fn counted_expert_ggml_type(&self) -> Result<Option<GgmlFileType>, GgmlFileTypeError> {
        if self.expert_blocks().is_empty() {
            return Ok(None);
        }

        let mut counts: HashMap<u32, usize> = HashMap::new();
        for block in self.expert_blocks() {
            for &ty in &block.ggml_f_type_values() {
                *counts.entry(ty).or_insert(0) += 1;
            }
        }
        Ok(Some(Self::count_ggml_f_type(counts)?))
    }

    pub fn count_ggml_f_type(
        counts: HashMap<u32, usize>,
    ) -> Result<GgmlFileType, GgmlFileTypeError> {
        debug_assert!(
            !counts.is_empty(),
            "all blocks must have at least one tensor type. This is checked in `from_gguf`."
        );

        // — find the leader, preferring lower numeric value on ties —
        let majority_ty = counts
            .into_iter()
            .max_by(|&(ty_a, cnt_a), &(ty_b, cnt_b)| {
                cnt_a.cmp(&cnt_b).then_with(|| ty_b.cmp(&ty_a))
            }) // tie-break: keep *smaller* ty
            .map(|(ty, _)| ty)
            .expect("Checkpoint must contain at least one tensor type");

        GgmlFileType::from_file_type_value(majority_ty)
    }
}

pub struct Block {
    #[allow(dead_code)]
    block_id: u64,
    tensors: Vec<Tensor>,
}

impl Block {
    pub fn params(&self) -> u64 {
        self.tensors.iter().map(|t| t.params).sum()
    }

    pub fn bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.bytes).sum()
    }

    pub fn bits(&self) -> u128 {
        self.tensors.iter().map(|t| (t.bytes as u128) * 8).sum()
    }

    pub fn ggml_f_type_values(&self) -> Vec<u32> {
        self.tensors.iter().map(|t| t.ggml_f_type_value).collect()
    }
}

pub struct Tensor {
    params: u64,
    bytes: u64,
    ggml_f_type_value: u32,
    experts: Option<u64>,
}
