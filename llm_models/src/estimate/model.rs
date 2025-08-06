use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{
    estimate::{
        math::{
            average_layer_size_bytes, average_layer_size_bytes_with_moe,
            estimate_context_size_bytes_checked,
        },
        memory::RuntimeMemorySpec,
    },
    fs::file_status::{CheckpointFiles, SourceLocator},
    manifest::{
        file_encoding_type::GgmlFileType,
        model::{CheckpointManifest, ManifestModelFormat, ModelManifest},
    },
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CanidateModel {
    pub format: ManifestModelFormat,
    pub base_name: String,
    pub source_locator: SourceLocator,
    pub checkpoints: HashMap<String, CanidateCheckpoint>,
    pub largest_compatible_dense_type: Option<GgmlFileType>,
    pub largest_compatible_moe_type: Option<GgmlFileType>,
}

impl CanidateModel {
    pub fn new(mem_spec: &RuntimeMemorySpec, model_manifest: ModelManifest) -> Self {
        let context_size_bytes = estimate_context_size_bytes_checked(
            mem_spec
                .inference_ctx_size
                .or(model_manifest.model_ctx_size),
            model_manifest.n_embd,
            model_manifest.n_head,
            model_manifest.n_head_kv,
            model_manifest.block_count,
            model_manifest.expert_used_count,
            mem_spec,
        )
        .unwrap();

        let mut canidate_checkpoints = HashMap::new();
        for (checkpoint_name, checkpoint_manifest) in model_manifest.checkpoints {
            let canidate_checkpoint = CanidateCheckpoint::new(
                model_manifest.block_count,
                context_size_bytes,
                model_manifest.experts_block_count,
                checkpoint_manifest,
            );

            canidate_checkpoints.insert(checkpoint_name.clone(), canidate_checkpoint);
        }

        Self {
            format: model_manifest.format,
            base_name: model_manifest.base_name.to_owned(),
            source_locator: model_manifest.source_locator.clone(),
            checkpoints: canidate_checkpoints,
            largest_compatible_dense_type: None,
            largest_compatible_moe_type: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CanidateCheckpoint {
    pub model_name: String,
    pub files: CheckpointFiles,
    pub estimated_memory_usage_bytes: u64,
    pub layer_avg_size_bytes: u64,
    pub layer_moe_dense_avg_size_bytes: Option<u64>,
    pub layer_moe_experts_avg_size_bytes: Option<u64>,
}

impl CanidateCheckpoint {
    pub fn new(
        block_count: u64,
        context_size_bytes: u64,
        experts_block_count: Option<u64>,
        checkpoint_manifest: CheckpointManifest,
    ) -> Self {
        let estimated_memory_usage_bytes =
            checkpoint_manifest.total_tensors_bytes + context_size_bytes;

        let (layer_moe_dense_avg_size_bytes, layer_moe_experts_avg_size_bytes) =
            match (experts_block_count, checkpoint_manifest.expert_blocks_bytes) {
                (Some(experts_block_count), Some(expert_blocks_bytes)) => {
                    let (dense, experts) = average_layer_size_bytes_with_moe(
                        block_count,
                        checkpoint_manifest.total_blocks_bytes,
                        context_size_bytes,
                        experts_block_count,
                        expert_blocks_bytes,
                    );
                    (Some(dense), Some(experts))
                }
                (None, None) => (None, None),
                _ => todo!("experts_block_count present but expert_blocks_bytes missing"),
            };

        Self {
            model_name: checkpoint_manifest.model_name.clone(),
            files: checkpoint_manifest.files.clone(),
            estimated_memory_usage_bytes,
            layer_avg_size_bytes: average_layer_size_bytes(
                block_count,
                checkpoint_manifest.total_blocks_bytes,
                context_size_bytes,
            )
            .unwrap(),
            layer_moe_dense_avg_size_bytes,
            layer_moe_experts_avg_size_bytes,
        }
    }
}
