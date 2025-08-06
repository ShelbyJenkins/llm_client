use std::collections::HashMap;

use serde::Deserialize;

use crate::{
    fs::file_status::{FileLocator, FileStatus, SourceLocator},
    hf::{
        client::HfClient,
        id::{HfRFilename, HfRFilenameError, HfRepoId, HfRepoShaError},
    },
    manifest::{
        file_name_parse::{FileNameParseError, GgufFileNameParse},
        model::{ManifestModelError, ManifestModelFormat, ModelManifest},
        profile::CheckpointCounts,
    },
};

#[derive(Debug, thiserror::Error)]
pub enum HfProfileError {
    #[error(transparent)]
    Api(#[from] hf_hub::api::tokio::ApiError),

    #[error(transparent)]
    Sha(#[from] HfRepoShaError),

    #[error(transparent)]
    RFilename(#[from] HfRFilenameError),

    #[error("SHA mismatch: expected {expected}, received {found}")]
    ShaMismatch { expected: String, found: String },

    #[error(transparent)]
    FileNameParse(#[from] FileNameParseError),

    #[error("No models '{repo_name}' found in the repository")]
    NoModels { repo_name: String },

    #[error("Multiple models found: '{base_names}'")]
    MultipleModels { base_names: String },

    #[error("GGUF header exceeded size limit (needed {needed} B, cap {limit} B)")]
    HeaderTooLarge { needed: usize, limit: usize },
}

#[derive(Deserialize)]
pub struct RawRepoInfo {
    siblings: Vec<RawSibling>,
    sha: String,
}

#[derive(Deserialize)]
struct RawSibling {
    pub rfilename: String,
    pub size: u64,
}

pub async fn model_from_hf_repo(
    client: &HfClient,
    mut repo_id: HfRepoId,
    format: ManifestModelFormat,
) -> Result<ModelManifest, ManifestModelError> {
    let repo_info = client.repo_info(&repo_id).await?;
    let repo_name_lower = repo_id.repo_name().to_ascii_lowercase();
    if let Some(sha) = repo_id.sha() {
        if sha != repo_info.sha {
            return Err(HfProfileError::ShaMismatch {
                expected: sha.to_string(),
                found: repo_info.sha,
            }
            .into());
        }
    } else {
        repo_id
            .set_sha(repo_info.sha)
            .map_err(|e| ManifestModelError::HfModel(e.into()))?;
    }

    let source_locator = SourceLocator::HfRepoId(repo_id.clone());
    let mut checkpoint_counts: HashMap<String /* model_name */, CheckpointCounts> = HashMap::new();
    let mut first_base_name: Option<String> = None;

    for s in repo_info.siblings {
        let (base_name, model_name, shard_id) = match format {
            ManifestModelFormat::SafeTensors => todo!(),
            ManifestModelFormat::Gguf => match GgufFileNameParse::from_fname(&s.rfilename) {
                Ok(parsed) => (parsed.base_name, parsed.model_name, parsed.shard_id),
                Err(e) => match e {
                    FileNameParseError::InvalidNameOutput { .. }
                    | FileNameParseError::ShardId { .. } => {
                        return Err(HfProfileError::FileNameParse(e).into());
                    }
                    _ => continue, // Other errors represent non-gguf model files, so we skip them
                },
            },
        };

        // Ensure we only process files that match the repo name aka model base name
        if !repo_name_lower.contains(&base_name.to_ascii_lowercase()) {
            continue;
        }

        if let Some(ref first_base_name) = first_base_name {
            if first_base_name != &base_name {
                return Err(HfProfileError::MultipleModels {
                    base_names: format!("{} and {}", first_base_name, base_name),
                }
                .into());
            }
        } else {
            first_base_name = Some(base_name.clone());
        }

        match checkpoint_counts.get_mut(&model_name) {
            None => {
                let checkpoint_count = CheckpointCounts::from_hf_repo(
                    client,
                    &repo_id,
                    &format,
                    &s.rfilename,
                    s.size,
                    &base_name,
                    &model_name,
                    &shard_id,
                )
                .await?;
                checkpoint_counts.insert(model_name.clone(), checkpoint_count);
            }
            Some(checkpoint_count) => {
                let file_locator =
                    FileLocator::HfRFilename(HfRFilename::try_new(&s.rfilename).unwrap());
                let status = FileStatus::new(client.cache_repo(&repo_id).get(&s.rfilename), s.size);

                checkpoint_count.push_file(status, file_locator, s.size, &shard_id)?;
            }
        }
    }

    if checkpoint_counts.is_empty() {
        return Err(HfProfileError::NoModels {
            repo_name: repo_id.repo_name().to_string(),
        }
        .into());
    }
    for checkpoint in checkpoint_counts.values() {
        checkpoint.validate()?;
    }
    Ok(ModelManifest::from_counts(
        format,
        first_base_name.unwrap(),
        source_locator,
        checkpoint_counts,
    )
    .unwrap())
}
