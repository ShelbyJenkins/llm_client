use ggus::{GGuf, GGufMetaError, GGufMetaMapExt};
use serde::{Deserialize, Serialize};
use url::Url;

use crate::manifest::file_encoding_type::GgmlFileType;

/// General metadata of the model, covering identity and basic info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct General {
    /// The model architecture (e.g., "llama", "gptneox", "gptj").
    pub architecture: Option<String>,
    /// Quantization format version (present if model is quantized).
    pub quantization_version: Option<u32>,
    /// Global tensor alignment (multiple of 8, default 32 if not specified).
    pub alignment: u32,
    /// Human-readable name of the model.
    pub name: Option<String>,
    /// Author or creator of the model.
    pub author: Option<String>,
    /// Version of the model (e.g., "v1.0").
    pub version: Option<String>,
    /// Organization or group associated with the model.
    pub organization: Option<String>,
    /// Base model name or architecture of the model.
    pub basename: Option<String>,
    /// Fine-tuning target or purpose of the model.
    pub finetune: Option<String>,
    /// Free-form description of the model.
    pub description: Option<String>,
    /// Name of the person or tool that quantized the model.
    pub quantized_by: Option<String>,
    /// Size classification of the model (e.g., "7B" for 7 billion parameters).
    pub size_label: Option<String>,
    /// License information for the model.
    pub license: License,
    /// URL to the model's homepage or documentation.
    pub url: Option<Url>,
    /// Digital Object Identifier (DOI) for the model.
    pub doi: Option<String>,
    /// Universally Unique Identifier (UUID) for the model.
    pub uuid: Option<String>,
    /// URL to the model's repository (e.g., GitHub or HuggingFace).
    pub repo_url: Option<Url>,
    /// List of keywords or tags relevant to the model.
    pub tags: Vec<String>,
    /// Languages the model supports (ISO 639-1 codes).
    pub languages: Vec<String>,
    /// Datasets used to train or fine-tune the model.
    pub datasets: Vec<String>,
    /// Enumerated file type (dominant tensor data type, e.g., 0=ALL_F32, 1=MOSTLY_F16).
    #[serde(skip)]
    pub file_type: Option<GgmlFileType>,
    /// Source/provenance metadata of the model.
    pub source: SourceMetadata,
}

impl General {
    pub fn new(gguf: &GGuf) -> Result<Self, GGufMetaError> {
        Ok(Self {
            architecture: gguf.general_architecture().ok().map(str::to_owned),
            quantization_version: gguf.general_quantization_version().ok().map(|v| v as u32),
            alignment: gguf.general_alignment().unwrap() as u32,
            file_type: gguf
                .general_filetype()
                .ok()
                .map(|v| GgmlFileType::from_file_type(v).ok())
                .flatten(),
            name: gguf.general_name().ok().map(str::to_owned),
            author: gguf.general_author().ok().map(str::to_owned),
            version: gguf.general_version().ok().map(str::to_owned),
            organization: gguf.general_organization().ok().map(str::to_owned),
            basename: gguf.general_basename().ok().map(str::to_owned),
            finetune: gguf.general_finetune().ok().map(str::to_owned),
            description: gguf.general_description().ok().map(str::to_owned),
            quantized_by: gguf.general_quantized_by().ok().map(str::to_owned),
            size_label: gguf.general_size_label().ok().map(str::to_owned),
            license: License::new(gguf),
            url: gguf
                .general_url()
                .ok()
                .map(|u| Url::parse(&u).ok())
                .flatten(),
            doi: gguf.general_doi().ok().map(str::to_owned),
            uuid: gguf.general_uuid().ok().map(str::to_owned),
            repo_url: gguf
                .general_repo_url()
                .ok()
                .map(|u| Url::parse(&u).ok())
                .flatten(),
            tags: gguf
                .general_tags()
                .ok()
                .map(|vals| {
                    vals.filter_map(Result::ok)
                        .map(str::to_owned)
                        .collect::<Vec<String>>()
                })
                .unwrap_or_default(),
            languages: gguf
                .general_languages()
                .ok()
                .map(|vals| {
                    vals.filter_map(Result::ok)
                        .map(str::to_owned)
                        .collect::<Vec<String>>()
                })
                .unwrap_or_default(),
            datasets: gguf
                .general_datasets()
                .ok()
                .map(|vals| {
                    vals.filter_map(Result::ok)
                        .map(str::to_owned)
                        .collect::<Vec<String>>()
                })
                .unwrap_or_default(),
            source: SourceMetadata::new(gguf),
        })
    }
}

/// License details of the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct License {
    /// SPDX license expression (e.g., "MIT OR Apache-2.0").
    pub spdx_expression: Option<String>,
    /// Human-readable license name (e.g., "Apache License 2.0").
    pub human_name: Option<String>,
    /// URL to the license text or terms.
    pub link: Option<Url>,
}

impl License {
    pub fn new(gguf: &ggus::GGuf) -> Self {
        Self {
            spdx_expression: gguf.general_license().ok().map(str::to_owned),
            human_name: gguf.general_license_name().ok().map(str::to_owned),
            link: gguf
                .general_license_link()
                .ok()
                .map(|u| Url::parse(&u).ok())
                .flatten(),
        }
    }
}

/// Metadata about the source or origin of the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceMetadata {
    /// URL to the source model's homepage or documentation.
    pub url: Option<Url>,
    /// Source model's DOI (Digital Object Identifier), if any.
    pub doi: Option<String>,
    /// Source model's UUID (Universally Unique Identifier), if any.
    pub uuid: Option<String>,
    /// URL to the source model's repository (e.g., GitHub or HuggingFace).
    pub repo_url: Option<Url>,
    /// Number of base (parent) models this model was derived from.
    pub base_model_count: Option<u32>,
    /// List of base model metadata entries (parent models).
    pub base_models: Vec<SourceModel>,
}

impl SourceMetadata {
    pub fn new(gguf: &ggus::GGuf) -> Self {
        let base_models = gguf
            .general_base_model_count()
            .ok()
            .map(|count| {
                (0..count)
                    .filter_map(|id| {
                        gguf.general_base_model_name(id)
                            .ok()
                            .map(|_| SourceModel::new(gguf, id))
                    })
                    .collect::<Vec<SourceModel>>()
            })
            .unwrap_or_default();
        Self {
            url: gguf
                .general_source_url()
                .ok()
                .map(|u| Url::parse(&u).ok())
                .flatten(),
            doi: gguf.general_source_doi().ok().map(str::to_owned),
            uuid: gguf.general_source_uuid().ok().map(str::to_owned),
            repo_url: gguf
                .general_source_repo_url()
                .ok()
                .map(|u| Url::parse(&u).ok())
                .flatten(),
            base_model_count: Some(base_models.len() as u32),
            base_models,
        }
    }
}

/// Information about a single base (parent) model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceModel {
    /// Name of the base model.
    pub name: Option<String>,
    /// Author of the base model.
    pub author: Option<String>,
    /// Version of the base model.
    pub version: Option<String>,
    /// Organization associated with the base model.
    pub organization: Option<String>,
    /// URL to the base model's homepage or documentation.
    pub url: Option<Url>,
    /// DOI of the base model, if available.
    pub doi: Option<String>,
    /// UUID of the base model, if available.
    pub uuid: Option<String>,
    /// Repository URL of the base model.
    pub repo_url: Option<Url>,
}

impl SourceModel {
    pub fn new(gguf: &ggus::GGuf, id: usize) -> Self {
        Self {
            name: gguf.general_base_model_name(id).ok().map(str::to_owned),
            author: gguf.general_base_model_author(id).ok().map(str::to_owned),
            version: gguf.general_base_model_version(id).ok().map(str::to_owned),
            organization: gguf
                .general_base_model_organization(id)
                .ok()
                .map(str::to_owned),
            url: gguf
                .general_base_model_url(id)
                .ok()
                .map(|u| Url::parse(&u).ok())
                .flatten(),
            doi: gguf.general_base_model_doi(id).ok().map(str::to_owned),
            uuid: gguf.general_base_model_uuid(id).ok().map(str::to_owned),
            repo_url: gguf
                .general_base_model_repo_url(id)
                .ok()
                .map(|u| Url::parse(&u).ok())
                .flatten(),
        }
    }
}
