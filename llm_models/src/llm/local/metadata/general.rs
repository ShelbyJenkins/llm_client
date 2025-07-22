use crate::llm::local::gguf::{gguf_file::GgufFile, gguf_tensors::GgmlDType};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct GeneralMetadata {
    // Required fields
    pub architecture: String,
    pub quantization_version: u32,
    pub alignment: u32,
    pub file_type: FileType,

    // Optional metadata fields
    pub name: Option<String>,
    pub author: Option<String>,
    pub version: Option<String>,
    pub organization: Option<String>,
    pub basename: Option<String>,
    pub finetune: Option<String>,
    pub description: Option<String>,
    pub quantized_by: Option<String>,
    pub size_label: Option<String>,
    pub license: Option<String>,
    pub license_name: Option<String>,
    pub license_link: Option<String>,
    pub url: Option<String>,
    pub doi: Option<String>,
    pub uuid: Option<String>,
    pub repo_url: Option<String>,
    pub tags: Option<Vec<String>>,
    pub languages: Option<Vec<String>>,
    pub datasets: Option<Vec<String>>,

    // Source metadata
    pub source: SourceMetadata,
}

impl GeneralMetadata {
    pub fn from_gguf_file(gguf: &GgufFile) -> crate::Result<Self> {
        Ok(Self {
            architecture: gguf.get_value("general.architecture")?,
            quantization_version: gguf.get_value("general.quantization_version")?,
            alignment: gguf.get_value("general.alignment")?,
            name: gguf.get_value("general.name")?,
            author: gguf.get_value("general.author")?,
            version: gguf.get_value("general.version")?,
            organization: gguf.get_value("general.organization")?,
            basename: gguf.get_value("general.basename")?,
            finetune: gguf.get_value("general.finetune")?,
            description: gguf.get_value("general.description")?,
            quantized_by: gguf.get_value("general.quantized_by")?,
            size_label: gguf.get_value("general.size_label")?,
            license: gguf.get_value("general.license")?,
            license_name: gguf.get_value("general.license_name")?,
            license_link: gguf.get_value("general.license_link")?,
            url: gguf.get_value("general.url")?,
            doi: gguf.get_value("general.doi")?,
            uuid: gguf.get_value("general.uuid")?,
            repo_url: gguf.get_value("general.repo_url")?,
            tags: gguf.get_value("general.tags")?,
            languages: gguf.get_value("general.languages")?,
            datasets: gguf.get_value("general.datasets")?,
            file_type: FileType::from_gguf_file(gguf)?,
            source: SourceMetadata::from_gguf_file(gguf)?,
        })
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct SourceMetadata {
    pub url: Option<String>,
    pub doi: Option<String>,
    pub uuid: Option<String>,
    pub repo_url: Option<String>,
}

impl SourceMetadata {
    pub fn from_gguf_file(gguf: &GgufFile) -> crate::Result<Self> {
        Ok(Self {
            url: gguf.get_value("general.source.url")?,
            doi: gguf.get_value("general.source.doi")?,
            uuid: gguf.get_value("general.source.uuid")?,
            repo_url: gguf.get_value("general.source.repo_url")?,
        })
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum FileType {
    AllF32 = 0,
    MostlyF16 = 1,
    MostlyQ4_0 = 2,
    MostlyQ4_1 = 3,
    MostlyQ4_1SomeF16 = 4,
    MostlyQ8_0 = 7,
    MostlyQ5_0 = 8,
    MostlyQ5_1 = 9,
    MostlyQ2K = 10,
    MostlyQ3KS = 11,
    MostlyQ3KM = 12,
    MostlyQ3KL = 13,
    MostlyQ4KS = 14,
    MostlyQ4KM = 15,
    MostlyQ5KS = 16,
    MostlyQ5KM = 17,
    MostlyQ6K = 18,
}

impl FileType {
    pub fn from_gguf_file(gguf: &GgufFile) -> crate::Result<Self> {
        match (
            Self::get_from_gguf_file(gguf),
            Self::infer_from_gguf_file(gguf),
        ) {
            (Ok(retrieved_file_type), Ok(inferred_file_type)) => {
                if retrieved_file_type != inferred_file_type {
                    crate::error!(
                        "file_type mismatch. Retrieved: {:?}, Inferred: {:?}. Using inferred.",
                        retrieved_file_type,
                        inferred_file_type
                    );
                }
                Ok(inferred_file_type)
            }
            (_, Ok(file_type)) => Ok(file_type),
            (Ok(file_type), _) => Ok(file_type),
            _ => crate::bail!("file_type not found in gguf file nor was it able to be inferred"),
        }
    }

    pub fn get_from_gguf_file(gguf: &GgufFile) -> crate::Result<Self> {
        let file_type: Option<u32> = gguf.get_value("general.file_type")?;
        if let Some(file_type) = file_type {
            Self::from_u32(file_type)
        } else {
            crate::bail!("file_type none in gguf file")
        }
    }

    pub fn infer_from_gguf_file(gguf: &GgufFile) -> crate::Result<Self> {
        // Suppose you have a Vec<TensorInfo> named `tensors`
        // We'll track how many times each ggml_dtype appears:
        let mut dtype_counts = std::collections::HashMap::new();
        for info in &gguf.tensors {
            *dtype_counts.entry(info.ggml_dtype).or_insert(0) += 1;
        }

        // Find which dtype occurs the most
        let dominant_dtype = dtype_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .ok_or_else(|| crate::anyhow!("no tensors found"))?
            .0
            .to_owned();

        Ok(Self::from_ggml_d_type(&dominant_dtype))
    }

    pub fn from_u32(u: u32) -> crate::Result<Self> {
        match u {
            0 => Ok(Self::AllF32),
            1 => Ok(Self::MostlyF16),
            2 => Ok(Self::MostlyQ4_0),
            3 => Ok(Self::MostlyQ4_1),
            4 => Ok(Self::MostlyQ4_1SomeF16),
            7 => Ok(Self::MostlyQ8_0),
            8 => Ok(Self::MostlyQ5_0),
            9 => Ok(Self::MostlyQ5_1),
            10 => Ok(Self::MostlyQ2K),
            11 => Ok(Self::MostlyQ3KS),
            12 => Ok(Self::MostlyQ3KM),
            13 => Ok(Self::MostlyQ3KL),
            14 => Ok(Self::MostlyQ4KS),
            15 => Ok(Self::MostlyQ4KM),
            16 => Ok(Self::MostlyQ5KS),
            17 => Ok(Self::MostlyQ5KM),
            18 => Ok(Self::MostlyQ6K),
            _ => crate::bail!("Invalid FileType value: {}", u),
        }
    }

    pub fn to_ggml_d_type(&self) -> GgmlDType {
        match self {
            Self::AllF32 => GgmlDType::F32,
            Self::MostlyF16 => GgmlDType::F16,
            Self::MostlyQ4_0 => GgmlDType::Q4_0,
            Self::MostlyQ4_1 => GgmlDType::Q4_1,
            Self::MostlyQ4_1SomeF16 => GgmlDType::Q4_1,
            Self::MostlyQ8_0 => GgmlDType::Q8_0,
            Self::MostlyQ5_0 => GgmlDType::Q5_0,
            Self::MostlyQ5_1 => GgmlDType::Q5_1,
            Self::MostlyQ2K => GgmlDType::Q2K,
            Self::MostlyQ3KS => GgmlDType::Q3K,
            Self::MostlyQ3KM => GgmlDType::Q3K,
            Self::MostlyQ3KL => GgmlDType::Q3K,
            Self::MostlyQ4KS => GgmlDType::Q4K,
            Self::MostlyQ4KM => GgmlDType::Q4K,
            Self::MostlyQ5KS => GgmlDType::Q5K,
            Self::MostlyQ5KM => GgmlDType::Q5K,
            Self::MostlyQ6K => GgmlDType::Q6K,
        }
    }

    pub fn from_ggml_d_type(ggml_d_type: &GgmlDType) -> Self {
        match ggml_d_type {
            GgmlDType::F32 => Self::AllF32,
            GgmlDType::F16 => Self::MostlyF16,
            GgmlDType::Q4_0 => Self::MostlyQ4_0,
            GgmlDType::Q4_1 => Self::MostlyQ4_1,
            GgmlDType::Q5_0 => Self::MostlyQ5_0,
            GgmlDType::Q5_1 => Self::MostlyQ5_1,
            GgmlDType::Q8_0 => Self::MostlyQ8_0,
            GgmlDType::Q8_1 => Self::MostlyQ8_0,
            GgmlDType::Q2K => Self::MostlyQ2K,
            GgmlDType::Q3K => Self::MostlyQ3KS,
            GgmlDType::Q4K => Self::MostlyQ4KS,
            GgmlDType::Q5K => Self::MostlyQ5KS,
            GgmlDType::Q6K => Self::MostlyQ6K,
            GgmlDType::Q8K => Self::MostlyQ8_0,
        }
    }

    pub fn to_quant_type(&self) -> (String, usize, Option<String>) {
        match self {
            Self::AllF32 => ("f".to_owned(), 32, None),
            Self::MostlyF16 => ("f".to_owned(), 16, None),
            Self::MostlyQ4_0 => ("q".to_owned(), 4, None),
            Self::MostlyQ4_1 => ("q".to_owned(), 4, None),
            Self::MostlyQ4_1SomeF16 => ("q".to_owned(), 4, None),
            Self::MostlyQ8_0 => ("q".to_owned(), 8, None),
            Self::MostlyQ5_0 => ("q".to_owned(), 5, None),
            Self::MostlyQ5_1 => ("q".to_owned(), 5, None),
            Self::MostlyQ2K => ("q".to_owned(), 2, Some("k".to_owned())),
            Self::MostlyQ3KS => ("q".to_owned(), 3, Some("ks".to_owned())),
            Self::MostlyQ3KM => ("q".to_owned(), 3, Some("km".to_owned())),
            Self::MostlyQ3KL => ("q".to_owned(), 3, Some("kl".to_owned())),
            Self::MostlyQ4KS => ("q".to_owned(), 4, Some("ks".to_owned())),
            Self::MostlyQ4KM => ("q".to_owned(), 4, Some("km".to_owned())),
            Self::MostlyQ5KS => ("q".to_owned(), 5, Some("ks".to_owned())),
            Self::MostlyQ5KM => ("q".to_owned(), 5, Some("km".to_owned())),
            Self::MostlyQ6K => ("q".to_owned(), 6, Some("k".to_owned())),
        }
    }

    pub fn to_quantization_level(&self) -> u8 {
        match self {
            Self::AllF32 => 32,
            Self::MostlyF16 => 16,
            Self::MostlyQ4_0 => 4,
            Self::MostlyQ4_1 => 4,
            Self::MostlyQ4_1SomeF16 => 4,
            Self::MostlyQ8_0 => 8,
            Self::MostlyQ5_0 => 5,
            Self::MostlyQ5_1 => 5,
            Self::MostlyQ2K => 2,
            Self::MostlyQ3KS => 3,
            Self::MostlyQ3KM => 3,
            Self::MostlyQ3KL => 3,
            Self::MostlyQ4KS => 4,
            Self::MostlyQ4KM => 4,
            Self::MostlyQ5KS => 5,
            Self::MostlyQ5KM => 5,
            Self::MostlyQ6K => 6,
        }
    }
}
