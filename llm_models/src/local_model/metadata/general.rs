use crate::local_model::gguf::tools::gguf_tensors::GgmlDType;

#[derive(Clone)]
pub struct GeneralMetadata {
    // Required fields
    pub architecture: String,
    pub quantization_version: u32,
    pub alignment: u32,

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
    pub file_type: Option<FileType>,

    // Source metadata
    pub source: SourceMetadata,
}

#[derive(Clone)]
pub struct SourceMetadata {
    pub url: Option<String>,
    pub doi: Option<String>,
    pub uuid: Option<String>,
    pub repo_url: Option<String>,
}

#[derive(Debug, Clone, Copy)]
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
}

impl GeneralMetadata {
    pub fn from_gguf(
        gguf: &crate::local_model::gguf::tools::gguf_file::GgufFile,
    ) -> crate::Result<Self> {
        let file_type: Option<u32> = gguf.get_value("general.file_type")?;
        let file_type = file_type.map(|u| FileType::from_u32(u)).transpose()?;
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
            file_type,
            source: SourceMetadata::from_gguf(gguf)?,
        })
    }
}

impl SourceMetadata {
    pub fn from_gguf(
        gguf: &crate::local_model::gguf::tools::gguf_file::GgufFile,
    ) -> crate::Result<Self> {
        Ok(Self {
            url: gguf.get_value("general.source.url")?,
            doi: gguf.get_value("general.source.doi")?,
            uuid: gguf.get_value("general.source.uuid")?,
            repo_url: gguf.get_value("general.source.repo_url")?,
        })
    }
}

impl std::fmt::Debug for GeneralMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("GeneralMetadata");

        // Helper macro to add fields
        macro_rules! add_field {
            ($field:ident) => {
                match &self.$field {
                    Some(value) => debug_struct.field(stringify!($field), value),
                    None => &mut debug_struct,
                };
            };
        }

        debug_struct.field("architecture", &self.architecture);
        debug_struct.field("quantization_version", &self.quantization_version);
        debug_struct.field("alignment", &self.alignment);

        add_field!(name);
        add_field!(author);
        add_field!(version);
        add_field!(organization);
        add_field!(basename);
        add_field!(finetune);
        add_field!(description);
        add_field!(quantized_by);
        add_field!(size_label);
        add_field!(license);
        add_field!(license_name);
        add_field!(license_link);
        add_field!(url);
        add_field!(doi);
        add_field!(uuid);
        add_field!(repo_url);
        add_field!(tags);
        add_field!(languages);
        add_field!(datasets);
        add_field!(file_type);

        debug_struct.field("source", &self.source);

        debug_struct.finish()
    }
}

impl std::fmt::Debug for SourceMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("SourceMetadata");

        macro_rules! add_field {
            ($field:ident) => {
                if let Some(value) = &self.$field {
                    debug_struct.field(stringify!($field), value);
                }
            };
        }

        add_field!(url);
        add_field!(doi);
        add_field!(uuid);
        add_field!(repo_url);

        debug_struct.finish()
    }
}
