use super::*;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GgufModel {
    pub model_base: LlmModelBase,
    pub organization: LocalLlmOrganization,
    pub model_repo_id: Option<Cow<'static, str>>,
    pub gguf_repo_id: Option<Cow<'static, str>>,
    pub tokenizer_path: Option<PathBuf>,
    pub local_model_path: PathBuf,
    pub chat_template: LlmChatTemplate,
    pub quant: GgufQuant,
    pub model_metadata: LocalLlmMetadata,
}

impl Default for GgufModel {
    fn default() -> Self {
        GgufPresetLoader::default()
            .load()
            .expect("Failed to load default GGUF preset")
    }
}

impl GgufModel {
    pub fn model_id(&self) -> &str {
        &self.model_base.model_id
    }

    pub fn local_model_path(&self) -> &Path {
        &self.local_model_path
    }

    pub fn inference_ctx_size(&self) -> u64 {
        self.model_base.inference_ctx_size
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct GgufQuant {
    pub q_lvl: FileType,
    pub file_name: String,
    pub total_file_size_bytes: u64,
    //
    pub downloaded: bool,
    pub on_disk_file_size_bytes: u64,
    pub estimated_memory_usage_bytes: u64,
    //
}

impl GgufQuant {
    pub fn new(
        local_model_path: &Path,
        model_metadata: &LocalLlmMetadata,
        file_name: &str,
        inference_ctx_size: u64,
    ) -> crate::Result<Self> {
        let file_size = std::fs::metadata(&local_model_path)
            .map_err(|e| crate::anyhow!(e))?
            .len();

        let estimated_memory_usage_bytes =
            model_metadata.estimate_model_memory_usage_bytes(Some(inference_ctx_size), None) as u64;

        Ok(Self {
            q_lvl: model_metadata.general.file_type.clone(),
            file_name: file_name.to_owned(),
            total_file_size_bytes: file_size,
            downloaded: true,
            on_disk_file_size_bytes: file_size,
            estimated_memory_usage_bytes,
        })
    }
}
