use super::*;
use crate::llm::local::gguf::gguf_preset::GgufPreset;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GgufPresetLoader {
    pub llm_preset: GgufPreset,
    pub preset_with_memory_gb: Option<u64>,
    pub preset_with_memory_bytes: Option<u64>,
    pub preset_with_quantization_level: Option<u8>,
    pub inference_ctx_size: Option<u64>,
    #[serde(skip)]
    pub hf_loader: HuggingFaceLoader,
}

impl Default for GgufPresetLoader {
    fn default() -> Self {
        Self {
            llm_preset: GgufPreset::default(),
            preset_with_memory_gb: Some(8),
            preset_with_memory_bytes: None,
            preset_with_quantization_level: None,
            inference_ctx_size: None,
            hf_loader: HuggingFaceLoader::default(),
        }
    }
}

impl GgufPresetLoader {
    pub fn preset_with_quantization_level(mut self, level: u8) -> Self {
        self.preset_with_quantization_level = Some(level);
        self
    }

    pub fn preset_with_memory_gb(mut self, preset_with_memory_gb: u64) -> Self {
        self.preset_with_memory_gb = Some(preset_with_memory_gb);
        self
    }

    // Manually set, or use an estimated available value.
    pub fn preset_with_memory_bytes(mut self, preset_with_memory_bytes: u64) -> Self {
        self.preset_with_memory_bytes = Some(preset_with_memory_bytes);
        self
    }

    pub fn load(self) -> crate::Result<GgufModel> {
        if (self.preset_with_memory_bytes.is_some() || self.preset_with_memory_gb.is_some())
            && self.preset_with_quantization_level.is_some()
        {
            crate::bail!(
                "Cannot specify both memory and quantization level for GGUF preset loading."
            );
        }

        let mut inference_ctx_size = self.inference_ctx_size.unwrap_or(DEFAULT_CONTEXT_LENGTH);
        if inference_ctx_size > self.llm_preset.config.context_length {
            crate::warn!(
                "The provided context size {} is greater than the model's context size {}. Using the model's maximum context size.",
                inference_ctx_size,
                self.llm_preset.config.context_length
            );
            inference_ctx_size = self.llm_preset.config.context_length;
        }

        let local_model_path = if let Some(q_lvl) = self.preset_with_quantization_level {
            self.load_from_q_lvl(q_lvl)?
        } else if let Some(available_memory) = self.preset_with_memory_bytes {
            self.load_from_available_memory(available_memory, inference_ctx_size)?
        } else if let Some(available_vram_gb) = self.preset_with_memory_gb {
            self.load_from_available_memory(
                (available_vram_gb as f64 * 1024.0 * 1024.0 * 1024.0) as u64,
                inference_ctx_size,
            )?
        } else {
            crate::bail!("No valid loading strategy found.")
        };

        let model_metadata: LocalLlmMetadata = LocalLlmMetadata::from_gguf_path(&local_model_path)?;

        let file_name = local_model_path
            .file_stem()
            .ok_or_else(|| crate::anyhow!("Failed to get file name"))?
            .to_string_lossy()
            .to_string();

        Ok(GgufModel {
            model_base: self.llm_preset.model_base.clone(),
            organization: self.llm_preset.organization.clone(),
            model_repo_id: Some(self.llm_preset.model_repo_id.clone()),
            gguf_repo_id: Some(self.llm_preset.gguf_repo_id.clone()),
            tokenizer_path: self.llm_preset.tokenizer_path()?,
            local_model_path: local_model_path.to_owned(),
            chat_template: LlmChatTemplate::from_gguf_tokenizer(&model_metadata.tokenizer)?,
            quant: GgufQuant::new(
                &local_model_path,
                &model_metadata,
                &file_name,
                inference_ctx_size,
            )?,
            model_metadata,
        })
    }

    fn load_from_q_lvl(&self, q_lvl: u8) -> crate::Result<PathBuf> {
        let quant = self
            .llm_preset
            .select_quant_for_q_lvl(q_lvl)
            .ok_or_else(|| {
                crate::Error::msg(format!(
                    "No model file found from quantization level: {}",
                    q_lvl
                ))
            })?;

        self.hf_loader
            .load_file(quant.fname, self.llm_preset.gguf_repo_id.clone())
    }

    fn load_from_available_memory(
        &self,
        use_memory_bytes: u64,
        inference_ctx_size: u64,
    ) -> crate::Result<PathBuf> {
        let quant = self
            .llm_preset
            .select_quant_for_available_memory(Some(inference_ctx_size), use_memory_bytes)?;

        self.hf_loader
            .load_file(quant.fname, self.llm_preset.gguf_repo_id.clone())
    }
}

impl HfTokenTrait for GgufPresetLoader {
    fn hf_token_mut(&mut self) -> &mut Option<String> {
        &mut self.hf_loader.hf_token
    }

    fn hf_token_env_var_mut(&mut self) -> &mut String {
        &mut self.hf_loader.hf_token_env_var
    }
}
