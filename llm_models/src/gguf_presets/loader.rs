use super::GgufPreset;
use crate::local_models::{
    gguf::load_chat_template, hf_loader::HuggingFaceLoader, metadata::LocalLlmMetadata,
    LocalLlmModel,
};

// Feature-specific internal imports
#[cfg(feature = "model-tokenizers")]
use crate::local_models::gguf::load_tokenizer;

pub(crate) const DEFAULT_PRESET_CONTEXT_LENGTH: usize = 4096;

#[derive(Clone)]
pub struct GgufPresetLoader {
    pub llm_preset: GgufPreset,
    pub preset_with_memory_gb: Option<usize>,
    pub preset_with_memory_bytes: Option<usize>,
    pub preset_with_max_ctx_size: Option<usize>,
    pub preset_with_quantization_level: Option<u8>,
}

impl Default for GgufPresetLoader {
    fn default() -> Self {
        Self {
            llm_preset: GgufPreset::default(),
            preset_with_memory_gb: None,
            preset_with_memory_bytes: None,
            preset_with_max_ctx_size: None,
            preset_with_quantization_level: None,
        }
    }
}

impl GgufPresetLoader {
    pub fn load(&mut self, hf_loader: &HuggingFaceLoader) -> crate::Result<LocalLlmModel> {
        let file_name = self.select_quant()?;

        let local_model_filename = hf_loader.load_file(file_name, self.llm_preset.gguf_repo_id)?;

        let local_model_path = HuggingFaceLoader::canonicalize_local_path(local_model_filename)?;

        let model_metadata = LocalLlmMetadata::from_gguf_path(&local_model_path)?;
        Ok(LocalLlmModel {
            model_base: crate::LlmModelBase {
                model_id: self.llm_preset.model_id.to_owned(),
                friendly_name: self.llm_preset.friendly_name.to_owned(),
                model_ctx_size: model_metadata.context_length(),
                inference_ctx_size: model_metadata.context_length(),
                #[cfg(feature = "model-tokenizers")]
                tokenizer: load_tokenizer(&self.llm_preset.tokenizer_path(), &model_metadata)?,
            },
            chat_template: load_chat_template(&model_metadata)?,
            model_metadata,
            local_model_path,
        })
    }

    fn select_quant(&mut self) -> crate::Result<String> {
        let ctx_size = if let Some(preset_with_max_ctx_size) = self.preset_with_max_ctx_size {
            // If the preset_with_max_ctx_size is set and greater than the model's context_length, we use the model's context_length.
            if preset_with_max_ctx_size > self.llm_preset.config.context_length {
                self.preset_with_max_ctx_size = Some(self.llm_preset.config.context_length);
                self.llm_preset.config.context_length
            } else {
                preset_with_max_ctx_size
            }
        } else {
            // If the preset_with_max_ctx_size is not set, we use DEFAULT_PRESET_CONTEXT_LENGTH
            self.preset_with_max_ctx_size = Some(DEFAULT_PRESET_CONTEXT_LENGTH);
            DEFAULT_PRESET_CONTEXT_LENGTH
        };

        if self.preset_with_memory_gb.is_none()
            && self.preset_with_memory_bytes.is_none()
            && self.preset_with_quantization_level.is_none()
        {
            self.preset_with_quantization_level = Some(8);
        };

        let file_name: String = if let Some(q_bits) = self.preset_with_quantization_level {
            if let Some(file_name) = self.llm_preset.quant_file_name_for_q_bit(q_bits) {
                file_name.to_owned()
            } else {
                crate::bail!(
                    "No model file found for given quantization level: {}",
                    q_bits
                );
            }
        } else {
            let available_ram_bytes =
                if let Some(available_vram_bytes) = self.preset_with_memory_bytes {
                    available_vram_bytes
                } else if let Some(available_vram_gb) = self.preset_with_memory_gb {
                    (available_vram_gb as f64 * 1024.0 * 1024.0 * 1024.0) as usize
                } else {
                    crate::bail!("No VRAM provided!")
                };
            let q_bits = self
                .llm_preset
                .select_quant_for_available_memory(Some(ctx_size), available_ram_bytes as u64)?;
            if let Some(file_name) = self.llm_preset.quant_file_name_for_q_bit(q_bits) {
                file_name.to_owned()
            } else {
                crate::bail!(
                    "No model file found for given quantization level: {}",
                    q_bits
                );
            }
        };

        Ok(file_name)
    }
}
