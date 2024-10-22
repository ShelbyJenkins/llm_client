use crate::local_model::{
    gguf::{
        load_chat_template, load_tokenizer, memory::estimate_quantization_level, preset::LlmPreset,
    },
    hf_loader::HuggingFaceLoader,
    metadata::LocalLlmMetadata,
    LocalLlmModel,
};
pub(crate) const DEFAULT_PRESET_CONTEXT_LENGTH: u64 = 4096;

#[derive(Clone)]
pub struct GgufPresetLoader {
    pub llm_preset: LlmPreset,
    pub preset_with_available_vram_gb: Option<u32>,
    pub preset_with_available_vram_bytes: Option<u64>,
    pub preset_with_max_ctx_size: Option<u64>,
    pub preset_with_quantization_level: Option<u8>,
}

impl Default for GgufPresetLoader {
    fn default() -> Self {
        Self {
            llm_preset: LlmPreset::Llama3_1_8bInstruct,
            preset_with_available_vram_gb: None,
            preset_with_available_vram_bytes: None,
            preset_with_max_ctx_size: None,
            preset_with_quantization_level: None,
        }
    }
}

impl GgufPresetLoader {
    pub fn load(&mut self, hf_loader: &HuggingFaceLoader) -> crate::Result<LocalLlmModel> {
        println!("{}", self.llm_preset.model_id());
        let file_name = self.select_quant()?;

        let local_model_filename =
            hf_loader.load_file(file_name, self.llm_preset.gguf_repo_id())?;

        let local_model_path = HuggingFaceLoader::canonicalize_local_path(local_model_filename)?;

        let model_metadata = LocalLlmMetadata::from_gguf_path(&local_model_path)?;
        Ok(LocalLlmModel {
            model_base: crate::LlmModelBase {
                model_id: self.llm_preset.model_id(),
                model_ctx_size: model_metadata.context_length(),
                inference_ctx_size: model_metadata.context_length(),
                tokenizer: load_tokenizer(
                    &Some(self.llm_preset.load_tokenizer(hf_loader)?),
                    &model_metadata,
                )?,
            },
            chat_template: load_chat_template(
                &Some(self.llm_preset.load_tokenizer_config(hf_loader)?),
                &model_metadata,
            )?,
            model_metadata,
            local_model_path,
        })
    }

    fn select_quant(&mut self) -> crate::Result<String> {
        let config_json = self.llm_preset.config_json()?;

        let ctx_size = if let Some(preset_with_max_ctx_size) = self.preset_with_max_ctx_size {
            // If the preset_with_max_ctx_size is set and greater than the model's context_length, we use the model's context_length.
            if preset_with_max_ctx_size > config_json.context_length {
                self.preset_with_max_ctx_size = Some(config_json.context_length);
                config_json.context_length
            } else {
                preset_with_max_ctx_size
            }
        } else {
            // If the preset_with_max_ctx_size is not set, we use DEFAULT_PRESET_CONTEXT_LENGTH
            self.preset_with_max_ctx_size = Some(DEFAULT_PRESET_CONTEXT_LENGTH);
            DEFAULT_PRESET_CONTEXT_LENGTH
        };

        if self.preset_with_available_vram_gb.is_none()
            && self.preset_with_available_vram_bytes.is_none()
            && self.preset_with_quantization_level.is_none()
        {
            self.preset_with_quantization_level = Some(8);
        };

        let file_name: String = if let Some(q_bits) = self.preset_with_quantization_level {
            if let Some(file_name) = self.llm_preset.f_name_for_q_bits(q_bits) {
                file_name
            } else {
                crate::bail!(
                    "No model file found for given quantization level: {}",
                    q_bits
                );
            }
        } else {
            let ctx_memory_size_bytes = config_json.estimate_context_size(ctx_size as u64);

            let initial_q_bits = estimate_quantization_level(
                self.llm_preset.number_of_parameters(),
                self.preset_with_available_vram_bytes,
                self.preset_with_available_vram_gb,
                ctx_memory_size_bytes,
            )?;
            let mut q_bits = initial_q_bits;
            loop {
                if let Some(file_name) = self.llm_preset.f_name_for_q_bits(q_bits) {
                    break file_name;
                }
                if q_bits == 1 {
                    crate::bail!(
                        "No model file found from quantization levels: {initial_q_bits}-{q_bits}",
                    );
                } else {
                    q_bits -= 1;
                }
            }
        };

        Ok(file_name)
    }
}
