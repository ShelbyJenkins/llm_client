use crate::{
    local_model::{
        gguf::{load_chat_template, load_tokenizer},
        metadata::LocalLlmMetadata,
        LocalLlmModel,
    },
    LlmModelBase,
};

#[derive(Default, Clone)]
pub struct GgufLocalLoader {
    pub local_quant_file_path: Option<std::path::PathBuf>,
    pub local_config_path: Option<std::path::PathBuf>,
    pub local_tokenizer_path: Option<std::path::PathBuf>,
    pub local_tokenizer_config_path: Option<std::path::PathBuf>,
    pub model_id: Option<String>,
}

impl GgufLocalLoader {
    pub fn load(&mut self) -> crate::Result<LocalLlmModel> {
        let local_model_path =
            if let Some(local_quant_file_path) = self.local_quant_file_path.as_ref() {
                local_quant_file_path.to_owned()
            } else {
                crate::bail!("local_quant_file_path must be set")
            };

        let model_id = if let Some(model_id) = &self.model_id {
            model_id.to_owned()
        } else {
            local_model_path.to_string_lossy().to_string()
        };

        let model_metadata = LocalLlmMetadata::from_gguf_path(&local_model_path)?;

        Ok(LocalLlmModel {
            model_base: LlmModelBase {
                model_id,
                model_ctx_size: model_metadata.context_length(),
                inference_ctx_size: model_metadata.context_length(),
                tokenizer: load_tokenizer(&self.local_tokenizer_path, &model_metadata)?,
            },
            chat_template: load_chat_template(&self.local_tokenizer_config_path, &model_metadata)?,
            model_metadata,
            local_model_path,
        })
    }
}
