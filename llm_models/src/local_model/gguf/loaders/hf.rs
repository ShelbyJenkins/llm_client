use crate::{
    local_model::{hf_loader::HuggingFaceLoader, metadata::LocalLlmMetadata, LocalLlmModel},
    LlmModelBase,
};

#[derive(Default, Clone)]
pub struct GgufHfLoader {
    pub hf_quant_file_url: Option<String>,
    pub hf_config_repo_id: Option<String>,
    pub hf_tokenizer_repo_id: Option<String>,
    pub hf_tokenizer_config_repo_id: Option<String>,
    pub model_id: Option<String>,
}

impl GgufHfLoader {
    pub fn load(&mut self, hf_loader: &HuggingFaceLoader) -> crate::Result<LocalLlmModel> {
        let hf_quant_file_url = if let Some(hf_quant_file_url) = self.hf_quant_file_url.as_ref() {
            hf_quant_file_url.to_owned()
        } else {
            crate::bail!("local_quant_file_path must be set")
        };

        let (model_id, repo_id, gguf_model_filename) =
            HuggingFaceLoader::parse_full_model_url(&hf_quant_file_url);

        let local_model_path = HuggingFaceLoader::canonicalize_local_path(
            hf_loader.load_file(gguf_model_filename, repo_id)?,
        )?;

        let local_tokenizer_path = if let Some(hf_tokenizer_repo_id) = &self.hf_tokenizer_repo_id {
            self.try_load_config(hf_loader, hf_tokenizer_repo_id, "tokenizer.json")
        } else if let Some(hf_config_repo_id) = &self.hf_config_repo_id {
            self.try_load_config(hf_loader, hf_config_repo_id, "tokenizer.json")
        } else {
            None
        };
        let local_tokenizer_config_path =
            if let Some(hf_tokenizer_config_repo_id) = &self.hf_tokenizer_config_repo_id {
                self.try_load_config(
                    hf_loader,
                    hf_tokenizer_config_repo_id,
                    "tokenizer_config.json",
                )
            } else if let Some(hf_config_repo_id) = &self.hf_config_repo_id {
                self.try_load_config(hf_loader, hf_config_repo_id, "tokenizer_config.json")
            } else {
                None
            };
        let model_metadata = LocalLlmMetadata::from_gguf_path(&local_model_path)?;

        Ok(LocalLlmModel {
            model_base: LlmModelBase {
                model_id,
                model_ctx_size: model_metadata.context_length(),
                inference_ctx_size: model_metadata.context_length(),
                tokenizer: crate::local_model::gguf::load_tokenizer(
                    &local_tokenizer_path,
                    &model_metadata,
                )?,
            },
            chat_template: crate::local_model::gguf::load_chat_template(
                &local_tokenizer_config_path,
                &model_metadata,
            )?,
            model_metadata,
            local_model_path,
        })
    }

    fn try_load_config(
        &self,
        hf_loader: &HuggingFaceLoader,
        repo_id: &str,
        file: &str,
    ) -> Option<std::path::PathBuf> {
        match hf_loader.load_file(file, repo_id) {
            Ok(path) => Some(path),
            Err(e) => {
                eprintln!("Failed to load {}: {}", file, e);
                None
            }
        }
    }
}
