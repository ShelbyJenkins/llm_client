use super::{
    hf_loader::HuggingFaceLoader, metadata::LocalLlmMetadata, GgufPresetTrait, HfTokenTrait,
    LlmChatTemplate, LocalLlmModel,
};
use crate::tokenizer::LlmTokenizer;
use loaders::{hf::GgufHfLoader, local::GgufLocalLoader, preset::GgufPresetLoader};
use tools::gguf_tokenizer::convert_gguf_to_hf_tokenizer;

pub mod loaders;
pub mod memory;
pub mod preset;
pub mod tools;

#[derive(Default, Clone)]
pub struct GgufLoader {
    pub gguf_preset_loader: GgufPresetLoader,
    pub gguf_local_loader: GgufLocalLoader,
    pub gguf_hf_loader: GgufHfLoader,
    pub hf_loader: HuggingFaceLoader,
}

impl GgufLoader {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn load(&mut self) -> crate::Result<LocalLlmModel> {
        if self.gguf_local_loader.local_quant_file_path.is_some() {
            return self.gguf_local_loader.load();
        } else if self.gguf_hf_loader.hf_quant_file_url.is_some() {
            return self.gguf_hf_loader.load(&self.hf_loader);
        } else {
            return self.gguf_preset_loader.load(&self.hf_loader);
        }
    }
}

impl GgufLoaderTrait for GgufLoader {
    fn gguf_loader(&mut self) -> &mut GgufLoader {
        self
    }
}

impl HfTokenTrait for GgufLoader {
    fn hf_token_mut(&mut self) -> &mut Option<String> {
        &mut self.hf_loader.hf_token
    }

    fn hf_token_env_var_mut(&mut self) -> &mut String {
        &mut self.hf_loader.hf_token_env_var
    }
}

impl GgufPresetTrait for GgufLoader {
    fn preset_loader(&mut self) -> &mut GgufPresetLoader {
        &mut self.gguf_preset_loader
    }
}

pub(crate) fn load_tokenizer(
    local_tokenizer_path: &Option<std::path::PathBuf>,
    model_metadata: &LocalLlmMetadata,
) -> crate::Result<std::sync::Arc<LlmTokenizer>> {
    if let Some(local_tokenizer_path) = &local_tokenizer_path {
        Ok(std::sync::Arc::new(LlmTokenizer::new_from_tokenizer_json(
            local_tokenizer_path,
        )?))
    } else {
        if let Some(ggml) = &model_metadata.tokenizer.ggml {
            let tokenizer = convert_gguf_to_hf_tokenizer(&ggml)?;
            Ok(std::sync::Arc::new(LlmTokenizer::new_from_tokenizer(
                tokenizer,
            )?))
        } else {
            crate::bail!("No tokenizer found in model metadata")
        }
    }
}

pub(crate) fn load_chat_template(
    local_tokenizer_config_path: &Option<std::path::PathBuf>,
    model_metadata: &LocalLlmMetadata,
) -> crate::Result<LlmChatTemplate> {
    if let Some(local_tokenizer_config_path) = local_tokenizer_config_path {
        LlmChatTemplate::from_local_path(local_tokenizer_config_path)
    } else {
        LlmChatTemplate::from_gguf_tokenizer(&model_metadata.tokenizer)
    }
}

pub trait GgufLoaderTrait {
    fn gguf_loader(&mut self) -> &mut GgufLoader;

    /// Sets the model id for the model config. Used for display purposes and debugging.
    /// Optional because this can be loaded from the URL, file path, or preset.
    fn model_id<S: AsRef<str>>(&mut self, model_id: S) -> &mut Self {
        self.gguf_loader().gguf_hf_loader.model_id = Some(model_id.as_ref().into());
        self.gguf_loader().gguf_local_loader.model_id = Some(model_id.as_ref().into());
        self
    }

    /// Sets the local path to the quantized model file.
    /// Use the /full/path/and/filename.gguf
    fn local_quant_file_path<S: Into<std::path::PathBuf>>(
        &mut self,
        local_quant_file_path: S,
    ) -> &mut Self {
        self.gguf_loader().gguf_local_loader.local_quant_file_path =
            Some(local_quant_file_path.into());
        self
    }

    /// Sets the local path to the model config.json file.
    /// Optional because this can be loaded from the GGUF file.
    fn local_config_path<P: AsRef<std::path::Path>>(&mut self, local_config_path: P) -> &mut Self {
        self.gguf_loader().gguf_local_loader.local_config_path =
            Some(local_config_path.as_ref().to_owned());
        self
    }

    /// Sets the local path to the tokenizer.json file.
    /// Optional because this can be loaded from the GGUF file.
    fn local_tokenizer_path<P: AsRef<std::path::Path>>(
        &mut self,
        local_tokenizer_path: P,
    ) -> &mut Self {
        self.gguf_loader().gguf_local_loader.local_tokenizer_path =
            Some(local_tokenizer_path.as_ref().to_owned());
        self
    }

    /// Sets the local path to the tokenizer_config.json file.
    /// Optional because this can be loaded from the GGUF file.
    fn local_tokenizer_config_path<P: AsRef<std::path::Path>>(
        &mut self,
        local_tokenizer_config_path: P,
    ) -> &mut Self {
        self.gguf_loader()
            .gguf_local_loader
            .local_tokenizer_config_path = Some(local_tokenizer_config_path.as_ref().to_owned());
        self
    }

    /// Sets the Hugging Face url to the quantized model file.
    /// The full url to the model on hugging face like:
    /// 'https://huggingface.co/MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF/blob/main/Meta-Llama-3-8B-Instruct.Q6_K.gguf'
    fn hf_quant_file_url<S: Into<String>>(&mut self, hf_quant_file_url: S) -> &mut Self {
        self.gguf_loader().gguf_hf_loader.hf_quant_file_url = Some(hf_quant_file_url.into());
        self
    }

    /// Sets the Hugging Face repo id to the model config.json file.
    /// Optional because this can be loaded from the GGUF file.
    fn hf_config_repo_id<S: Into<String>>(&mut self, hf_config_repo_id: S) -> &mut Self {
        self.gguf_loader().gguf_hf_loader.hf_config_repo_id = Some(hf_config_repo_id.into());
        self
    }

    /// Sets the Hugging Face repo id for the tokenizer. This is used for loading the tokenizer.
    /// Optional because this can be loaded from the GGUF file.
    fn hf_tokenizer_repo_id<S: Into<String>>(&mut self, hf_tokenizer_repo_id: S) -> &mut Self {
        self.gguf_loader().gguf_hf_loader.hf_tokenizer_repo_id = Some(hf_tokenizer_repo_id.into());
        self
    }

    /// Sets the Hugging Face repo id for the tokenizer config. This is used for loading the chat template.
    /// Optional because this can be loaded from the GGUF file.
    fn hf_tokenizer_config_repo_id<S: Into<String>>(
        &mut self,
        hf_tokenizer_config_repo_id: S,
    ) -> &mut Self {
        self.gguf_loader()
            .gguf_hf_loader
            .hf_tokenizer_config_repo_id = Some(hf_tokenizer_config_repo_id.into());
        self
    }
}
