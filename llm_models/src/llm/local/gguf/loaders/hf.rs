use super::*;

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct GgufHfLoader {
    #[serde(skip)]
    pub hf_loader: HuggingFaceLoader,
    pub hf_quant_file_url: Option<String>,
    pub hf_tokenizer_repo_id: Option<String>,
    pub model_id: Option<String>,
    pub inference_ctx_size: Option<u64>,
}

impl GgufHfLoader {
    /// Sets the Hugging Face url to the quantized model file.
    /// The full url to the model on hugging face like:
    /// 'https://huggingface.co/MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF/blob/main/Meta-Llama-3-8B-Instruct.Q6_K.gguf'
    pub fn hf_quant_file_url<S: Into<String>>(&mut self, hf_quant_file_url: S) -> &mut Self {
        self.hf_quant_file_url = Some(hf_quant_file_url.into());
        self
    }

    /// Sets the Hugging Face repo id for the tokenizer. This is used for loading the tokenizer.
    /// Optional because this can be loaded from the GGUF file.
    pub fn hf_tokenizer_repo_id<S: Into<String>>(&mut self, hf_tokenizer_repo_id: S) -> &mut Self {
        self.hf_tokenizer_repo_id = Some(hf_tokenizer_repo_id.into());
        self
    }

    /// Override the model‑id that will appear in `LlmModelBase`.
    pub fn model_id<S: Into<String>>(mut self, id: S) -> Self {
        self.model_id = Some(id.into());
        self
    }

    /// Set an inference‑context size different from the model’s default.
    pub fn inference_ctx_size(mut self, n: u64) -> Self {
        self.inference_ctx_size = Some(n);
        self
    }

    pub fn load(self) -> crate::Result<GgufModel> {
        let hf_quant_file_url = if let Some(hf_quant_file_url) = self.hf_quant_file_url.as_ref() {
            hf_quant_file_url.to_owned()
        } else {
            crate::bail!("hf_quant_file_url must be set")
        };

        let (model_id, repo_id, gguf_model_filename) =
            HuggingFaceLoader::parse_full_model_url(&hf_quant_file_url);

        let local_model_path = self.hf_loader.load_file(gguf_model_filename, repo_id)?;

        let mut loader = GgufLocalLoader::new(local_model_path).model_id(model_id);
        if let Some(inference_ctx_size) = self.inference_ctx_size {
            loader = loader.inference_ctx_size(inference_ctx_size);
        }

        if let Some(hf_tokenizer_repo_id) = &self.hf_tokenizer_repo_id {
            match self
                .hf_loader
                .load_file("tokenizer.json", hf_tokenizer_repo_id)
            {
                Ok(path) => loader = loader.local_tokenizer_path(path),
                Err(e) => {
                    crate::error!("Failed to load tokenizer.json: {}", e);
                }
            }
        };

        loader.load()
    }
}

impl HfTokenTrait for GgufHfLoader {
    fn hf_token_mut(&mut self) -> &mut Option<String> {
        &mut self.hf_loader.hf_token
    }

    fn hf_token_env_var_mut(&mut self) -> &mut String {
        &mut self.hf_loader.hf_token_env_var
    }
}
