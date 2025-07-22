use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GgufLocalLoader {
    local_model_path: PathBuf,
    model_id: Option<String>,
    friendly_name: Option<String>,
    organization: Option<LocalLlmOrganization>,
    gguf_repo_id: Option<String>,
    model_repo_id: Option<String>,
    local_tokenizer_path: Option<PathBuf>,
    inference_ctx_size: Option<u64>,
}

impl GgufLocalLoader {
    /// Start the builder with the mandatory model file path.
    pub fn new<P: Into<PathBuf>>(path: P) -> Self {
        Self {
            local_model_path: path.into(),
            model_id: None,
            friendly_name: None,
            local_tokenizer_path: None,
            organization: None,
            gguf_repo_id: None,
            model_repo_id: None,
            inference_ctx_size: None,
        }
    }

    /// Override the model‑id that will appear in `LlmModelBase`.
    pub fn model_id<S: Into<String>>(mut self, id: S) -> Self {
        self.model_id = Some(id.into());
        self
    }

    /// Override the friendly name that will appear in `LlmModelBase`.
    pub fn friendly_name<S: Into<String>>(mut self, name: S) -> Self {
        self.friendly_name = Some(name.into());
        self
    }

    /// Override the organization that will appear in `LlmModelBase`.
    pub fn organization(mut self, org: LocalLlmOrganization) -> Self {
        self.organization = Some(org);
        self
    }

    /// Override the GGUF repository ID that will appear in `GgufModel`.
    pub fn gguf_repo_id<S: Into<String>>(mut self, id: S) -> Self {
        self.gguf_repo_id = Some(id.into());
        self
    }

    /// Override the model repository ID that will appear in `GgufModel`.
    pub fn model_repo_id<S: Into<String>>(mut self, id: S) -> Self {
        self.model_repo_id = Some(id.into());
        self
    }

    /// Provide an explicit tokenizer JSON / GGUF path instead of using the one
    /// embedded in the model file.
    pub fn local_tokenizer_path<P: Into<PathBuf>>(mut self, p: P) -> Self {
        self.local_tokenizer_path = Some(p.into());
        self
    }

    /// Set an inference‑context size different from the model’s default.
    pub fn inference_ctx_size(mut self, n: u64) -> Self {
        self.inference_ctx_size = Some(n);
        self
    }

    /// Consume the builder and return a fully initialised [`GgufModel`].
    ///
    /// All real work is delegated to `GgufModel::from_local`, so the behaviour
    /// stays exactly the same as before.
    pub fn load(self) -> crate::Result<GgufModel> {
        let metadata = LocalLlmMetadata::from_gguf_path(&self.local_model_path)?;

        let file_name = self
            .local_model_path
            .file_stem()
            .ok_or_else(|| crate::anyhow!("missing file name"))?
            .to_string_lossy()
            .to_string();

        let model_id = if let Some(model_id) = self.model_id.as_deref() {
            model_id.to_owned()
        } else if let Some(name) = &metadata.general.name {
            name.clone()
        } else {
            file_name.clone()
        };

        let friendly_name = self
            .friendly_name
            .as_deref()
            .unwrap_or(&model_id)
            .to_owned();

        let organization = if let Some(org) = self.organization {
            org
        } else {
            let org_friendly_name = metadata.general.organization.as_deref().unwrap_or("local");
            let org_slug = metadata
                .general
                .organization
                .as_deref()
                .or_else(|| {
                    metadata
                        .general
                        .source
                        .repo_url
                        .as_deref()
                        .and_then(|s| s.split_once('/').map(|(org, _)| org))
                })
                .unwrap_or("unknown");
            LocalLlmOrganization::new(org_friendly_name, Some(org_slug))
        };

        let gguf_repo_id = if let Some(repo_id) = self.gguf_repo_id {
            Some(repo_id.into())
        } else {
            metadata
                .general
                .repo_url
                .as_deref()
                .map(|url| url.to_string().into())
        };

        let model_repo_id = if let Some(repo_id) = self.model_repo_id {
            Some(repo_id.into())
        } else {
            metadata
                .general
                .source
                .repo_url
                .as_deref()
                .map(|url| url.to_string().into())
        };
        let inference_ctx_size = self.inference_ctx_size.unwrap_or(DEFAULT_CONTEXT_LENGTH);
        Ok(GgufModel {
            model_base: LlmModelBase::new(
                &model_id,
                Some(&friendly_name),
                metadata.model_ctx_size(),
                Some(inference_ctx_size),
            ),
            organization,
            model_repo_id,
            gguf_repo_id,
            tokenizer_path: self.local_tokenizer_path.map(|path| path.to_path_buf()),
            local_model_path: self.local_model_path.to_owned(),
            chat_template: LlmChatTemplate::from_gguf_tokenizer(&metadata.tokenizer)?,
            quant: GgufQuant::new(
                &self.local_model_path,
                &metadata,
                &file_name,
                inference_ctx_size,
            )?,
            model_metadata: metadata,
        })
    }
}
