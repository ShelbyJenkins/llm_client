//! Downloads to Path: "/root/.cache/huggingface/hub/
use anyhow::{anyhow, Result};
use dotenvy::dotenv;
use hf_hub::api::sync::{Api, ApiBuilder};
use std::{cell::OnceCell, path::PathBuf};

const DEFAULT_ENV_VAR: &str = "HUGGING_FACE_TOKEN";

#[derive(Clone)]
pub struct HuggingFaceLoader {
    pub hf_token: Option<String>,
    pub hf_token_env_var: String,
    pub hf_api: OnceCell<Api>,
}

impl Default for HuggingFaceLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl HuggingFaceLoader {
    pub fn new() -> Self {
        Self {
            hf_token: None,
            hf_token_env_var: DEFAULT_ENV_VAR.to_string(),
            hf_api: OnceCell::new(),
        }
    }

    pub fn hf_api(&self) -> &Api {
        self.hf_api.get_or_init(|| {
            ApiBuilder::new()
                .with_progress(true)
                .with_token(self.load_hf_token())
                .build()
                .expect("Failed to build Hugging Face API")
        })
    }

    fn load_hf_token(&self) -> Option<String> {
        if let Some(hf_token) = &self.hf_token {
            println!("Using hf_token from parameter");
            return Some(hf_token.to_owned());
        }

        dotenv().ok();

        match dotenvy::var(&self.hf_token_env_var) {
            Ok(hf_token) => Some(hf_token),
            Err(_) => {
                println!(
                    "{} not found in dotenv, nor was it set manually",
                    self.hf_token_env_var
                );
                None
            }
        }
    }

    pub fn load_file<T: AsRef<str>, S: Into<String>>(
        &self,
        file_name: T,
        repo_id: S,
    ) -> Result<PathBuf> {
        self.hf_api()
            .model(repo_id.into())
            .get(file_name.as_ref())
            .map_err(|e| anyhow!(e))
    }

    pub fn load_model_safe_tensors<S: Into<String>>(&self, repo_id: S) -> Result<Vec<PathBuf>> {
        let repo_id = repo_id.into();
        let mut safe_tensor_filenames = vec![];
        let siblings = self.hf_api().model(repo_id.clone()).info()?.siblings;
        for sib in siblings {
            if sib.rfilename.ends_with(".safetensors") {
                safe_tensor_filenames.push(sib.rfilename);
            }
        }
        let mut safe_tensor_paths = vec![];
        for safe_tensor_filename in &safe_tensor_filenames {
            let safe_tensor_path = self
                .hf_api()
                .model(repo_id.clone())
                .get(safe_tensor_filename)
                .map_err(|e| anyhow!(e))?;
            let safe_tensor_path = Self::canonicalize_local_path(safe_tensor_path)?;
            println!("Downloaded safe tensor: {:?}", safe_tensor_path);
            safe_tensor_paths.push(safe_tensor_path);
        }
        Ok(safe_tensor_paths)
    }

    pub fn canonicalize_local_path(local_path: PathBuf) -> Result<PathBuf> {
        local_path.canonicalize().map_err(|e| anyhow!(e))
    }

    pub fn parse_full_model_url(model_url: &str) -> (String, String, String) {
        if !model_url.starts_with("https://huggingface.co") {
            panic!("URL does not start with https://huggingface.co\n Format should be like: https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF/blob/main/zephyr-7b-alpha.Q8_0.gguf");
        } else if !model_url.ends_with(".gguf") {
            panic!("URL does not end with .gguf\n Format should be like: https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF/blob/main/zephyr-7b-alpha.Q8_0.gguf");
        } else {
            let parts: Vec<&str> = model_url.split('/').collect();
            if parts.len() < 5 {
                panic!("URL does not have enough parts\n Format should be like: https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF/blob/main/zephyr-7b-alpha.Q8_0.gguf");
            }
            let model_id = parts[4].to_string();
            let repo_id = format!("{}/{}", parts[3], parts[4]);
            let gguf_model_filename = parts.last().unwrap_or(&"").to_string();
            (model_id, repo_id, gguf_model_filename)
        }
    }

    pub fn model_url_from_repo_and_local_filename(
        repo_id: &str,
        local_model_filename: &str,
    ) -> String {
        let filename = std::path::Path::new(local_model_filename)
            .file_name()
            .and_then(|os_str| os_str.to_str())
            .unwrap_or(local_model_filename);

        format!("https://huggingface.co/{}/blob/main/{}", repo_id, filename)
    }

    pub fn model_url_from_repo(repo_id: &str) -> String {
        format!("https://huggingface.co/{}", repo_id)
    }

    pub fn model_id_from_url(model_url: &str) -> String {
        let parts = Self::parse_full_model_url(model_url);
        parts.0
    }
}

impl HfTokenTrait for HuggingFaceLoader {
    fn hf_token_mut(&mut self) -> &mut Option<String> {
        &mut self.hf_token
    }

    fn hf_token_env_var_mut(&mut self) -> &mut String {
        &mut self.hf_token_env_var
    }
}

pub trait HfTokenTrait {
    fn hf_token_mut(&mut self) -> &mut Option<String>;

    fn hf_token_env_var_mut(&mut self) -> &mut String;

    /// Set the API key for the client. Otherwise it will attempt to load it from the .env file.
    fn hf_token<S: Into<String>>(mut self, hf_token: S) -> Self
    where
        Self: Sized,
    {
        *self.hf_token_mut() = Some(hf_token.into());
        self
    }

    /// Set the environment variable name for the API key. Default is "HUGGING_FACE_TOKEN".
    fn hf_token_env_var<S: Into<String>>(mut self, hf_token_env_var: S) -> Self
    where
        Self: Sized,
    {
        *self.hf_token_env_var_mut() = hf_token_env_var.into();
        self
    }
}
