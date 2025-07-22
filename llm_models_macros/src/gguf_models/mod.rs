mod model;
mod organization;
mod quant;
mod tokenizers;

use super::*;
use dotenvy::dotenv;
use hf_hub::{api::sync::Api, api::sync::ApiBuilder, Cache};
use model::MacroGgufPreset;
use organization::{MacroPresetOrganization, MacroPresetOrganizations};
use proc_macro2::Ident;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use std::fs;
use std::sync::{LazyLock, OnceLock};

const PATH_TO_ORGS_DATA_DIR: std::sync::LazyLock<std::path::PathBuf> =
    std::sync::LazyLock::new(|| {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("gguf_models")
            .join("data")
    });

pub fn generate_local_organizations_and_models(output_path: std::path::PathBuf) {
    model::generate(&output_path);
    organization::generate(&output_path);
    tokenizers::generate(&output_path);
}

static HF_CACHE: LazyLock<Cache> = LazyLock::new(|| Cache::default());
const DEFAULT_ENV_VAR: &str = "HUGGING_FACE_TOKEN";
static HF_API: OnceLock<Api> = OnceLock::new();

pub fn hf_api() -> &'static Api {
    HF_API.get_or_init(|| {
        ApiBuilder::from_cache(HF_CACHE.clone())
            .with_progress(true)
            .with_token(load_hf_token())
            .build()
            .expect("Failed to build Hugging Face API")
    })
}

fn load_hf_token() -> Option<String> {
    dotenv().ok();

    match dotenvy::var(DEFAULT_ENV_VAR) {
        Ok(hf_token) => Some(hf_token),
        Err(e) => {
            panic!(
                "Failed to load Hugging Face token from environment variable {}: {}",
                DEFAULT_ENV_VAR, e
            )
        }
    }
}

pub fn load_file(file_name: &str, repo_id: &str) -> std::path::PathBuf {
    match hf_api().model(repo_id.to_owned()).get(file_name) {
        Ok(path) => path,
        Err(e) => {
            panic!(
                "Failed to load file {} from repo {}: {}",
                file_name, repo_id, e
            )
        }
    }
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct HuggingFaceRepoInfo {
    pub siblings: Vec<HuggingFaceSibling>,
}

impl HuggingFaceRepoInfo {
    pub fn get_file(&self, filename: &str) -> Option<&HuggingFaceSibling> {
        self.siblings.iter().find(|sib| sib.rfilename == filename)
    }
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct HuggingFaceSibling {
    pub rfilename: String,
    pub size: u64,
}
