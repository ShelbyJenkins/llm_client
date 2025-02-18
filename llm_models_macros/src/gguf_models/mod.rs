mod model;
mod organization;
mod quant;
mod tokenizers;

use super::*;
use model::MacroGgufPreset;
use organization::{MacroPresetOrganization, MacroPresetOrganizations};
use proc_macro2::Ident;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use std::fs;

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
