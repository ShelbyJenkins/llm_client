mod data;
mod model;
mod provider;

use super::*;
use data::*;
use model::DeApiLlmPresets;
use model::MacroApiLlmPreset;
use proc_macro2::Ident;
use proc_macro2::TokenStream;
use provider::MacroApiLlmProvider;
use quote::{format_ident, quote};

pub fn generate_api_providers_and_models(output_path: std::path::PathBuf) {
    model::generate(&output_path);
    provider::generate(&output_path);
}
