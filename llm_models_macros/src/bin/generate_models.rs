use llm_models_macros::generate_api_providers_and_models;
use llm_models_macros::generate_local_organizations_and_models;

fn main() {
    let api_providers_and_models_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("Failed to get parent directory")
        .join("llm_models")
        .join("src")
        .join("api_models");

    generate_api_providers_and_models(api_providers_and_models_path);

    let orgs_and_gguf_presets_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("Failed to get parent directory")
        .join("llm_models")
        .join("src")
        .join("gguf_presets");

    generate_local_organizations_and_models(orgs_and_gguf_presets_path)
}
