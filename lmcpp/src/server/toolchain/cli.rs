use crate::server::toolchain::builder::{ComputeBackendConfig, LmcppBuildInstallMode};

/// Options that uniquely identify *one* variant of the tool‑chain.
/// Re‑used (`flatten`ed) by every sub‑command so there is no duplication.
#[derive(Debug, clap::Args)]
pub struct RecipeSpec {
    /// Git tag or commit to fetch; falls back to library default.
    #[arg(long)]
    pub repo_tag: Option<String>,

    /// CPU, CUDA, Metal, …
    #[arg(long, default_value = "default", value_enum)]
    pub backend: ComputeBackendConfig,

    /// Build / install strategy.
    #[arg(long, default_value = "build-or-install", value_enum)]
    pub mode: LmcppBuildInstallMode,

    /// Extra CMake flags (repeatable).
    #[arg(long, num_args = 1..)]
    pub build_args: Vec<String>,
}
