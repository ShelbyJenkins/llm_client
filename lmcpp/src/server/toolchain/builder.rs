//! High-level *public* API for obtaining `llama-server`
//! ====================================================
//!
//! This file exposes the **one-stop** interface a caller needs to
//! *download*, *build*, *cache*, *validate*, or *remove* a fully-functional
//! **llama.cpp** server binary.  Everything lower-level lives in
//! [`recipe.rs`]; consumers should interact **only** with the types re-exported
//! here.
//!
//! # Core abstraction
//!
//! | Type | Purpose |
//! |------|---------|
//! | [`LmcppToolChain`]          | The *fluent builder* capturing every input (repo-tag, backend, build flags, …). |
//! | [`ComputeBackendConfig`]   | Caller’s *intent* – “CUDA if you can, CPU otherwise”. |
//! | [`ComputeBackend`]         | The *resolved* backend chosen after runtime inspection. |
//! | [`LmcppBuildInstallMode`]   | Strategy: always build, always install, or “try build then install”. |
//! | [`LmcppToolchainOutcome`]   | Rich, serialisable summary: status, duration, paths. |
//!
//! All public structs and enums derive **`Serialize`/`Deserialize`** so the
//! chosen configuration can be stored (e.g. `~/.config/myapp.toml`) or shipped
//! over IPC unchanged.
//!
//! # Typical lifecycle
//!
//! ```rust,ignore
//! fn main() -> Result<()> {
//!     // 1. Describe what we want
//!     let outcome = LmcppToolChain::new()                    // defaults are sane
//!         .repo_tag("b5890")                                // pick a ggml-org tag
//!         .compute_backend(ComputeBackendConfig::CudaIfAvailable)
//!         .build_or_install()                               // strategy
//!         .run()?;                                          // 2. Execute
//!
//!     // 3. Use the server
//!     std::process::Command::new(outcome.bin_path()?).spawn()?;
//!     Ok(())
//! }
//! ```
//!
//! # Concurrency & safety
//!
//! * The underlying recipe enforces **cross-process** and **cross-thread**
//!   exclusivity, so multiple invocations are safe – they will serialize on the
//!   same cache directory instead of clobbering each other.
//! * `run`, `validate`, and `remove` **consume** the builder; you cannot forget
//!   to call one, and you cannot re-use a potentially inconsistent value.
//!
//! # Caching layout
//!
//! ```text
//! <cache-root>/                          # e.g. ~/.local/share/com/<project>/llama_cpp
//! └── llama_cpp_<tag>_<backend>/
//!     ├── working_dir/   # scratch space (ephemeral)
//!     └── bin/           # published artifacts
//!         └── llama-server[.exe]
//! ```
//!
//! Override the root with `.override_root(<PATH>)` or the environment variable
//! `LLAMA_CPP_INSTALL_DIR`.
//!
//! # When to favour *InstallOnly*
//!
//! * CI images that lack a full C/C++ tool-chain.  
//! * Quick prototyping on end-user laptops.  
//! * Deterministic bit-for-bit reproducibility across machines.
//!
//! Conversely, *BuildOnly* gives you bleeding-edge commits or local patches.
//!
//! # CLI parity
//!
//! The crate ships with a tiny companion binary,
//!
//! ```text
//! $ llama-cpp-toolchain-cli install --backend cuda --repo-tag b5890
//! $ llama-cpp-toolchain-cli validate --backend default
//! $ llama-cpp-toolchain-cli remove   --backend default
//! ```
//!
//! ensuring the same functionality is available from the shell.
//!
//! ---
//! *You are reading the public interface; if you find yourself needing anything
//! from `recipe.rs`, please open an issue – that code is considered private and
//! may change without notice.*

use std::{path::Path, time::Duration};

use bon::Builder;
use serde::{Deserialize, Serialize};

use crate::{
    error::{LmcppError, LmcppResult},
    server::{
        toolchain::recipe::LmcppRecipe,
        types::file::{ValidDir, ValidFile},
    },
};

const DEFAULT_PROJECT_NAME: &str = "llama_cpp_toolchain";
const DEFAULT_FAIL_LIMIT: u8 = 3;

/// https://github.com/ggml-org/llama.cpp/releases/tag/b6097
/// 2025/08/05
const LLAMA_CPP_DEFAULT_TAG: &str = "b6097";

#[cfg(any(target_os = "linux", target_os = "macos"))]
pub const LMCPP_SERVER_EXECUTABLE: &str = "llama-server";
#[cfg(target_os = "windows")]
pub const LMCPP_SERVER_EXECUTABLE: &str = "llama-server.exe";

/// Fluent *builder* that captures *all* inputs required to obtain a working
/// `llama-server` executable – either by **building from source** or by
/// **downloading a pre-built artefact**.
///
/// ### Typical lifecycle
///
/// 1.  Call [`LmcppToolChain::new`] or rely on [`Default`].  
/// 2.  Chain any number of setters (`repo_tag`, `compute_backend`, `build_arg`, …).  
/// 3.  Finish with **one** of  
///    * [`run`](Self::run)      – build/install and *ensure* the binary exists;  
///    * [`validate`](Self::validate) – check fingerprint only (no side effects);  
///    * [`remove`](Self::remove)    – purge the cached files.
///
/// The object is *consumed* by these terminal calls so it is impossible to
/// forget one.
#[derive(Serialize, Deserialize, Debug, Clone, Builder)]
#[builder(derive(Debug, Clone), finish_fn(vis = "", name = build_internal))]
pub struct LmcppToolChain {
    /// Additional `-D…` CMake flags to inject verbatim during **source builds**.
    /// Internal scratch space – unique & sorted by definition.
    #[builder(field)]
    build_args: ArgSet,

    /// A custom path to the binary. In this case, the path is validated and returned.
    #[builder(into)]
    pub custom_bin_path: Option<std::path::PathBuf>,

    /// Project name..
    #[builder(default = DEFAULT_PROJECT_NAME.to_string(), into)]
    pub project: String,

    #[builder(with = |dir: impl TryInto<ValidDir, Error = LmcppError>| -> LmcppResult<_> {
        dir.try_into()
    })]
    pub override_root: Option<ValidDir>,

    #[builder(default = DEFAULT_FAIL_LIMIT)]
    pub fail_limit: u8,

    /// The git revision that will be checked out.  Accepts *tags*.
    /// Defaults to [module-level default](LLAMA_CPP_DEFAULT_TAG).
    #[builder(default = LLAMA_CPP_DEFAULT_TAG.to_string(), into)]
    pub repo_tag: String,

    /// Desired compute-backend *policy*.  See [`ComputeBackendConfig`] for
    /// platform-specific semantics and fall-back rules.
    /// *Note:* policies ending in `…IfAvailable` will gracefully fall back to
    /// CPU when the preferred accelerator is missing.
    #[builder(default, name = compute_backend)]
    pub compute_cfg: ComputeBackendConfig,

    /// Build / install strategy.  See [`LmcppBuildInstallMode`].
    #[builder(default, setters(vis = "", name = mode_internal))]
    pub mode: LmcppBuildInstallMode,

    /// Toggle the *curl* tool for model downloads. Requires `libcurl` to be installed.
    #[builder(default = false)]
    pub curl_enabled: bool,

    /// distributed-compute RPC back-end
    #[builder(with = || true, default = false)]
    pub rpc_enabled: bool,

    /// Toggle the *llguidance* structured-output helper.
    #[builder(with = || true, default = false)]
    pub llguidance_enabled: bool,
}

impl Default for LmcppToolChain {
    fn default() -> Self {
        LmcppToolChain::builder()
            .build()
            .expect("Default toolchain should always be available")
    }
}

impl LmcppToolChain {
    // ── Server-only CMake flags ────────────────────────────────────────────────

    /// Skip the distributed-compute RPC back-end unless you need it.
    pub const RPC_OFF: &str = "-DGGML_RPC=OFF";

    /// Omit libcurl to remove HTTP/HF download code (models must be local).
    pub const CURL_OFF: &str = "-DLLAMA_CURL=OFF";

    /// Optional helper for llama-llguidance structured-output; keep off by default.
    pub const LLGUIDANCE_ON: &str = "-DLLAMA_LLGUIDANCE=ON";

    pub const METAL_OFF: &str = "-DGGML_METAL=OFF";
    pub const METAL_ON: &str = "-DGGML_METAL=ON";
    pub const CUDA_ARG: &str = "-DGGML_CUDA=ON";
    /// Build *or* install (depending on [`self.mode`](LmcppToolChain::mode)) and
    /// return a rich [`LmcppToolchainOutcome`] with timing, status and
    /// binary path.
    pub fn run(&self) -> LmcppResult<LmcppToolchainOutcome> {
        if let Some(custom_bin_path) = &self.custom_bin_path {
            return self.validate_custom_bin_path(custom_bin_path);
        }
        let mut recipe = LmcppRecipe::new(
            &self.project,
            &self.override_root,
            self.fail_limit,
            &self.repo_tag,
            &self.compute_cfg,
            &self.mode,
            &self.build_args,
        )?;
        let res = recipe.run()?;
        Ok(res)
    }

    /// Validate that a cached build exists *and* its fingerprint matches the
    /// current configuration.  Leaves the file-system untouched.
    pub fn validate(&self) -> LmcppResult<LmcppToolchainOutcome> {
        if let Some(custom_bin_path) = &self.custom_bin_path {
            return self.validate_custom_bin_path(custom_bin_path);
        }
        let mut recipe = LmcppRecipe::new(
            &self.project,
            &self.override_root,
            self.fail_limit,
            &self.repo_tag,
            &self.compute_cfg,
            &self.mode,
            &self.build_args,
        )?;
        let res = recipe.validate()?;
        Ok(res)
    }

    /// Remove all cached artefacts for the selected `repo_tag` × backend × mode
    /// combination.  Always succeeds unless the files are locked by *another*
    /// process.
    pub fn remove(self) -> LmcppResult<()> {
        let mut recipe = LmcppRecipe::new(
            &self.project,
            &self.override_root,
            self.fail_limit,
            &self.repo_tag,
            &self.compute_cfg,
            &self.mode,
            &self.build_args,
        )?;
        recipe.remove()
    }

    fn validate_custom_bin_path(
        &self,
        custom_bin_path: &std::path::Path,
    ) -> LmcppResult<LmcppToolchainOutcome> {
        let t0 = std::time::Instant::now();

        let bin_path = ValidFile::new(custom_bin_path)?;

        let executable_name = bin_path
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| LmcppError::InvalidConfig {
                field: "custom_bin_path",
                reason: "No executable name in custom binary path".into(),
            })?
            .to_string();

        let compute_backend: ComputeBackend = self.compute_cfg.to_backend(&self.mode)?;

        Ok(LmcppToolchainOutcome {
            duration: t0.elapsed(),
            bin_path: Some(bin_path),
            status: LmcppBuildInstallStatus::CustomBinPath,
            repo_tag: "custom_bin_path".to_string(),
            compute_backend, // No backend for custom paths
            executable_name,
            error: None, // No error for custom paths
        })
    }
}

use lmcpp_tool_chain_builder::{IsUnset, SetMode, State};

impl<S: State> LmcppToolChainBuilder<S> {
    /// Push a single `-D…` CMake flag.  Duplicate flags are ignored.
    pub fn build_arg(mut self, arg: impl Into<String>) -> Self {
        self.build_args.insert(arg.into());
        self
    }

    /// Add many flags at once – internally re-uses `build_arg`
    /// so duplicate suppression stays in one place.
    pub fn build_args<I, T>(self, args: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        args.into_iter().fold(self, |b, a| b.build_arg(a))
    }

    /// Force a **source build**.  The pre-built installer path is skipped.
    pub fn build_only(self) -> LmcppToolChainBuilder<SetMode<S>>
    where
        S::Mode: IsUnset,
    {
        self.mode_internal(LmcppBuildInstallMode::BuildOnly)
    }

    /// Force **binary installation**.  Will error out if the current platform
    /// has no official pre-built binaries.
    pub fn install_only(self) -> LmcppToolChainBuilder<SetMode<S>>
    where
        S::Mode: IsUnset,
    {
        self.mode_internal(LmcppBuildInstallMode::InstallOnly)
    }

    /// Equivalent to [`Default`] – attempt **build first**, fall back to
    /// installation on failure.
    pub fn build_or_install(self) -> LmcppToolChainBuilder<SetMode<S>>
    where
        S::Mode: IsUnset,
    {
        self.mode_internal(LmcppBuildInstallMode::BuildOrInstall)
    }

    /// Explicitly set the build / install mode.  The three possibilities are
    /// documented in [`LmcppBuildInstallMode`].
    pub fn build_install_mode(
        self,
        mode: LmcppBuildInstallMode,
    ) -> LmcppToolChainBuilder<SetMode<S>>
    where
        S::Mode: IsUnset,
    {
        self.mode_internal(mode)
    }
}

impl<S: lmcpp_tool_chain_builder::IsComplete> LmcppToolChainBuilder<S> {
    pub fn build(self) -> LmcppResult<LmcppToolChain> {
        // Delegate to `build_internal()` to get the instance of user.
        let mut chain = self.build_internal();
        if chain.project.is_empty() {
            return Err(LmcppError::InvalidConfig {
                field: "project",
                reason: "cannot be empty".into(),
            });
        }
        if chain.repo_tag.is_empty() {
            return Err(LmcppError::InvalidConfig {
                field: "repo_tag",
                reason: "cannot be empty".into(),
            });
        }
        if chain.fail_limit == 0 {
            return Err(LmcppError::InvalidConfig {
                field: "fail_limit",
                reason: "must be greater than zero".into(),
            });
        }

        if !chain.curl_enabled {
            chain.build_args.insert(LmcppToolChain::CURL_OFF.into());
        }
        if !chain.rpc_enabled {
            chain.build_args.insert(LmcppToolChain::RPC_OFF.into());
        }
        if chain.llguidance_enabled {
            chain
                .build_args
                .insert(LmcppToolChain::LLGUIDANCE_ON.into());
        }

        if chain.build_args.iter().any(|s| s.is_empty()) {
            return Err(LmcppError::InvalidConfig {
                field: "build args",
                reason: "individual arguments cannot be empty".into(),
            });
        }

        Ok(chain)
    }
}

impl std::fmt::Display for LmcppToolChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use std::fmt::Write;
        writeln!(f, "LmcppToolChain:")?;
        let mut indented = indenter::indented(f).with_str("   ");
        writeln!(indented, "Project: {}", self.project)?;
        writeln!(indented, "Repo tag: {}", self.repo_tag)?;
        writeln!(indented, "Compute backend: {:?}", self.compute_cfg)?;
        writeln!(indented, "Mode: {:?}", self.mode)?;
        writeln!(indented, "Build args: {:?}", self.build_args)?;
        Ok(())
    }
}

#[derive(Default, Serialize, Deserialize, Debug, Clone)]
#[repr(transparent)]
pub struct ArgSet(std::collections::BTreeSet<String>);

impl std::ops::Deref for ArgSet {
    type Target = std::collections::BTreeSet<String>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for ArgSet {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Strategy that decides **how** we obtain the `llama-server` binary.
#[derive(Serialize, Deserialize, PartialEq, Debug, clap::ValueEnum, Clone, Copy)]
pub enum LmcppBuildInstallMode {
    /// Compile *from source* – fails if the host has no C/C++ tool-chain.
    BuildOnly,

    /// Download and extract a *pre-built* archive if one exists for the current
    /// platform; errors otherwise.
    InstallOnly,

    /// First *try* a source build; on failure, fall back to `InstallOnly`.
    BuildOrInstall,
}

impl Default for LmcppBuildInstallMode {
    fn default() -> Self {
        LmcppBuildInstallMode::BuildOrInstall
    }
}

/// Result of the build / install phase recorded on disk and re-read on the next
/// invocation.  Used to detect whether a future run needs to rebuild.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum LmcppBuildInstallStatus {
    /// The binary was *downloaded* and verified.
    Installed,

    /// The binary was *built from source*.
    Built,

    /// Nothing was produced yet (fresh cache or after `remove`).
    NotBuiltOrInstalled,

    /// The binary was manually set by the user.
    CustomBinPath,
}

impl Default for LmcppBuildInstallStatus {
    fn default() -> Self {
        LmcppBuildInstallStatus::NotBuiltOrInstalled
    }
}

/// Human-readable summary object returned by [`LmcppToolChain::run`] and
/// [`validate`](LmcppToolChain::validate).  Implements [`Display`] for
/// pretty printing.
#[derive(Serialize, Debug)]
pub struct LmcppToolchainOutcome {
    /// End-to-end wall-clock time spent in the workflow.
    pub duration: Duration,
    /// Git tag / commit that was built or installed.
    pub repo_tag: String,
    /// Whether the binary was *Built*, *Installed* or *re-used* unchanged.
    pub status: LmcppBuildInstallStatus,
    /// Concrete backend chosen after all fall-backs (never *Default*).
    pub compute_backend: ComputeBackend,
    /// Absolute path to the resulting `llama-server` executable, if present.
    pub bin_path: Option<ValidFile>,
    /// The process name of the executable, e.g. `llama-server`.
    pub executable_name: String,
    /// The error that occurred, if any.
    pub error: Option<LmcppError>,
}

impl LmcppToolchainOutcome {
    pub fn bin_path(&self) -> Option<&Path> {
        self.bin_path.as_ref().map(|f| f.as_ref())
    }

    pub fn bin_dir(&self) -> Option<&Path> {
        self.bin_path()?.parent()
    }
}

impl std::fmt::Display for LmcppToolchainOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use std::fmt::Write;
        writeln!(f, "LmcppToolchainOutcome:")?;
        let mut indented = indenter::indented(f).with_str("   ");
        writeln!(indented, "Duration: {:?}", self.duration)?;
        writeln!(indented, "Repo tag: {}", self.repo_tag)?;
        writeln!(indented, "Status: {:?}", self.status)?;
        writeln!(indented, "Compute backend: {}", self.compute_backend)?;
        if let Some(bin_path) = &self.bin_path {
            writeln!(indented, "Binary path: {}", bin_path.display())?;
        } else {
            writeln!(indented, "Binary path: None")?;
        }
        if let Some(error) = &self.error {
            writeln!(indented, "Error: {}", error)?;
        } else {
            writeln!(indented, "Error: None")?;
        }
        Ok(())
    }
}

/// Desired compute backend as expressed by the *caller*.
///
/// The enum distinguishes between *hard requirements* (`Cuda`, `Metal`) and
/// *preferences* (`CudaIfAvailable`, `MetalIfAvailable`).  The `Default` variant
/// resolves to whichever accelerator is most performant on the current host,
/// falling back to `Cpu` if none are present.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
pub enum ComputeBackendConfig {
    /// Use the fastest backend detected on this platform.
    Default,
    /// Force a pure CPU build (always succeeds).
    Cpu,
    /// Require CUDA – error out if no NVIDIA GPU is found.
    Cuda,
    /// Prefer CUDA but silently fall back to CPU when unavailable.
    CudaIfAvailable,
    /// Require Apple Metal (macOS only).
    Metal,
    /// Prefer Metal but fall back to CPU on non-Mac systems.
    MetalIfAvailable,
}

impl ComputeBackendConfig {
    #[cfg(target_os = "macos")]
    pub fn validate_cuda(_: &LmcppBuildInstallMode) -> LmcppResult<ComputeBackend> {
        return Err(LmcppError::BackendUnavailable {
            what: "CUDA",
            os: std::env::consts::OS,
            arch: std::env::consts::ARCH,
            reason: "Not Linux or Windows".into(),
        });
    }

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub fn validate_cuda(mode: &LmcppBuildInstallMode) -> LmcppResult<ComputeBackend> {
        {
            use nvml_wrapper::Nvml;

            let nvml = Nvml::init().map_err(|e| LmcppError::BackendUnavailable {
                what: "CUDA",
                os: std::env::consts::OS,
                arch: std::env::consts::ARCH,
                reason: format!("NVML initialisation failed: {e}"),
            })?;

            if nvml.device_count().unwrap_or(0) == 0 {
                return Err(LmcppError::BackendUnavailable {
                    what: "CUDA",
                    os: std::env::consts::OS,
                    arch: std::env::consts::ARCH,
                    reason: "no CUDA-capable GPU detected".into(),
                });
            }
            match mode {
                LmcppBuildInstallMode::BuildOnly => {
                    let nvcc_ok = std::process::Command::new("nvcc")
                        .arg("--version")
                        .stdout(std::process::Stdio::null())
                        .stderr(std::process::Stdio::null())
                        .status()
                        .map(|s| s.success())
                        .unwrap_or(false);

                    if !nvcc_ok {
                        return Err(LmcppError::BackendUnavailable {
                            what: "CUDA",
                            os: std::env::consts::OS,
                            arch: std::env::consts::ARCH,
                            reason: "CUDA toolkit required to build with CUDA support. Install it, or switch to LmcppBuildInstallMode::InstallOnly.".into(),
                        });
                    }
                }
                _ => {}
            }

            Ok(ComputeBackend::Cuda)
        }
    }

    pub fn validate_metal() -> LmcppResult<ComputeBackend> {
        // 1. Reject anything that is *not* macOS.
        if cfg!(not(target_os = "macos")) {
            return Err(LmcppError::BackendUnavailable {
                what: "Metal",
                os: std::env::consts::OS,
                arch: std::env::consts::ARCH,
                reason: "Not macOS".into(),
            });
        }

        // 2.  (TODO) More detailed checks could go here:
        //     - `MTLCreateSystemDefaultDevice()` returns null
        //     - macOS version too old for M‑series inference, etc.

        Ok(ComputeBackend::Metal)
    }

    pub fn to_backend(self, mode: &LmcppBuildInstallMode) -> LmcppResult<ComputeBackend> {
        match self {
            ComputeBackendConfig::Default => Self::default_backend(mode),
            ComputeBackendConfig::Cpu => Ok(ComputeBackend::Cpu),
            ComputeBackendConfig::Cuda => Self::validate_cuda(mode),
            ComputeBackendConfig::CudaIfAvailable => Self::cuda_if_available(mode),
            ComputeBackendConfig::Metal => Self::validate_metal(),
            ComputeBackendConfig::MetalIfAvailable => Self::metal_if_available(),
        }
    }

    fn default_backend(mode: &LmcppBuildInstallMode) -> LmcppResult<ComputeBackend> {
        if cfg!(target_os = "macos") {
            Self::metal_if_available()
        } else if cfg!(any(target_os = "linux", target_os = "windows")) {
            Self::validate_cuda(mode)
                // .or_else(|_| Self::validate_amd())
                // .or_else(|_| Self::validate_intel()) // Intel Arc, etc.
                .or_else(|_| Ok(ComputeBackend::Cpu))
        } else {
            Ok(ComputeBackend::Cpu)
        }
    }

    fn cuda_if_available(mode: &LmcppBuildInstallMode) -> LmcppResult<ComputeBackend> {
        match Self::validate_cuda(mode) {
            Ok(backend) => Ok(backend),
            Err(_) => Ok(ComputeBackend::Cpu), // Fallback to CPU if CUDA is not available
        }
    }

    fn metal_if_available() -> LmcppResult<ComputeBackend> {
        match Self::validate_metal() {
            Ok(_) => Ok(ComputeBackend::Metal),
            Err(_) => Ok(ComputeBackend::Cpu), // Fallback to CPU if Metal is not available
        }
    }
}

impl Default for ComputeBackendConfig {
    fn default() -> Self {
        ComputeBackendConfig::Default
    }
}

/// The *resolved* backend that a recipe was actually built for.
///
/// Once a build is complete the backend is stored alongside the produced
/// binaries so that the next invocation can detect drift (e.g. binaries that
/// were compiled for CPU only but are now requested for CUDA).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputeBackend {
    Cpu,
    Cuda,
    Metal,
}

impl std::fmt::Display for ComputeBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComputeBackend::Cpu => write!(f, "CPU"),
            ComputeBackend::Cuda => write!(f, "CUDA"),
            ComputeBackend::Metal => write!(f, "Metal"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Build-arg semantics ──────────────────────────────────────────────────

    /// `build_args` must be lexicographically sorted **and** duplicates removed,
    /// even when the caller injects the same flag several times and the builder
    /// adds its own automatic flags.
    #[test]
    fn build_args_are_sorted_and_deduped() {
        let chain = LmcppToolChain::builder()
            .build_arg("-DGGML_RPC=OFF") // duplicate …
            .build_arg("-DLLAMA_CURL=OFF")
            .build_arg("-DGGML_RPC=OFF") // … to be deduped
            .build_only()
            .build()
            .expect("builder must succeed");

        let actual: Vec<_> = chain.build_args.iter().cloned().collect();
        let expected = vec!["-DGGML_RPC=OFF".to_string(), "-DLLAMA_CURL=OFF".to_string()];

        assert_eq!(
            actual, expected,
            "builder should sort lexicographically and drop duplicates"
        );
    }

    // ── Validation logic (consolidated) ──────────────────────────────────────

    #[test]
    fn builder_invalid_inputs_error_out() {
        let cases: Vec<(&str, LmcppResult<LmcppToolChain>)> = vec![
            (
                "empty project",
                LmcppToolChain::builder().project("").build_only().build(),
            ),
            (
                "empty repo_tag",
                LmcppToolChain::builder().repo_tag("").build_only().build(),
            ),
            (
                "zero fail_limit",
                LmcppToolChain::builder().fail_limit(0).build_only().build(),
            ),
            (
                "empty build-arg",
                LmcppToolChain::builder().build_arg("").build_only().build(),
            ),
        ];

        for (name, res) in cases {
            assert!(res.is_err(), "builder must reject invalid input: {}", name);
        }
    }

    // ── Automatic flag injection (consolidated) ─────────────────────────────
    #[test]
    fn flag_injection_scenarios() {
        struct Scenario {
            name: &'static str,
            chain: LmcppToolChain,
            expect_curl_off: bool,
            expect_rpc_off: bool,
            expect_llg_on: bool,
        }

        let scenarios = vec![
            // defaults: curl=false, rpc=false, llguidance=false
            Scenario {
                name: "defaults",
                chain: LmcppToolChain::builder()
                    .build_only()
                    .build()
                    .expect("defaults"),
                expect_curl_off: true,
                expect_rpc_off: true,
                expect_llg_on: false,
            },
            Scenario {
                name: "llguidance ON",
                chain: LmcppToolChain::builder()
                    .llguidance_enabled()
                    .build_only()
                    .build()
                    .expect("llguidance ON"),
                expect_curl_off: true,
                expect_rpc_off: true,
                expect_llg_on: true,
            },
            Scenario {
                name: "curl enabled",
                chain: LmcppToolChain::builder()
                    .curl_enabled(true)
                    .build_only()
                    .build()
                    .expect("curl enabled"),
                expect_curl_off: false,
                expect_rpc_off: true,
                expect_llg_on: false,
            },
            Scenario {
                name: "rpc enabled",
                chain: LmcppToolChain::builder()
                    .rpc_enabled()
                    .build_only()
                    .build()
                    .expect("rpc enabled"),
                expect_curl_off: true,
                expect_rpc_off: false,
                expect_llg_on: false,
            },
            Scenario {
                name: "all toggled",
                chain: LmcppToolChain::builder()
                    .curl_enabled(true)
                    .rpc_enabled()
                    .llguidance_enabled()
                    .build_only()
                    .build()
                    .expect("all toggled"),
                expect_curl_off: false,
                expect_rpc_off: false,
                expect_llg_on: true,
            },
        ];

        for s in scenarios {
            let flags = &s.chain.build_args;
            assert_eq!(
                flags.contains(LmcppToolChain::CURL_OFF),
                s.expect_curl_off,
                "{}: CURL_OFF presence mismatch",
                s.name
            );
            assert_eq!(
                flags.contains(LmcppToolChain::RPC_OFF),
                s.expect_rpc_off,
                "{}: RPC_OFF presence mismatch",
                s.name
            );
            assert_eq!(
                flags.contains(LmcppToolChain::LLGUIDANCE_ON),
                s.expect_llg_on,
                "{}: LLGUIDANCE_ON presence mismatch",
                s.name
            );
        }
    }

    // ── Mode helpers & default value (consolidated) ─────────────────────────

    #[test]
    fn build_install_mode_helpers() {
        assert_eq!(
            LmcppToolChain::builder().build_only().build().unwrap().mode,
            LmcppBuildInstallMode::BuildOnly
        );
        assert_eq!(
            LmcppToolChain::builder()
                .install_only()
                .build()
                .unwrap()
                .mode,
            LmcppBuildInstallMode::InstallOnly
        );
        assert_eq!(
            LmcppToolChain::builder()
                .build_or_install()
                .build()
                .unwrap()
                .mode,
            LmcppBuildInstallMode::BuildOrInstall
        );
        // `Default` impl must match the most permissive helper.
        assert_eq!(
            LmcppBuildInstallMode::default(),
            LmcppBuildInstallMode::BuildOrInstall
        );
    }

    // ── ComputeBackend & Display helpers (unchanged) ────────────────────────

    #[test]
    fn compute_backend_display_strings() {
        assert_eq!(ComputeBackend::Cpu.to_string(), "CPU");
        assert_eq!(ComputeBackend::Cuda.to_string(), "CUDA");
        assert_eq!(ComputeBackend::Metal.to_string(), "Metal");
    }

    #[test]
    fn compute_backend_config_to_backend_cpu() {
        let backend = ComputeBackendConfig::Cpu
            .to_backend(&LmcppBuildInstallMode::default())
            .expect("CPU backend must always be available");
        assert_eq!(backend, ComputeBackend::Cpu);
    }
}
