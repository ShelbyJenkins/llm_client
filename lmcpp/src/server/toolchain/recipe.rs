//! Internal *recipe engine* for the **llama-cpp tool-chain**
//! =========================================================
//!
//! This module holds the *state machine* that actually **builds**, **downloads**,
//! **fingerprints**, and **publishes** a `llama-server` executable.  It is a
//! **private implementation detail** – everything here is consumed by
//! `toolchain.rs`, never by end-users.
//!
//! # Responsibilities
//!
//! 1. **Cache layout & paths**  
//!    * Figures out the root directory (`resolve_root`) from an explicit
//!      override, an environment variable (`LLAMA_CPP_INSTALL_DIR`),
//!      or `directories::ProjectDirs`.
//!    * Creates deterministic sub-directories
//!      `…/<version>/working_dir` and `…/<version>/bin` on the **same file-system**
//!      to allow atomic moves.
//!
//! 2. **Exclusive access**  
//!    * Guards the entire tree with **two* layers of locking:  
//!       • an *in-process* `Mutex` (one per path, leaked for `'static`)  
//!       • a *cross-process* advisory `flock` on `"<version>.lock"`  
//!    The helper `lock_file` returns **both** guards to the caller, guaranteeing
//!    that no competing thread or process can corrupt an ongoing build/install.
//!
//! 3. **Build / install pipelines**  
//!    * [`build_from_source`]: CMake checkout → configure → build.  
//!    * [`install_prebuilt`]: download an official zip and unzip it.  
//!    Selection is driven by [`LmcppBuildInstallMode`]; *BuildOrInstall* will try
//!    source first, then fall back to binaries.
//!
//! 4. **Fingerprinting & drift detection**  
//!    * Persists [`LmcppToolchainState`] via `confy` (one TOML per
//!      `repo_tag × backend × mode`) in the cache root.
//!    * `fingerprint_matches` rejects stale binaries when the build args,
//!      backend, or status no longer line up with the current request.
//!
//! 5. **Failure handling & self-healing**  
//!    * Tracks a *fail counter*; after `fail_limit` consecutive errors the
//!      entire `<version>` directory is purged and the next invocation starts
//!      from scratch.
//!
//! 6. **Finalisation**  
//!    * Marks the fresh binary as executable (Unix).  
//!    * Atomically replaces `bin/` with a single `rename`, so readers never see
//!      a half-populated directory.  
//!    * Resets `working_dir` to keep the cache tidy.
//!
//! # Why lock *twice*?
//!
//! *The mutex* protects against races inside **one** process (tests spawning
//! multiple threads, for instance).  
//! *The `flock`* extends the same guarantee to **other** processes that might
//! share the cache (parallel CI jobs, multiple shells, etc.).  Holding the file
//! handle open for the full lifetime of the guard prevents silent unlocks on
//! Windows.
//!
//! # Error strategy
//!
//! All public helpers return `Result<T>` with rich context via
//! `anyhow::Context`.  The outer caller (the builder in `toolchain.rs`) decides
//! whether to propagate or recover.
//!
//! **Never** import this module directly from application code; rely on
//! `LmcppToolChain` instead.

use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io::ErrorKind,
    path::PathBuf,
    sync::{Mutex, MutexGuard, OnceLock},
};

use fs4::fs_std::FileExt;
use serde::{Deserialize, Serialize};

use crate::{
    error::{LmcppError, LmcppResult},
    server::{
        toolchain::builder::{
            ArgSet, ComputeBackend, ComputeBackendConfig, LMCPP_SERVER_EXECUTABLE,
            LmcppBuildInstallMode, LmcppBuildInstallStatus, LmcppToolChain, LmcppToolchainOutcome,
        },
        types::file::{ValidDir, ValidFile},
    },
};

static IN_PROC_LOCKS: OnceLock<Mutex<HashMap<PathBuf, &'static Mutex<()>>>> = OnceLock::new();

pub struct LmcppRecipe {
    pub cfg: LmcppToolchainState,
    pub mode: LmcppBuildInstallMode,
    pub expected_build_args: ArgSet,
    pub root_dir: ValidDir,
    pub working_dir: ValidDir,
    pub bin_dir: ValidDir,
    pub fail_limit: u8,
    pub version: String,
    pub project: String,
}

impl LmcppRecipe {
    const RECIPE_NAME: &'static str = "llama_cpp";

    const LLAMA_CPP_REPO_URL: &str = "https://github.com/ggml-org/llama.cpp";
    const LLAMA_CPP_ENV_OVERRIDE: &str = "LLAMA_CPP_INSTALL_DIR";

    pub fn new(
        project: &str,
        override_root: &Option<ValidDir>,
        fail_limit: u8,
        repo_tag: &str,
        compute_cfg: &ComputeBackendConfig,
        mode: &LmcppBuildInstallMode,
        build_args: &ArgSet,
    ) -> LmcppResult<Self> {
        assert!(!project.is_empty(), "Project name cannot be empty");
        assert!(!repo_tag.is_empty(), "Repo tag cannot be empty");
        assert!(fail_limit > 0, "Fail limit must be greater than zero");
        assert!(
            !build_args.iter().any(|s| s.is_empty()),
            "A build argument cannot be empty"
        );
        let mut build_args = build_args.clone();

        let compute_backend: ComputeBackend = compute_cfg.to_backend(mode)?;
        #[cfg(any(target_os = "linux", windows))]
        {
            if matches!(compute_backend, ComputeBackend::Cuda) {
                build_args.insert(LmcppToolChain::CUDA_ARG.to_owned());
            }
        }
        #[cfg(target_os = "macos")]
        {
            build_args.insert("-DBUILD_SHARED_LIBS=OFF".to_owned());
            if matches!(compute_backend, ComputeBackend::Metal) {
                build_args.insert(LmcppToolChain::METAL_ON.to_owned());
            } else {
                build_args.insert(LmcppToolChain::METAL_OFF.to_owned());
            }
        }

        let version = format!("llama_cpp_{}_{}", repo_tag, compute_backend);
        let root_dir = Self::resolve_root(override_root.as_ref(), &project)?;
        let working_dir = ValidDir::new(root_dir.join(&version).join("working_dir"))?;
        let bin_dir = ValidDir::new(root_dir.join(&version).join("bin"))?;

        // Ensure the working_dir and bin_dir exist inside root_dir, and all are on the same filesystem.
        #[cfg(debug_assertions)]
        {
            debug_assert!(working_dir.starts_with(&root_dir), "must be in root_dir");
            debug_assert!(bin_dir.starts_with(&root_dir), "must be in root_dir");
            #[cfg(unix)] // Linux, macOS, *BSD
            {
                use std::os::unix::fs::MetadataExt;
                let dev_root = std::fs::metadata(&root_dir).unwrap().dev();
                debug_assert!(
                    dev_root == std::fs::metadata(&working_dir).unwrap().dev()
                        && dev_root == std::fs::metadata(&bin_dir).unwrap().dev(),
                    "root_dir, working_dir and bin_dir must be on the same filesystem"
                );
            }

            #[cfg(windows)]
            {
                // ── 1. Canonicalise once and keep the PathBufs alive ─────────────────────────
                let root_canon = root_dir.canonicalize().unwrap();
                let working_canon = working_dir.canonicalize().unwrap();
                let bin_canon = bin_dir.canonicalize().unwrap();

                // ── 2. Take the drive component from each canonical path ─────────────────────
                let root_drive = root_canon.components().next().unwrap();
                let working_drive = working_canon.components().next().unwrap();
                let bin_drive = bin_canon.components().next().unwrap();

                // ── 3. Assert they’re all the same ────────────────────────────────────────────
                debug_assert!(
                    root_drive == working_drive && root_drive == bin_drive,
                    "root_dir, working_dir and bin_dir must be on the same drive"
                );
            }
        }

        let mut cfg: LmcppToolchainState =
            confy::load(&project, Some(version.as_str())).map_err(|e| {
                LmcppError::file_system(
                    "loading confy configuration file",
                    "confy derives path",
                    std::io::Error::new(std::io::ErrorKind::Other, e.to_string()),
                )
            })?;

        if let Some(ref cfg_repo_tag) = cfg.repo_tag {
            debug_assert!(*cfg_repo_tag == repo_tag, "Repo tag mismatch");
        } else {
            cfg.repo_tag = Some(repo_tag.to_string());
        }

        if let Some(ref cfg_compute_backend) = cfg.compute_backend {
            debug_assert!(
                *cfg_compute_backend == compute_backend,
                "Compute backend mismatch"
            );
        } else {
            cfg.compute_backend = Some(compute_backend);
        }

        Ok(Self {
            cfg,
            mode: mode.clone(),
            root_dir,
            working_dir,
            bin_dir,
            fail_limit,
            version,
            project: project.to_owned(),
            expected_build_args: build_args.clone(),
        })
    }

    pub fn run(&mut self) -> LmcppResult<LmcppToolchainOutcome> {
        let t0 = std::time::Instant::now();

        let exec: std::result::Result<ValidFile, _> =
            ValidFile::find_specific_file(&self.bin_dir, LMCPP_SERVER_EXECUTABLE);

        if let Ok(ref bin_path) = exec {
            if self.fingerprint_matches().is_ok() {
                // Fast-path: nothing to do.
                return Ok(LmcppToolchainOutcome {
                    duration: t0.elapsed(),
                    bin_path: Some(bin_path.clone()),
                    status: self.cfg.status.clone(),
                    repo_tag: self
                        .cfg
                        .repo_tag
                        .as_ref()
                        .expect("Repo tag set in constructor")
                        .clone(),
                    compute_backend: self
                        .cfg
                        .compute_backend
                        .as_ref()
                        .expect("Compute backend set in constructor")
                        .clone(),
                    executable_name: LMCPP_SERVER_EXECUTABLE.to_string(),
                    error: None,
                });
            }
        }

        // Either the binary is missing, or its fingerprint is stale.
        // From here on we need exclusive access.
        let _lock = self.lock_file()?;

        self.reset_toolchain()?;
        let working_dir = self.working_dir.clone();
        let uncheked_result = match self.mode {
            LmcppBuildInstallMode::InstallOnly => self.install_prebuilt(&working_dir),
            LmcppBuildInstallMode::BuildOnly => self.build_from_source(&working_dir),
            LmcppBuildInstallMode::BuildOrInstall => {
                let res = self.build_from_source(&working_dir);
                match res {
                    Ok(src_binary) => Ok(src_binary),
                    Err(_) => {
                        self.expected_build_args.clear();
                        self.install_prebuilt(&working_dir)
                    }
                }
            }
        };

        let inner_result = match uncheked_result {
            Ok(src_binary) => {
                // ── 1  Finalise the installation ───────────────────────────────
                self.finalise(src_binary)
            }
            Err(e) => {
                // ── 2  If the recipe failed, log it and return the error ─────
                crate::error!("Failed to run recipe for {}: {e}", Self::RECIPE_NAME);
                Err(e)
            }
        };

        let duration = t0.elapsed();

        crate::trace!(
            "{} build/install completed in {:02}:{:02}:{:02}.{:03} for {}",
            Self::RECIPE_NAME,
            duration.as_secs() / 3600,
            (duration.as_secs() % 3600) / 60,
            duration.as_secs() % 60,
            duration.subsec_millis(),
            self.root_dir.display()
        );

        match inner_result {
            Ok(bin_path) => {
                self.cfg.fail_count = 0;
                self.store_cfg()?;
                Ok(LmcppToolchainOutcome {
                    duration: t0.elapsed(),
                    bin_path: Some(bin_path.clone()),
                    status: self.cfg.status.clone(),
                    repo_tag: self
                        .cfg
                        .repo_tag
                        .as_ref()
                        .expect("Repo tag set in constructor")
                        .clone(),
                    compute_backend: self
                        .cfg
                        .compute_backend
                        .as_ref()
                        .expect("Compute backend set in constructor")
                        .clone(),
                    executable_name: LMCPP_SERVER_EXECUTABLE.to_string(),
                    error: None,
                })
            }
            Err(e) => {
                self.cfg.status = LmcppBuildInstallStatus::NotBuiltOrInstalled;
                self.cfg.fail_count = self.cfg.fail_count.saturating_add(1);
                let n = self.cfg.fail_count;

                if n >= self.fail_limit {
                    crate::error!(
                        "{n} consecutive failures - purging {} and starting fresh",
                        self.root_dir.display()
                    );
                    self.reset_toolchain()?;
                } else {
                    crate::error!(" {n} consecutive failures");
                    self.store_cfg()?;
                };
                Ok(LmcppToolchainOutcome {
                    duration: t0.elapsed(),
                    bin_path: None,
                    status: self.cfg.status.clone(),
                    repo_tag: self
                        .cfg
                        .repo_tag
                        .as_ref()
                        .expect("Repo tag set in constructor")
                        .clone(),
                    compute_backend: self
                        .cfg
                        .compute_backend
                        .as_ref()
                        .expect("Compute backend set in constructor")
                        .clone(),
                    executable_name: "error".to_string(),
                    error: Some(e),
                })
            }
        }
    }

    pub fn validate(&mut self) -> LmcppResult<LmcppToolchainOutcome> {
        let t0 = std::time::Instant::now();
        let bin_path = ValidFile::find_specific_file(&self.bin_dir, LMCPP_SERVER_EXECUTABLE)?;

        self.fingerprint_matches()?;
        Ok(LmcppToolchainOutcome {
            duration: t0.elapsed(),
            bin_path: Some(bin_path),
            status: self.cfg.status.clone(),
            repo_tag: self
                .cfg
                .repo_tag
                .as_ref()
                .expect("Repo tag set in constructor")
                .clone(),
            compute_backend: self
                .cfg
                .compute_backend
                .as_ref()
                .expect("Compute backend set in constructor")
                .clone(),
            executable_name: LMCPP_SERVER_EXECUTABLE.to_string(),
            error: None,
        })
    }

    pub fn remove(&mut self) -> LmcppResult<()> {
        // Make sure we are the only process touching this tree ──────────────
        {
            // Hold the lock for the briefest possible time; it is automatically
            // released when the guard goes out of scope so Windows can delete the
            // lock file afterwards.
            let _lock = self.lock_file()?;
        }

        // Purge the entire recipe directory tree ────────────────────────────
        if self.root_dir.exists() {
            self.root_dir.remove()?; // ValidDir::remove() -> rm -r <root-dir>
        }
        Ok(())
    }

    fn install_prebuilt(&mut self, working_dir: &ValidDir) -> LmcppResult<ValidFile> {
        let repo_tag = &self
            .cfg
            .repo_tag
            .as_ref()
            .expect("Repo tag set in constructor");
        let repo_url = Self::LLAMA_CPP_REPO_URL;

        let url = if cfg!(target_os = "linux") {
            format!("{repo_url}/releases/download/{repo_tag}/llama-{repo_tag}-bin-ubuntu-x64.zip")
        } else if cfg!(target_os = "macos") {
            match std::env::consts::ARCH {
                "aarch64" => format!(
                    "{repo_url}/releases/download/{repo_tag}/llama-{repo_tag}-bin-macos-arm64.zip"
                ),
                "x86_64" => format!(
                    "{repo_url}/releases/download/{repo_tag}/llama-{repo_tag}-bin-macos-x64.zip"
                ),
                arch => panic!("Unsupported architecture on macOS: {}", arch),
            }
        } else if cfg!(target_os = "windows") {
            format!(
                "{repo_url}/releases/download/{repo_tag}/llama-{repo_tag}-bin-win-cuda-12.4-x64.zip"
            )
        } else {
            return Err(LmcppError::BackendUnavailable {
                what: "Llama.cpp",
                os: std::env::consts::OS,
                arch: std::env::consts::ARCH,
                reason: format!("Unsupported OS: {}", std::env::consts::OS),
            });
        };

        super::zip::download_and_extract_zip(&url, working_dir, "llama_cpp_binary")?;
        let bin_path = ValidFile::find_specific_file(working_dir, LMCPP_SERVER_EXECUTABLE)?;
        self.cfg.status = LmcppBuildInstallStatus::Installed;
        self.cfg.actual_build_args = ArgSet::default();
        Ok(bin_path)
    }

    fn build_from_source(&mut self, working_dir: &ValidDir) -> LmcppResult<ValidFile> {
        super::cmake::cmake_is_available()?;

        // Determine whether curl is enabled (i.e. the OFF flag is *absent*)
        let curl_disabled = self
            .expected_build_args
            .iter()
            .any(|arg| arg == LmcppToolChain::CURL_OFF);

        if !curl_disabled {
            super::cmake::curl_is_available()?;
        }

        let repo_url = Self::LLAMA_CPP_REPO_URL;

        let url = format!(
            "{repo_url}/archive/refs/tags/{}.zip",
            self.cfg
                .repo_tag
                .as_ref()
                .expect("Repo tag set in constructor")
        );
        let build_args: Vec<&str> = self
            .expected_build_args
            .iter()
            .map(String::as_str)
            .collect();

        super::zip::download_and_extract_zip(&url, working_dir, "llama_cpp_repo")?;
        super::cmake::cmake_project_buildsystem(working_dir, &build_args)?;
        super::cmake::cmake_build_project(working_dir)?;
        let bin_path = ValidFile::find_specific_file(working_dir, LMCPP_SERVER_EXECUTABLE)?;
        self.cfg.status = LmcppBuildInstallStatus::Built;
        self.cfg.actual_build_args = self.expected_build_args.clone();
        println!("actual build args: {:?}", self.cfg.actual_build_args);
        println!("expected build args: {:?}", self.expected_build_args);
        Ok(bin_path)
    }

    fn resolve_root(override_root: Option<&ValidDir>, project: &str) -> LmcppResult<ValidDir> {
        // ── 0. Environment override, but only if the variable is *actually* set ─────
        let env_path = std::env::var_os(Self::LLAMA_CPP_ENV_OVERRIDE).map(PathBuf::from); //  None if not set

        // 1. explicit path from the caller
        if let Some(p) = override_root {
            return ValidDir::new(p);
        }

        // 2. environment variable (only when set)
        if let Some(p) = env_path.as_ref() {
            return ValidDir::new(p);
        }

        // 3. platform-native data directory
        let project_dir = directories::ProjectDirs::from("com", project, Self::RECIPE_NAME)
            .ok_or_else(|| {
                LmcppError::file_system(
                    "resolve_root",
                    "directories derives path",
                    std::io::Error::new(
                        ErrorKind::NotFound,
                        "Failed to resolve platform-native data directory",
                    ),
                )
            })?;
        let p = ValidDir::new(project_dir.data_dir())?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755))
                .map_err(|e| LmcppError::file_system("resolve_root", p.as_ref(), e))?;
        }
        Ok(p)
    }

    fn fingerprint_matches(&self) -> LmcppResult<()> {
        // ── status consistency ───────────────────────────────────────────────────
        match self.mode {
            LmcppBuildInstallMode::InstallOnly => {
                if self.cfg.status != LmcppBuildInstallStatus::Installed {
                    return Err(LmcppError::Fingerprint {
                        reason: format!("expected installed, found {:?}", self.cfg.status),
                    });
                }
            }
            LmcppBuildInstallMode::BuildOnly => {
                if self.cfg.status != LmcppBuildInstallStatus::Built {
                    return Err(LmcppError::Fingerprint {
                        reason: format!("expected built, found {:?}", self.cfg.status),
                    });
                }
            }
            LmcppBuildInstallMode::BuildOrInstall => {
                if self.cfg.status == LmcppBuildInstallStatus::NotBuiltOrInstalled {
                    return Err(LmcppError::Fingerprint {
                        reason: format!("expected built or installed, found {:?}", self.cfg.status),
                    });
                }
            }
        }

        // ── build-argument parity (skip only for pure-install workflows) ─────────
        if self.mode != LmcppBuildInstallMode::InstallOnly {
            let args_match = self.expected_build_args.len() == self.cfg.actual_build_args.len()
                && self
                    .expected_build_args
                    .iter()
                    .all(|arg| self.cfg.actual_build_args.contains(arg));

            if !args_match {
                return Err(LmcppError::Fingerprint {
                    reason: format!(
                        "expected build arguments {:?}, found {:?}",
                        self.expected_build_args, self.cfg.actual_build_args
                    ),
                });
            }
        }

        Ok(())
    }

    fn store_cfg(&self) -> LmcppResult<()> {
        confy::store(&self.project, Some(self.version.as_str()), &self.cfg).map_err(|e| {
            {
                LmcppError::file_system(
                    "saving confy configuration file",
                    "confy derives path",
                    std::io::Error::new(std::io::ErrorKind::Other, e.to_string()),
                )
            }
        })
    }

    fn reset_toolchain(&mut self) -> LmcppResult<()> {
        if self.working_dir.exists() {
            self.working_dir.remove()?;
        }
        if self.bin_dir.exists() {
            self.bin_dir.remove()?;
        }
        self.cfg.fail_count = 0;
        self.cfg.status = LmcppBuildInstallStatus::NotBuiltOrInstalled;
        self.store_cfg()?;
        Ok(())
    }

    fn finalise(&self, src_binary: ValidFile) -> LmcppResult<ValidFile> {
        self.fingerprint_matches()?;
        // ── 1  Make the fresh binary executable (Unix only) ───────────────
        src_binary.make_executable()?;

        // ── 2  Atomically publish “bin/” inside root_dir ──────────────────
        let src_dir = src_binary.parent().ok_or_else(|| LmcppError::FileSystem {
            operation: "get src_dir from src_binary",
            path: src_binary.to_path_buf(),
            source: std::io::Error::new(
                ErrorKind::Other,
                "Failed to determine parent directory of the binary".to_string(),
            ),
        })?;

        if self.bin_dir.exists() {
            self.bin_dir.remove()?;
        }

        std::fs::rename(src_dir, &self.bin_dir).map_err(|e| {
            LmcppError::file_system("move src_dir to bin_dir", self.bin_dir.as_ref(), e)
        })?;

        // ── 3  Reset working_dir for next attempt ─────────────────────────
        self.working_dir.reset()?;

        // ── 4  Final integrity check ──────────────────────────────────────
        ValidFile::find_specific_file(&self.bin_dir, LMCPP_SERVER_EXECUTABLE)
    }

    /// Acquire both an **in-process** (`Mutex`) and **cross-process**
    /// (filesystem) lock for this tool-chain directory/version.
    ///
    /// Returns a tuple holding:
    /// 1. `File`  – the open handle whose exclusive advisory lock keeps
    ///               *other processes* out.
    /// 2. `MutexGuard<'static, ()>` – the guard that keeps *other threads in
    ///               this process* out.
    ///
    /// The guard’s `'static` lifetime is achieved by leaking one
    /// `Mutex<()>` per unique lock-file path.  That lets us return a guard
    /// that lives as long as the program, while never exposing the `Mutex`
    /// itself to callers.
    fn lock_file(&self) -> LmcppResult<(File, MutexGuard<'static, ()>)> {
        let lock_path = self.root_dir.join(format!("{}.lock", self.version));

        // ────────────────────────────────────────────────────────────────
        // 1. In-process lock  (blocks other threads in *this* process)
        // ────────────────────────────────────────────────────────────────
        //
        // We keep a global map   PathBuf → &'static Mutex<()>
        // so every unique path gets its own process-wide mutex.
        //
        // • The OnceLock ensures the map is initialised exactly once.
        // • We lock the map only long enough to look up / create the entry.
        // • Box::leak turns the Mutex into a leaked ’static reference,
        //   making guards valid for ’static and therefore returnable.
        let guard = {
            let mut map = IN_PROC_LOCKS
                .get_or_init(|| Mutex::new(HashMap::new()))
                .lock()
                .unwrap();

            // Fetch or create the leaked mutex for this path.
            let m: &'static Mutex<()> = *map
                .entry(lock_path.clone())
                .or_insert_with(|| Box::leak(Box::new(Mutex::new(()))));

            // Try to take the mutex immediately; fail if another thread
            // already holds it.
            m.try_lock().map_err(|e| {
                LmcppError::file_system(
                    "tool-chain dir is already being modified by another thread",
                    self.root_dir.as_ref(),
                    std::io::Error::new(ErrorKind::Other, e.to_string()),
                )
            })?
        };

        // ────────────────────────────────────────────────────────────────
        // 2. Cross-process lock  (blocks other *processes*)
        // ────────────────────────────────────────────────────────────────
        //
        // Create/open the lock file and request an exclusive advisory lock.
        // The returned `File` must stay alive for the duration of the lock,
        // so we return it to the caller.
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&lock_path)
            .map_err(|e| LmcppError::file_system("open lock file", &lock_path, e))?;

        file.try_lock_exclusive().map_err(|e| match e.kind() {
            ErrorKind::WouldBlock => LmcppError::file_system(
                "tool-chain dir is already being modified by another *process*",
                lock_path,
                e,
            ),
            _ => LmcppError::file_system("acquire directory lock", lock_path, e),
        })?;

        // Holding both `file` and `guard` now guarantees exclusive access.
        Ok((file, guard))
    }
}

#[derive(Serialize, Deserialize, Default, Debug)]
pub struct LmcppToolchainState {
    repo_tag: Option<String>,
    status: LmcppBuildInstallStatus,
    compute_backend: Option<ComputeBackend>,
    actual_build_args: ArgSet,
    fail_count: u8,
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;
    use crate::error::LmcppError;

    /* ────────────────────────────────────────────────────────────────
     *  Helper: a “dummy” recipe that never succeeds
     * ───────────────────────────────────────────────────────────── */

    /// A minimal harness that exercises the *failure* paths of
    /// `LmcppRecipe::run` / `reset_toolchain` without invoking CMake,
    /// downloading archives, etc.
    struct DummyRecipe {
        chain: LmcppRecipe,
    }

    impl DummyRecipe {
        fn new(root: &std::path::Path, fail_limit: u8) -> Self {
            let override_root = Some(ValidDir::new(root).unwrap());

            let chain = LmcppRecipe::new(
                "dummy_project",                   // project
                &override_root,                    // override_root
                fail_limit,                        // fail‑limit
                "v0",                              // repo_tag
                &ComputeBackendConfig::Cpu,        // backend = CPU
                &LmcppBuildInstallMode::BuildOnly, // any build‑only mode will hit CMake early
                &ArgSet::default(),                // no build‑args
            )
            .unwrap();

            Self { chain }
        }

        /// Deliberately *fails* and triggers the same error‑handling
        /// branches that `LmcppRecipe::run` would exercise after a build error.
        fn run_toolchain(&mut self) -> LmcppResult<LmcppToolchainOutcome> {
            // Simulate a failed build/install attempt ────────────────
            let build_err = LmcppError::InvalidConfig {
                field: "dummy_field",
                reason: "Simulated failure for testing purposes".into(),
            };

            // Increment the fail counter exactly the way `run` does —
            // including the “reset or store” logic.
            self.chain.cfg.fail_count = self.chain.cfg.fail_count.saturating_add(1);
            let n = self.chain.cfg.fail_count;

            if n >= self.chain.fail_limit {
                self.chain.reset_toolchain()?; // purge working_dir & bin_dir
            } else {
                self.chain.store_cfg()?; // just persist state
            }

            Err(build_err.into())
        }
    }

    /* ────────────────────────────────────────────────────────────────
     *   reset_toolchain() is called after N consecutive failures
     * ───────────────────────────────────────────────────────────── */

    #[test]
    fn resets_after_n_fails() {
        let tmp = tempdir().unwrap();
        let root = tmp.path();

        // fail‑limit == 1   →  second failure must purge directories
        let mut recipe = DummyRecipe::new(root, /*fail_limit=*/ 1);

        /* 1️⃣  First run — fails but *also* resets right away
         *     because n==fail_limit.                                */
        let _ = recipe.run_toolchain().expect_err("must fail #1");

        /* Create a sentinel file inside bin/ that should disappear
         * when reset_toolchain() is triggered again.                */
        std::fs::create_dir_all(&recipe.chain.bin_dir).unwrap();
        let sentry = recipe.chain.bin_dir.join("sentry");
        std::fs::write(&sentry, b"x").unwrap();
        assert!(sentry.exists(), "sentry file must exist before 2nd run");

        /* 2️⃣  Second failure — should purge bin/ and therefore the sentinel. */
        let _ = recipe.run_toolchain().expect_err("must fail #2");

        assert!(
            !sentry.exists(),
            "sentry file must be removed by reset_toolchain()"
        );
    }

    /* ────────────────────────────────────────────────────────────────
     *  finalise() moves artefacts and validates fingerprint
     * ───────────────────────────────────────────────────────────── */

    #[test]
    fn finalise_happy_path_moves_and_validates() -> LmcppResult<()> {
        let tmp = tempdir().unwrap();
        let override_root = Some(ValidDir::new(tmp.path())?);

        let mut recipe = LmcppRecipe::new(
            "finalise_test",
            &override_root,
            3, // fail‑limit (unused here)
            "v0",
            &ComputeBackendConfig::Cpu,
            &LmcppBuildInstallMode::BuildOnly, // fingerprint expects “Built”
            &ArgSet::default(),
        )?;

        /* Make fingerprint consistent with a successful *build*. */
        recipe.cfg.status = LmcppBuildInstallStatus::Built;

        /* Seed bin/ with junk to ensure finalise wipes it. */
        std::fs::create_dir_all(&recipe.bin_dir).unwrap();
        std::fs::write(recipe.bin_dir.join("stale"), b"x").unwrap();

        /* Produce a dummy `llama-server` inside working_dir/build_out/. */
        let src_dir = recipe.working_dir.join("build_out");
        std::fs::create_dir_all(&src_dir).unwrap();
        let src_bin_path = src_dir.join(LMCPP_SERVER_EXECUTABLE);
        std::fs::write(&src_bin_path, b"dummy-binary").unwrap();

        // Convert path → ValidFile
        let src_bin: ValidFile = src_bin_path.try_into()?;

        recipe.cfg.actual_build_args = recipe.expected_build_args.clone();
        /* Execute finalise(); it should
         *   – chmod the binary
         *   – move build_out/ → bin/
         *   – empty working_dir/                                       */
        let result = recipe.finalise(src_bin)?;

        /*  ✅  Assertions                                               */
        assert!(result.exists());
        assert_eq!(
            result.file_name().unwrap(),
            std::ffi::OsStr::new(LMCPP_SERVER_EXECUTABLE)
        );

        // bin/ now contains **only** the fresh binary
        let names: Vec<_> = std::fs::read_dir(&recipe.bin_dir)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(
            names,
            vec![std::ffi::OsString::from(LMCPP_SERVER_EXECUTABLE)]
        );

        // working_dir should be empty
        assert!(
            std::fs::read_dir(&recipe.working_dir)
                .unwrap()
                .next()
                .is_none()
        );

        Ok(())
    }

    /// Create a minimal recipe rooted in a fresh temporary directory.
    fn recipe_skeleton(mode: LmcppBuildInstallMode) -> LmcppRecipe {
        let tmp_dir = tempfile::tempdir().unwrap();

        let override_root = Some(ValidDir::new(&tmp_dir).unwrap());

        LmcppRecipe::new(
            "test_project",             // project
            &override_root,             // override_root
            3,                          // fail_limit
            "test-tag",                 // repo_tag
            &ComputeBackendConfig::Cpu, // compute_cfg
            &mode,
            &ArgSet::default(), // build_args
        )
        .expect("recipe construction must succeed")
    }

    /* ────────────────────────────────────────────
     * Finger-print consistency checks
     * ────────────────────────────────────────── */

    #[test]
    fn detects_status_mismatch() {
        let mut recipe = recipe_skeleton(LmcppBuildInstallMode::BuildOnly);

        // Pretend the binary was *installed* even though we are in BuildOnly mode.
        recipe.cfg.status = LmcppBuildInstallStatus::Installed;

        assert!(
            recipe.fingerprint_matches().is_err(),
            "status mismatch should be reported as an error"
        );
    }

    #[test]
    fn install_then_build_invalidates_fingerprint() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let root_dir = ValidDir::new(&tmp_dir).unwrap();

        /* ── 1. Simulate a successful *install-only* run ─────────────── */
        let mut install_recipe = LmcppRecipe::new(
            "test_project",
            &Some(root_dir.clone()),
            3,
            "test-tag",
            &ComputeBackendConfig::Cpu,
            &LmcppBuildInstallMode::InstallOnly,
            &ArgSet::default(),
        )
        .unwrap();

        install_recipe.cfg.status = LmcppBuildInstallStatus::Installed;
        install_recipe.store_cfg().unwrap(); // persist cache

        /* ── 2. A new *build-only* invocation must see drift ─────────── */
        let build_recipe = LmcppRecipe::new(
            "test_project".into(),
            &Some(root_dir),
            3,
            "test-tag".into(),
            &ComputeBackendConfig::Cpu,
            &LmcppBuildInstallMode::BuildOnly,
            &ArgSet::default(),
        )
        .unwrap();

        assert!(
            build_recipe.fingerprint_matches().is_err(),
            "install-only cache must be invalid when switching to build-only mode"
        );
    }

    /* ────────────────────────────────────────────
     * In-process locking guarantees
     * ────────────────────────────────────────── */

    #[test]
    fn in_process_lock_blocks_second_thread() {
        // Temporary directory stays alive for the whole test.
        let tmp_dir = tempfile::tempdir().unwrap();
        let root_pb = tmp_dir.path().to_path_buf(); // clone the PathBuf

        // Share the path between threads via Arc so every call can clone it.
        let root_arc = std::sync::Arc::new(root_pb);

        let new_recipe = {
            // Move one Arc clone *into* the closure so it owns the path ('static).
            let root_arc = root_arc.clone();
            move || {
                let root_dir = ValidDir::new(&*root_arc).unwrap(); // fresh ValidDir
                LmcppRecipe::new(
                    "lock_test",
                    &Some(root_dir),
                    3,
                    "v1",
                    &ComputeBackendConfig::Cpu,
                    &LmcppBuildInstallMode::BuildOrInstall,
                    &ArgSet::default(),
                )
                .unwrap()
            }
        };

        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));

        /* Thread A – acquires the lock and holds it briefly */
        let barrier_a = barrier.clone();
        let handle_a = std::thread::spawn({
            let new_recipe = new_recipe.clone(); // closure is `Fn`, therefore `Clone`
            move || {
                let recipe = new_recipe();
                let _guard = recipe.lock_file().unwrap(); // owns both guards
                barrier_a.wait(); // let B proceed
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
        });

        /* Thread B – must fail with “already being modified by another thread” */
        let barrier_b = barrier.clone();
        let handle_b = std::thread::spawn(move || {
            barrier_b.wait(); // wait for A
            let recipe = new_recipe();
            let err = recipe
                .lock_file()
                .expect_err("second thread should be blocked by in‑process mutex");

            assert!(
                err.to_string()
                    .contains("already being modified by another thread"),
                "unexpected error: {err}"
            );
        });

        handle_a.join().unwrap();
        handle_b.join().unwrap();
    }

    /* ────────────────────────────────────────────
     * `resolve_root` precedence & permissions
     * ────────────────────────────────────────── */

    #[test]
    fn resolve_root_precedence_and_permissions() -> LmcppResult<()> {
        let override_dir = tempfile::tempdir().unwrap();
        let env_dir = tempfile::tempdir().unwrap();

        let env_key = LmcppRecipe::LLAMA_CPP_ENV_OVERRIDE;

        /* 1️⃣  Explicit override wins over everything */
        unsafe {
            std::env::set_var(env_key, env_dir.path());
        }

        let override_valid = ValidDir::new(override_dir.path())?;
        let dir = LmcppRecipe::resolve_root(Some(&override_valid), "root_test")?;

        // ── Canonicalise *both* sides before comparing ──
        let expected = std::fs::canonicalize(override_dir.path()).unwrap();
        assert_eq!(dir.as_ref(), expected);

        /* 2️⃣  Environment variable wins when no override is given             */
        let dir = LmcppRecipe::resolve_root(None, "root_test")?;
        let expected = std::fs::canonicalize(env_dir.path()).unwrap();
        assert_eq!(dir.as_ref(), expected);

        /* 3️⃣  Platform data-dir is fallback when nothing is set               */
        unsafe {
            std::env::remove_var(env_key);
        }
        let dir = LmcppRecipe::resolve_root(None, "root_test")?;
        let proj_dirs = directories::ProjectDirs::from("com", "root_test", "llama_cpp").unwrap();
        assert_eq!(
            dir.as_ref(),
            std::fs::canonicalize(proj_dirs.data_dir()).unwrap()
        );

        /* 4️⃣  Unix: directory is chmod 755                                    */
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o755, "platform data dir should be chmod-ed to 755");
        }

        Ok(())
    }
}
