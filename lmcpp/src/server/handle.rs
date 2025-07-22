//! Server – Handle
//! ==============
//!
//! Runtime handle for a **llama.cpp** server process started by this crate.  
//! The handle owns the child process and exposes a typed client so the rest of
//! the crate can talk to the model without caring how it was launched.
//!
//! ## Core ideas
//! * **Process isolation** – the server lives in a separate process, keeping
//!   crashes and memory leaks away from the host application.
//! * **Deterministic boot** – we distinguish _loading_, _running_ and
//!   _unhealthy/offline_ states and fail fast when the wrong model is seen.
//! * **Time budgets** – callers choose how patient to be for downloads, model
//!   loading and individual polling retries.
//!
//! Dropping [`LmcppServer`] stops the external process automatically.

use std::{
    path::PathBuf,
    thread::sleep,
    time::{Duration, Instant},
};

use crate::{
    client::props::PropsResponse,
    error::{LmcppError, LmcppResult},
    server::{
        ipc::{ServerClient, ServerClientExt, error::ClientError},
        process::guard::ServerProcessGuard,
        types::start_args::ServerArgs,
    },
};

/// Observable state of a llama.cpp server obtained via `/health` and `/props`.
#[derive(PartialEq, Debug)]
pub enum ServerStatus {
    /// Model is ready to serve requests.  
    /// Inner string = canonicalised model filename (lower‑case, no extension).
    RunningModel(String),
    /// HTTP 503: the server is still mapping weights into memory.
    Loading,
    /// Unreachable or fatally mis‑configured server.  
    /// Contains a human‑readable diagnostic message.
    ErrorOrOffline(String),
}

/// Maximum time the caller is willing to wait for the **model to load** after
/// the server binary has started.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct LoadBudget(pub std::time::Duration);

impl Default for LoadBudget {
    fn default() -> Self {
        LoadBudget(Duration::from_secs(45))
    }
}

impl From<std::time::Duration> for LoadBudget {
    fn from(value: std::time::Duration) -> Self {
        LoadBudget(value)
    }
}

/// Maximum time the caller is willing to wait for a **model download** to
/// complete when the server launches with a remote source (HF repo / URL).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct DownloadBudget(pub std::time::Duration);

impl Default for DownloadBudget {
    fn default() -> Self {
        DownloadBudget(Duration::from_secs(600))
    }
}

impl From<std::time::Duration> for DownloadBudget {
    fn from(value: std::time::Duration) -> Self {
        DownloadBudget(value)
    }
}

/// Back‑off applied between repeated health probes during start‑up.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct RetryDelay(pub std::time::Duration);

impl Default for RetryDelay {
    fn default() -> Self {
        RetryDelay(Duration::from_millis(100))
    }
}

impl From<std::time::Duration> for RetryDelay {
    fn from(value: std::time::Duration) -> Self {
        RetryDelay(value)
    }
}

/// Handle representing a live llama.cpp server process.
///
/// *Owns* a [`ServerProcessGuard`] to shut the process down and a boxed
/// [`ServerClient`] to perform HTTP/IPC calls.
///
/// # Equality & Hashing
/// Two handles compare equal (and hash alike) when they refer to the same
/// `base_url` **and** the same `model_name`, irrespective of other fields.
#[derive(Debug)]
pub struct LmcppServer {
    /// Guarantees the external process is killed when the handle is dropped.
    pub guard: ServerProcessGuard,
    /// Low‑level HTTP/IPC client bound to the server’s base URL.
    pub client: Box<dyn ServerClient>,
    /// Canonicalised model filename the caller expects to be loaded.
    pub model_name: String,
    pub pidfile_path: PathBuf,
}

impl LmcppServer {
    /// Spawn a new llama.cpp server (or attach to one already booting) and wait
    /// until the desired model is fully loaded.
    ///
    /// * `executable_name` – filename (not path) of the server binary; used
    ///   only for PID discovery and error messages.
    /// * `bin_path` / `bin_dir` – absolute path to the executable and its
    ///   directory; required for process management on *some* platforms.
    /// * `server_args` – command‑line arguments describing the model source.
    /// * `load_budget` – time‑limit for model *loading* after the binary runs.
    /// * `download_budget` – time‑limit for model *download* (if applicable).
    /// * `retry_delay` – interval between `/health` polls.
    /// * `client` – pre‑configured HTTP/IPC client pointing at the _target_
    ///   host/port.
    ///
    /// # Errors
    /// * [`LmcppError::InvalidConfig`] for malformed model specs.
    /// * [`LmcppError::ServerLaunch`] if the server fails to start the correct
    ///   model within the given budgets.
    pub fn new(
        bin_path: PathBuf,
        bin_dir: PathBuf,
        mut server_args: ServerArgs,
        load_budget: LoadBudget,
        download_budget: DownloadBudget,
        retry_delay: RetryDelay,
        client: Box<dyn ServerClient>,
    ) -> LmcppResult<Self> {
        debug_assert!(
            server_args.host.is_some(),
            "Host should be set in LmcppServerBuilder."
        );

        let pid_id = client.pid_id();
        debug_assert!(!pid_id.is_empty(), "PID ID must not be empty");
        let pidfile = format!("{pid_id}.pid");
        debug_assert!(
            pidfile.len() <= 240,
            "PID ID must not exceed 240 characters"
        );
        debug_assert!(sanitize_filename::is_sanitized(&pidfile));
        let pidfile_path = bin_dir.join(pidfile);

        let model_name = if let Some(path) = &server_args.model {
            path.0
                .file_name()
                .and_then(|s| s.to_str())
                .map(|s| s.to_owned())
                .ok_or_else(|| LmcppError::InvalidConfig {
                    field: "model",
                    reason: format!(
                        "Model path `{}` has no filename component",
                        path.0.display()
                    ),
                })?
        } else if let Some(repo) = &server_args.hf_repo {
            // Format: "username/model" (ignore optional :quant suffix if you have one)
            repo.0
                .split_once('/')
                .map(|(_, model)| model.to_owned())
                .ok_or_else(|| LmcppError::InvalidConfig {
                    field: "hf_repo",
                    reason: format!(
                        "Hugging Face repo `{}` is missing the `user/model` slash",
                        repo.0
                    ),
                })?
        } else if let Some(url) = &server_args.model_url {
            url.0
                .path_segments()
                .and_then(|segments| segments.last())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_owned())
                .ok_or_else(|| LmcppError::InvalidConfig {
                    field: "model_url",
                    reason: format!("URL `{}` has no filename component", url.0),
                })?
        } else {
            return Err(LmcppError::InvalidConfig {
                field: "model",
                reason: "No model source specified (model, hf_repo, or model_url)".to_string(),
            });
        };

        let expect_download = server_args.hf_repo.is_some() || server_args.model_url.is_some();

        let alias = server_args
            .model_id
            .clone()
            .unwrap_or_else(|| model_name.clone());

        server_args.alias = Some(alias);

        match Self::server_status(&client, Duration::from_millis(500), retry_delay.0) {
            ServerStatus::Loading => {
                crate::error!(
                    "The client at that address is already loading a model. This shouldn't happen. Attempting to kill it before starting LmcppServer with correct model."
                );
                crate::server::process::kill::kill_by_client(&pidfile_path, &client.host())?;
            }
            ServerStatus::RunningModel(running_model_name) => {
                crate::error!(
                    "The client at that address is already running a model. This shouldn't happen. Expected: {}, got: {} Attempting to kill it before starting LmcppServer with correct model.",
                    model_name,
                    running_model_name
                );
                crate::server::process::kill::kill_by_client(&pidfile_path, &client.host())?;
            }
            ServerStatus::ErrorOrOffline(_) => (), // Expected
        };

        // let original = if !use_gpu {
        //     let original = std::env::var("CUDA_VISIBLE_DEVICES").ok();
        //     std::env::set_var("CUDA_VISIBLE_DEVICES", "");
        //     original
        // } else {
        //     None
        // };

        let mut guard = ServerProcessGuard::new(&bin_path, &bin_dir, &pidfile_path, &server_args)?;

        match Self::start_up_loop(
            &client,
            &mut guard,
            download_budget,
            load_budget,
            retry_delay,
            &model_name,
            expect_download,
        ) {
            Ok(()) => (),
            Err(e) => {
                crate::error!("Failed to start LmcppServer: {e}");
                return Err(e);
            }
        }

        let server = Self {
            guard,
            client,
            model_name,
            pidfile_path,
        };
        crate::trace!("Started LmcppServer: {server}");
        Ok(server)
    }

    /// Low‑level polling loop used by [`LmcppServer::new`]; returns once the
    /// server is **healthy and running the expected model** or a hard error
    /// occurs.
    fn start_up_loop(
        client: &Box<dyn ServerClient>,
        guard: &mut ServerProcessGuard,
        download_budget: DownloadBudget,
        load_budget: LoadBudget,
        retry_delay: RetryDelay,
        model_name: &str,
        expect_download: bool,
    ) -> LmcppResult<()> {
        // dynamic time‑budget
        let overall_budget = if expect_download {
            download_budget.0
        } else {
            load_budget.0
        };
        let retry_delay = retry_delay.0;
        let deadline = Instant::now() + overall_budget;
        loop {
            match Self::server_status(&client, Duration::from_secs(3), retry_delay) {
                ServerStatus::RunningModel(running_model_name)
                    if model_ids_match(&running_model_name, &model_name) =>
                {
                    return Ok(());
                }

                ServerStatus::RunningModel(other) => {
                    guard.stop()?;
                    return Err(LmcppError::ServerLaunch(format!(
                        "Server started with wrong model. Expected {}, got {}",
                        model_name, other
                    )));
                }

                // still loading the correct model → wait and retry
                ServerStatus::Loading => (),

                // this should fail instantly
                // however, when we download using llama.cpp the health endpoint returns this rather than "loading"
                ServerStatus::ErrorOrOffline(msg) => {
                    if !expect_download {
                        guard.stop()?;
                        return Err(LmcppError::ServerLaunch(format!(
                            "Server failed to start: {msg}"
                        )));
                    }
                    // in download mode: treat as “still starting” while time & process permit
                }
            }
            if Instant::now() >= deadline {
                guard.stop()?;
                return Err(LmcppError::ServerLaunch(format!(
                    "Timed out after {overall_budget:?} waiting for model to load"
                )));
            }
            sleep(retry_delay);
        }
    }

    /// Gracefully terminate the external process.
    ///
    /// Safe to call multiple times; idempotent after the first success.
    pub fn stop(&self) -> LmcppResult<()> {
        self.guard.stop()?;
        Ok(())
    }

    /// One‑shot probe translating the current `/health` + `/props` responses
    /// into a [`ServerStatus`] value.
    pub fn status(&self) -> ServerStatus {
        Self::server_status(
            &self.client,
            Duration::from_millis(1000),
            Duration::from_millis(100),
        )
    }

    /// Probe `/health` and `/props` with retries until `total_budget`
    /// is exhausted.
    ///
    /// Returns a [`ServerStatus`] snapshot that never blocks longer than
    /// `total_budget`.
    pub fn server_status(
        client: &Box<dyn ServerClient>,
        total_budget: Duration,
        retry_delay: Duration,
    ) -> ServerStatus {
        debug_assert!(
            retry_delay > Duration::ZERO,
            "Retry delay must be greater than zero"
        );
        debug_assert!(
            total_budget >= retry_delay,
            "Total budget must be greater than or equal to retry delay"
        );
        let deadline = Instant::now() + total_budget;
        loop {
            match client.get::<serde_json::Value>("/health") {
                // ── 200 OK ────────────────────────────────────────────────────────
                Ok(_) => break,
                Err(e) => {
                    if let ClientError::Remote {
                        code: 503,
                        message: _,
                    } = &e
                    {
                        return ServerStatus::Loading;
                    }
                    if Instant::now() >= deadline {
                        return ServerStatus::ErrorOrOffline(format!(
                            "Health check failed after {:?}: {e:?}",
                            total_budget
                        ));
                    }
                    crate::trace!("health check error ({e:?}); retrying in {retry_delay:?}");
                    sleep(retry_delay);
                }
            }
        }
        loop {
            // single-shot probe; no internal loop
            match client.get::<PropsResponse>("/props") {
                Ok(props) => {
                    let path = match &props.model_path {
                        Some(p) => p,
                        None => {
                            return ServerStatus::ErrorOrOffline(
                                "No model path in /props response".to_string(),
                            );
                        }
                    };
                    let file_osstr = match path.file_name() {
                        Some(f) => f,
                        None => {
                            return ServerStatus::ErrorOrOffline(format!(
                                "Model path `{}` has no filename component",
                                path.display()
                            ));
                        }
                    };

                    let file_str = match file_osstr.to_str() {
                        Some(s) => s,
                        None => {
                            return ServerStatus::ErrorOrOffline(format!(
                                "Model path `{}` is not valid UTF-8",
                                path.display()
                            ));
                        }
                    };

                    let model_name = file_str.to_ascii_lowercase();

                    return ServerStatus::RunningModel(model_name);
                }
                // any non-alive (or transport) error: retry if time remains
                Err(e) => {
                    if Instant::now() >= deadline {
                        return ServerStatus::ErrorOrOffline(format!(
                            "Health check failed after {:?}: {e:?}",
                            total_budget
                        ));
                    }
                    crate::trace!("health check not ready ({e:?}); retrying in {retry_delay:?}");
                    sleep(retry_delay);
                }
            }
        }
    }

    pub fn pid(&self) -> u32 {
        self.guard.pid()
    }

    #[cfg(test)]
    pub fn dummy() -> Self {
        Self {
            guard: ServerProcessGuard::dummy(),
            client: Box::new(super::ipc::uds::UdsClient::dummy()),
            model_name: "dummy_model".into(),
            pidfile_path: PathBuf::from("/tmp/dummy.pid"),
        }
    }
}

impl Drop for LmcppServer {
    fn drop(&mut self) {
        if let Err(e) = self.stop() {
            // Never panic inside Drop; just record the problem.
            crate::error!("Failed to stop LmcppServer during drop: {e}");
        }
    }
}

impl PartialEq for LmcppServer {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.client.pid_id() == other.client.pid_id() && self.model_name == other.model_name
    }
}

impl Eq for LmcppServer {}

impl std::hash::Hash for LmcppServer {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.client.pid_id().hash(state);
        self.model_name.hash(state);
    }
}

impl std::fmt::Display for LmcppServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LmcppServer {{ {}, model: {} }}",
            self.client, self.model_name
        )
    }
}

/// Decide whether two user‑supplied model identifiers refer to the **same**
/// model file.
///
/// The comparison is tolerant to:
/// * differing case (`Gemma-2B` vs `gemma-2b`);
/// * inserted/removed “gguf” tokens;
/// * different non‑alphanumeric separators (`-`, `_`, `:`);
/// * presence/absence of the final extension (`.gguf`, `.gguf:q4_k_m` …).
///
/// After canonicalisation the function:
/// * returns `true` on exact equality, **or**
/// * performs a longest‑common‑substring check and returns `true` when at
///   least 75 % of the **shorter** canonical string matches.
///
/// Intended for noisy inputs coming from CLIs or remote APIs.
pub fn model_ids_match(a: &str, b: &str) -> bool {
    debug_assert!(
        !a.is_empty() && !b.is_empty(),
        "Model identifiers must not be empty"
    );
    // --- canonicalise ------------------------------------------------------
    let canonicalise = |s: &str| {
        let stem = s.rsplit_once('.').map_or(s, |(prefix, _)| prefix);

        let mut out = String::with_capacity(stem.len());
        let mut last_us = false; // previous char was '_'
        for ch in stem.chars().map(|c| c.to_ascii_lowercase()) {
            // skip “gguf”
            if out.ends_with("ggu") && ch == 'f' {
                out.truncate(out.len() - 3); // drop the preceding "ggu"
                last_us = out.ends_with('_');
                continue;
            }

            let mapped = if ch.is_ascii_alphanumeric() { ch } else { '_' };
            if mapped == '_' && last_us {
                continue; // collapse duplicates
            }
            last_us = mapped == '_';
            out.push(mapped);
        }
        // trim leading/trailing '_'
        out.trim_matches('_').to_string()
    };

    let ca = canonicalise(a);
    let cb = canonicalise(b);
    if ca == cb {
        return true;
    }

    // --- approximate match (≥ 75 % LCS) ------------------------------------
    let (short, long) = if ca.len() <= cb.len() {
        (&ca, &cb)
    } else {
        (&cb, &ca)
    };
    let min_match = (short.len() * 75 + 99) / 100; // ceil(0.75 * len)

    // O(n²) scan – fine for short identifiers
    for len in (min_match..=short.len()).rev() {
        for i in 0..=short.len() - len {
            if long.contains(&short[i..i + len]) {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use std::hash::Hasher;

    use super::*;

    // ──────────────────────────────────────────────────────────────────────
    //   model_ids_match – table‑driven
    // ──────────────────────────────────────────────────────────────────────
    #[test]
    fn model_ids_match_cases() {
        let cases = [
            ("Gemma-3B-It-Q4_K_M.gguf", "gemma_3b_it_q4-k-m", true),
            (
                "google_gemma-3-1b-it-qat-GGUF:q4_k_m",
                "google_gemma-3-1b-it-qat-Q4_K_M.gguf",
                true,
            ),
            ("alpaca.gguf", "gemma_3b.gguf", false),
            (
                "Llama-3-8B-Instruct:q4_k_m",
                "llama-3_8b-instruct.Q4_K_M.gguf",
                true,
            ),
            ("Llama-3-8B-Instruct", "llama-3-8b-instruct.gguf", true),
            (
                "Mixtral-8x22B-Instruct-v0.1:q4_k_m",
                "mixtral-8x22b-instruct-v0_1.Q4_K_M.gguf",
                true,
            ),
            (
                "Mixtral-8x22B-Instruct-v0.1",
                "mixtral-8x22b-instruct-v0_1.Q4_K_M.gguf",
                true,
            ),
            (
                "Qwen2-72B-Instruct:q4_k_m",
                "qwen2-72b-instruct.q4_k_m.GGUF",
                true,
            ),
            (
                "Phi-3-mini-4k-instruct",
                "phi-3-mini-4k-instruct.Q8_0.gguf",
                true,
            ),
            (
                "Llama-3-8B-Instruct",
                "mixtral-8x22b-instruct-v0_1.Q4_K_M.gguf",
                false,
            ),
        ];
        for (a, b, expect) in cases {
            assert_eq!(model_ids_match(a, b), expect, "({a}, {b})");
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // PartialEq / Hash semantics
    // ──────────────────────────────────────────────────────────────────────
    #[test]
    fn lmcpp_server_inequality_and_hash() {
        use std::collections::hash_map::DefaultHasher;

        let a = LmcppServer::dummy();
        let mut b = LmcppServer::dummy();
        b.model_name = "different".into();

        assert_ne!(a, b);

        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        std::hash::Hash::hash(&a, &mut h1);
        std::hash::Hash::hash(&b, &mut h2);
        assert_ne!(h1.finish(), h2.finish());
    }
}
