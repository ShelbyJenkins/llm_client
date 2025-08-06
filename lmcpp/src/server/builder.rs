//! Server – Builder
//! =================
//!
//! High‑level entry‑point that orchestrates the
//! **compile → spawn → connect → handle** lifecycle of a `llama.cpp` server.
//! For complete semantics see [`LmcppServerLauncher`].

use std::path::PathBuf;

use bon::Builder;

use crate::{
    LmcppServer,
    error::{LmcppError, LmcppResult},
    server::{
        handle::{DownloadBudget, LoadBudget, RetryDelay},
        ipc::{ServerClient, http::HttpClient, uds::UdsClient},
        toolchain::builder::LmcppToolChain,
        types::start_args::ServerArgs,
    },
};

/// Fluent one‑shot *builder & bootstrapper* that
///
/// 1. **Builds / validates** a `llama‑server` binary through [`LmcppToolChain`].  
/// 2. **Determines transport** deterministically  
///    * **UDS** ­when *all* of `webui`, `http`, `host`, and `port` are unset.  
///    * **HTTP** in every other case (including explicit `--webui`).  
/// 3. **Spawns or attaches** to the process and blocks until the model
///    reaches the `RunningModel` state, mirroring the server’s internal
///    state‑machine (`Loading` → `RunningModel` → `ErrorOrOffline`).
///
/// ### Deterministic budgets
/// * **`DownloadBudget`** – ceiling for fetching remote weights.  
/// * **`LoadBudget`**     – max time to map weights & build a KV‑cache.  
/// * **`RetryDelay`**     – polling cadence for `/health` & `/props`.  
///
/// ### Concurrency & safety
/// [`LmcppToolChain::run`] uses **cross‑process locks** on the cache
/// directory, so multiple threads—or even distinct programs—may call
/// [`load`](Self::load) concurrently without corrupting the build cache.
///
/// ### Related modules
/// * **Server – Handle** – runtime health & state semantics.  
/// * **Server Types**    – CLI surface forwarded through [`ServerArgs`].  
/// * **Tool‑chain**      – binary discovery, build, and install logic.  
#[derive(Debug, Clone, Builder)]
pub struct LmcppServerLauncher {
    /// Tool‑chain that guarantees a valid `llama‑server` binary
    /// (builds, installs, or re‑uses a cached one).
    #[builder(default = LmcppToolChain::default())]
    pub toolchain: LmcppToolChain,

    /// Strongly‑typed CLI configuration forwarded verbatim to the server
    /// process (e.g. `hf_repo`, `host`, `port`, `no_webui`).
    #[builder(default = ServerArgs::default())]
    pub server_args: ServerArgs,

    /// Force‑enable the built‑in Web UI.  
    /// Overrides `server_args.no_webui` **and** selects the HTTP transport.
    #[builder(default)]
    pub webui: bool,

    /// Explicit request for the HTTP transport without toggling the Web UI.
    /// Ignored if `webui` is `true` or if `host`/`port` are preset.
    #[builder(default)]
    pub http: bool,

    /// Upper bound on time spent mapping model weights and creating the
    /// KV‑cache (mirrors the `Loading` → `RunningModel` transition).
    #[builder(default, into)]
    pub load_budget: LoadBudget,

    /// Hard ceiling for fetching remote GGUF weights before giving up.
    #[builder(default, into)]
    pub download_budget: DownloadBudget,

    /// Polling interval for health probes while waiting for the model to
    /// become ready.
    #[builder(default, into)]
    pub retry_delay: RetryDelay,
}

impl Default for LmcppServerLauncher {
    fn default() -> Self {
        LmcppServerLauncher::builder().build()
    }
}

impl<S> LmcppServerLauncherBuilder<S>
where
    S: lmcpp_server_launcher_builder::IsComplete,
{
    /// Compile (or reuse) the binary, decide transport, spawn / attach,
    /// and block until the model is healthy.  
    /// Returns a fully‑initialised [`LmcppServer`] handle.
    pub fn load(self) -> LmcppResult<LmcppServer> {
        self.build().load()
    }
}

impl LmcppServerLauncher {
    /// Compile (or reuse) the binary, decide transport, spawn / attach,
    /// and block until the model is healthy.  
    /// Returns a fully‑initialised [`LmcppServer`] handle.
    pub fn load(&self) -> LmcppResult<LmcppServer> {
        let toolchain_result = self.toolchain.run()?;
        let msg = format!("Toolchain:{}\nResult:{}", self.toolchain, toolchain_result,);
        if toolchain_result.error.is_some() || toolchain_result.bin_path().is_none() {
            eprintln!("{msg}");
            crate::error!(msg);
            return Err(LmcppError::BuildFailed(msg));
        }

        let bin_path = match toolchain_result.bin_path() {
            Some(path) => path.to_path_buf(),
            None => {
                eprintln!("{msg}");
                crate::error!(msg);
                return Err(LmcppError::BuildFailed(
                    "Toolchain did not produce a binary path".to_string(),
                ));
            }
        };
        let bin_dir = match toolchain_result.bin_dir() {
            Some(path) => path.to_path_buf(),
            None => {
                eprintln!("{msg}");
                crate::error!(msg);
                return Err(LmcppError::BuildFailed(
                    "Toolchain did not produce a binary directory".to_string(),
                ));
            }
        };
        println!("{msg}");
        crate::info!(msg);
        let executable_name = toolchain_result.executable_name.clone();

        let mut server_args = self.server_args.clone();

        if self.webui {
            server_args.no_webui = false;
        }
        if self.webui || self.http || server_args.host.is_some() || server_args.port.is_some() {
            return self.load_http(server_args, bin_path, bin_dir, executable_name);
        } else {
            return self.load_uds(server_args, bin_path, bin_dir, executable_name);
        }
    }

    fn load_uds(
        &self,
        mut server_args: ServerArgs,
        bin_path: PathBuf,
        bin_dir: PathBuf,
        executable_name: String,
    ) -> LmcppResult<LmcppServer> {
        debug_assert!(
            !self.http && self.server_args.host.is_none() && self.server_args.port.is_none(),
            "load_uds should only be called when http is false and host/port are not set"
        );

        let client = UdsClient::new(&executable_name)?;

        server_args.host = Some(client.host());

        LmcppServer::new(
            bin_path,
            bin_dir,
            server_args,
            self.load_budget.clone(),
            self.download_budget.clone(),
            self.retry_delay.clone(),
            Box::new(client),
        )
    }

    fn load_http(
        &self,
        mut server_args: ServerArgs,
        bin_path: PathBuf,
        bin_dir: PathBuf,
        executable_name: String,
    ) -> LmcppResult<LmcppServer> {
        debug_assert!(
            self.webui || self.http || server_args.host.is_some() || server_args.port.is_some(),
            "load_http should only be called when http is true, webui is true, or host/port are set"
        );

        let client = HttpClient::new(
            &executable_name,
            server_args.host.as_deref(),
            server_args.port,
        )?;

        if let Some(args_host) = server_args.host.as_deref() {
            debug_assert_eq!(
                args_host,
                &client.host(),
                "Host in server_args should match the client's host"
            );
        } else {
            server_args.host = Some(client.host());
        }

        if let Some(args_port) = server_args.port {
            debug_assert_eq!(
                args_port, client.port,
                "Port in server_args should match the client's port"
            );
        } else {
            server_args.port = Some(client.port);
        }

        LmcppServer::new(
            bin_path,
            bin_dir,
            server_args,
            self.load_budget.clone(),
            self.download_budget.clone(),
            self.retry_delay.clone(),
            Box::new(client),
        )
    }
}

#[cfg(test)]
mod tests {

    use crate::*;

    #[test]
    #[ignore]
    fn gpt_oss_demo() -> LmcppResult<()> {
        let server = LmcppServerLauncher::builder()
            .toolchain(LmcppToolChain::builder().curl_enabled(true).build()?)
            .server_args(
                ServerArgs::builder()
                    .hf_repo("unsloth/gpt-oss-20b-GGUF")?
                    .reasoning_format(ReasoningFormat::None)
                    .jinja(true)
                    .build(),
            )
            .load()?;

        let res = server.completion(
            CompletionRequest::builder()
                .prompt("Tell me a joke about Rust.")
                .n_predict(256),
        )?;
        let content = res.content.unwrap();
        assert!(
            content.contains("Rust is like a well-typed joke: it takes a while to get it, but when you do, it's solid!"),
            "Expected funny, got: {content}"
        );
        server.stop()?;
        Ok(())
    }

    #[test]
    #[ignore]
    fn webui_gpt_oss_demo() -> LmcppResult<()> {
        let _server = LmcppServerLauncher::builder()
            .toolchain(LmcppToolChain::builder().curl_enabled(true).build()?)
            .server_args(
                ServerArgs::builder()
                    .hf_repo("ggml-org/gpt-oss-20b-GGUF")?
                    .reasoning_format(ReasoningFormat::None)
                    .ctx_size(0)
                    .flash_attn(true)
                    .jinja(true)
                    .build(),
            )
            .webui(true)
            .load()?;

        println!("Web UI server started. Open your browser to the provided URL.");
        std::thread::park();
        Ok(())
    }

    #[test]
    #[ignore]
    fn launch_example() -> LmcppResult<()> {
        let server = LmcppServerLauncher::builder()
            .toolchain(LmcppToolChain::builder().curl_enabled(true).build()?)
            .server_args(
                ServerArgs::builder()
                    .hf_repo("bartowski/google_gemma-3-1b-it-qat-GGUF")?
                    .build(),
            )
            .load()?;

        let res = server.completion(
            CompletionRequest::builder()
                .prompt("Tell me a joke about Rust.")
                .n_predict(128),
        )?;
        println!("Completion response: {:#?}", res.content);
        Ok(())
    }

    #[test]
    #[ignore]
    fn webui_example() -> LmcppResult<()> {
        let _server = LmcppServerLauncher::builder()
            .toolchain(LmcppToolChain::builder().curl_enabled(true).build()?)
            .server_args(
                ServerArgs::builder()
                    .hf_repo("bartowski/google_gemma-3-1b-it-qat-GGUF")?
                    .build(),
            )
            .webui(true)
            .load()?;

        // This is a placeholder; actual web UI testing would require a different approach. Park a thread?
        println!("Web UI server started. Open your browser to the provided URL.");
        std::thread::park();
        Ok(())
    }
}
