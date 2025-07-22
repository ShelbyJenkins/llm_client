//! Lmcpp Server CLI — Binary crate entry‑point
//! ===========================================
//!
//! Thin, ergonomic command‑line wrapper around the `lmcpp` orchestration layer.
//! Its job is to **locate or build** a `llama.cpp` server binary, **spawn or
//! attach** to the process, and then **block until the model is healthy** – all
//! from one command.  In addition it exposes a few convenience sub‑commands for
//! everyday operations such as listing or terminating running servers.
//!
//! ## Typical usage
//! ```text
//! # Start a local model (Unix Domain Socket by default)
//! $ lmcpp-server-cli -m ./google_gemma-3-1b-it-qat-Q4_K_M.gguf
//!
//! # Same model but enable the bundled Web‑UI
//! $ lmcpp-server-cli -m ./google_gemma-3-1b-it-qat-Q4_K_M.gguf --webui
//!
//! # Remote model, download if missing, time‑box heavy operations
//! $ lmcpp-server-cli -u https://huggingface.co/bartowski/google_gemma-3-1b-it-qat-GGUF/blob/main/google_gemma-3-1b-it-qat-Q4_K_M.gguf
//!
//! # Maintenance helpers
//! $ lmcpp-server-cli pids      # list running servers
//! $ lmcpp-server-cli kill-all  # send SIGTERM to all servers
//! ```
//!
//! The binary is intentionally *stateless*: every invocation fully describes
//! the desired outcome via flags and then exits when the underlying process
//! stops (Ctrl‑C or budget exceeded).  See the item‑level documentation below
//! for semantics of each flag.

// cargo run --bin lmcpp-server-cli -- --webui

use std::{path::PathBuf, time::Duration};

use clap::{Args, Parser, Subcommand};
use lmcpp::*;

#[derive(Debug, Parser)]
#[command(name = "lmcpp-server-cli", version)]
struct Cli {
    /// Re‑use all “recipe” flags from the tool‑chain CLI
    #[command(flatten)]
    recipe: RecipeSpec,

    /// Where the model comes from (exactly one)
    #[command(flatten)]
    model: ModelSource,

    /// Expose HTTP even if no host/port given
    #[arg(long)]
    http: bool,

    /// Enable built‑in Web‑UI
    #[arg(long)]
    webui: bool,

    // Optional tuning knobs
    #[command(flatten)]
    timing: Budgets,

    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Debug, Subcommand)]
enum Cmd {
    /// Show PIDs of all running llama.cpp servers
    Pids,

    /// Send a terminate signal to every running server process
    KillAll,
}

#[derive(Debug, Args)]
struct ModelSource {
    /// Local GGUF file
    #[arg(long, short = 'm', value_name = "PATH", conflicts_with = "model_url")]
    local_model_path: Option<PathBuf>,

    /// Remote URL (download & cache)
    #[arg(
        long,
        short = 'u',
        value_name = "URL",
        conflicts_with = "local_model_path"
    )]
    model_url: Option<url::Url>,
}

#[derive(Debug, Args)]
struct Budgets {
    #[arg(long, default_value_t = 120)]
    load_budget_secs: u32,
    #[arg(long, default_value_t = 300)]
    download_budget_secs: u32,
}

fn main() -> LmcppResult<()> {
    let cli = Cli::parse();

    match cli.cmd {
        Some(Cmd::Pids) => {
            let pids = get_all_server_pids(LMCPP_SERVER_EXECUTABLE);
            if pids.is_empty() {
                println!("No running `{}` processes.", LMCPP_SERVER_EXECUTABLE);
            } else {
                println!("Running `{}` instances: {pids:?}", LMCPP_SERVER_EXECUTABLE);
            }
            return Ok(());
        }

        Some(Cmd::KillAll) => {
            kill_all_servers(LMCPP_SERVER_EXECUTABLE)?;
            println!(
                "Sent termination signal to all `{}` processes.",
                LMCPP_SERVER_EXECUTABLE
            );
            return Ok(());
        }

        None => { /* fall through to normal launch path */ }
    }

    let toolchain = LmcppToolChain::builder()
        .compute_backend(cli.recipe.backend)
        .maybe_repo_tag(cli.recipe.repo_tag)
        .build_install_mode(cli.recipe.mode)
        .build_args(cli.recipe.build_args)
        .build()?;

    let builder = LmcppServerLauncher::builder()
        .toolchain(toolchain)
        .webui(cli.webui)
        .http(cli.http)
        .load_budget(LoadBudget(Duration::from_secs(
            cli.timing.load_budget_secs.into(),
        )))
        .download_budget(DownloadBudget(Duration::from_secs(
            cli.timing.download_budget_secs.into(),
        )));

    let server = if let Some(p) = cli.model.local_model_path {
        builder
            .server_args(ServerArgs::builder().model(p)?.build())
            .load()?
    } else if let Some(url) = cli.model.model_url {
        builder
            .server_args(ServerArgs::builder().model_url(url)?.build())
            .load()?
    } else {
        builder.server_args(ServerArgs::default()).load()?
    };

    println!("✅ server running: {server}");
    println!("Press Ctrl-C to stop.");

    let (tx, rx) = std::sync::mpsc::channel();

    ctrlc::set_handler(move || {
        let _ = tx.send(());
    })
    .map_err(|e| LmcppError::Internal(format!("Failed to set Ctrl-C handler: {e}")))?;

    let _ = rx.recv();

    println!("🔻 server stopped");
    Ok(())
}
