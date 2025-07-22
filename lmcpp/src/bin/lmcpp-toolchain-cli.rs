//! Lmcpp Toolchain CLI — manage local builds of **llama.cpp**
//! ==================================================
//!
//! A thin command-line wrapper around the `campion_toolchain` library that
//! **downloads, builds, validates, and removes** cached copies of *llama.cpp*
//! for a chosen compute backend (CPU, CUDA, Metal, …).
//!
//! ---
//! ## Building the binary
//! ```bash
//! cargo build --bin lmcpp-toolchain-cli
//! ```
//!
//! ## Quick start
//! Install the latest CPU-only build into the default cache root:
//! ```bash
//! cargo run --bin lmcpp-toolchain-cli -- install
//! ```
//!
//! ---
//! ## Subcommands
//! | Command           | What it does                                                                                 |
//! |-------------------|----------------------------------------------------------------------------------------------|
//! | **install**       | Build *or* fetch a pre-built artefact, then cache it. Re-runs only if the spec changed.      |
//! | **validate**      | Check that a cached build exists **and** matches the requested spec; prints a status report. |
//! | **remove**        | Delete the cached files for the requested spec.                                              |
//!
//! (Add `--help` after any subcommand to see all flags.)
//!
//! ---
//! ## Shared *recipe* flags
//! | Flag &nbsp;(*repeatable?*)           | Default                | Purpose                                                     |
//! |-------------------------------------|-------------------------|-------------------------------------------------------------|
//! | `--repo-tag <TAG>`                  | library default         | Git tag or commit to check out.                             |
//! | `--backend <BACKEND>`               | `default`               | Target hardware: `cpu`, `cuda`, `cuda-if-available`, …      |
//! | `--mode <MODE>`                     | `build-or-install`      | Choose build-from-source, install-only, or auto.            |
//! | `--build-arg <FLAG>` *(repeatable)* | _(none)_                | Extra `-D…` CMake flags forwarded verbatim.                 |
//!
//! ## Global flags
//! | Flag                 | Default path / value     | Purpose                                                   |
//! |----------------------|---------------------------|-----------------------------------------------------------|
//! | `--root <PATH>`      | platform data directory   | Override the cache / build root.                          |
//! | `--project <NAME>`   | `campion_toolchain`       | Sub-directory name under the data dir.                    |
//!
//! ---
//! ## Examples
//! *Build a tagged CUDA build into `/tmp/campion`:*
//! ```bash
//! lmcpp-toolchain-cli install --repo-tag v0.2.1 --backend cuda --root /tmp/campion
//! ```
//!
//! *Validate an existing Metal build:*
//! ```bash
//! lmcpp-toolchain-cli validate --backend metal
//! ```
//!
//! *Nuke the default spec from the cache:*
//! ```bash
//! lmcpp-toolchain-cli remove --backend default
//! ```
//!
//! ---
//! ## Exit codes
//! * `0` — success / up-to-date  
//! * `1` — validation or build error  
//! * `2` — argument parsing error (from **clap**)  
//!

use lmcpp::*;

#[derive(Debug, clap::Parser)]
#[command(name = "lmcpp-toolchain-cli", version)]
struct Cli {
    /// Override the cache / build root that would otherwise live
    /// in the per‑user data directory.
    #[arg(long, value_name = "PATH")]
    root: Option<std::path::PathBuf>,

    /// Project name used when resolving per‑user data directories
    /// (defaults to “campion_toolchain”).
    #[arg(long)]
    project: Option<String>,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Debug, clap::Subcommand)]
enum Cmd {
    /// Build *or* install (update) the requested version.
    Install {
        #[command(flatten)]
        spec: RecipeSpec,
    },

    /// Validate that the cached version is present and up‑to‑date.
    Validate {
        #[command(flatten)]
        spec: RecipeSpec,
    },

    /// Delete the cached version completely.
    Remove {
        #[command(flatten)]
        spec: RecipeSpec,
    },
}

fn main() -> LmcppResult<()> {
    let cli = <Cli as clap::Parser>::parse();

    match &cli.cmd {
        // ─────────────────────────── Install / Update ──────────────────────────
        Cmd::Install { spec } => {
            let builder = build_recipe(spec, &cli)?;

            // Run the full build / install workflow.
            let outcome = builder.run()?;
            println!("{outcome}");
        }

        // ──────────────────────────── Validate only ────────────────────────────
        Cmd::Validate { spec } => {
            let builder = build_recipe(spec, &cli)?;
            let report = builder.validate()?;
            println!("{report}");
        }

        // ───────────────────────────── Remove / Purge ──────────────────────────
        Cmd::Remove { spec } => {
            let builder = build_recipe(spec, &cli)?;
            let report = builder.validate()?;
            let builder = build_recipe(spec, &cli)?;
            builder.remove()?;
            println!("removed: {report}");
        }
    }

    Ok(())
}

fn build_recipe(spec: &RecipeSpec, cli: &Cli) -> LmcppResult<LmcppToolChain> {
    LmcppToolChain::builder()
        .compute_backend(spec.backend)
        .build_install_mode(spec.mode.clone())
        .maybe_repo_tag(spec.repo_tag.clone())
        .maybe_override_root(cli.root.clone())?
        .maybe_project(cli.project.clone())
        .build_args(spec.build_args.clone())
        .build()
}
