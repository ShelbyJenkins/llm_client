//! lmcpp – `llama.cpp`'s [`llama-server`](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) for Rust
//! =============================================================================================================
//!
//! ## Fully Managed
//! - **Automated Toolchain** – Downloads, builds, and manages the `llama.cpp` toolchain with [`LmcppToolChain`].  
//! - **Supported Platforms** – Linux, macOS, and Windows with CPU, CUDA, and Metal support.  
//! - **Multiple Versions** – Each release tag and backend is cached separately, allowing you to install multiple versions of `llama.cpp`.
//!
//! ## Blazing Fast UDS
//! - **UDS IPC** – Integrates with `llama-server`’s Unix-domain-socket client on Linux, macOS, and Windows.  
//! - **Fast!** – Is it faster than HTTP? Yes. Is it *measurably* faster? Maybe.
//!
//! ## Fully Typed / Fully Documented
//! - **Server Args** – *All* `llama-server` arguments implemented by [`ServerArgs`].  
//! - **Endpoints** – Each endpoint has request and response types defined.
//! - **Good Docs** – Every parameter was researched to improve upon the original `llama-server` documentation.
//!
//! ## CLI Tools & Web UI
//! - **`lmcpp-toolchain-cli`** – Manage the `llama.cpp` toolchain: download, build, cache.  
//! - **`lmcpp-server-cli`**    – Start, stop, and list servers.  
//! - **Easy Web UI** – Use [`LmcppServerLauncher::webui`] to start with HTTP *and* the Web UI enabled.
//!
//! ---
//!
//! ```rust,no_run
//! use lmcpp::*;
//!
//! fn main() -> LmcppResult<()> {
//!     let server = LmcppServerLauncher::builder()
//!         .server_args(
//!             ServerArgs::builder()
//!                 .hf_repo("bartowski/google_gemma-3-1b-it-qat-GGUF")?
//!                 .build(),
//!         )
//!         .load()?;
//!
//!     let res = server.completion(
//!         CompletionRequest::builder()
//!             .prompt("Tell me a joke about Rust.")
//!             .n_predict(64),
//!     )?;
//!
//!     println!("Completion response: {:#?}", res.content);
//!     Ok(())
//! }
//! ```
//!
//! ```sh,no_run
//! cargo run --bin lmcpp-server-cli -- --webui
//! // Or with a specific model from URL:
//! cargo run --bin lmcpp-server-cli -- --webui -u https://huggingface.co/bartowski/google_gemma-3-1b-it-qat-GGUF/blob/main/google_gemma-3-1b-it-qat-Q4_K_M.gguf
//! // Or with a specific local model:
//! cargo run --bin lmcpp-server-cli -- --webui -l /path/to/local/model.gguf
//! ```
//!
//! ---
//!
//! ## How It Works
//!
//! ```text
//! Your Rust App
//!       │
//!       ├─→ LmcppToolChain        (downloads / builds / caches)
//!       │         ↓
//!       ├─→ LmcppServerLauncher   (spawns & monitors)
//!       │         ↓
//!       └─→ LmcppServer           (typed handle over UDS*)
//!                 │
//!                 ├─→ completion()       → text generation
//!                 └─→ other endpoints    → fill-in-the-middle
//! ```
//! *Windows transparently swaps in a named pipe.*
//!
//! ---
//!
//! ### Endpoints ⇄ Typed Helpers
//! | HTTP Route          | Helper on `LmcppServer` | Request type            | Response type          |
//! |---------------------|-------------------------|-------------------------|------------------------|
//! | `POST /completion`  | `completion()`          | [`CompletionRequest`]   | [`CompletionResponse`] |
//! | `POST /infill`      | `infill()`              | [`InfillRequest`]       | [`CompletionResponse`] |
//! | `POST /embeddings`  | `embeddings()`          | [`EmbeddingsRequest`]   | [`EmbeddingsResponse`] |
//! | `POST /tokenize`    | `tokenize()`            | [`TokenizeRequest`]     | [`TokenizeResponse`]   |
//! | `POST /detokenize`  | `detokenize()`          | [`DetokenizeRequest`]   | [`DetokenizeResponse`] |
//! | `GET  /props`       | `props()`               | –                       | [`PropsResponse`]      |
//! | *custom*            | `status()` ¹            | –                       | [`ServerStatus`]       |
//! | *Open AI*           | `open_ai_v1_*()`        | – [`serde_json::Value`] | [`serde_json::Value`]  |
//!
//! ¹ Internal helper for server health.
//!
//! ---
//! ## Supported Platforms
//! | Platform   | CPU | CUDA | Metal | Binary Sources       |
//! |------------|-----|------|-------|----------------------|
//! | Linux x64  | ✅ | ✅ | –  | Pre-built + Source |
//! | macOS ARM  | ✅ | –  | ✅ | Pre-built + Source |
//! | macOS x64  | ✅ | –  | ✅ | Pre-built + Source |
//! | Windows x64| ✅ | ✅ | –  | Pre-built + Source |
//!
//! ---

#[allow(unused_imports)]
use tracing::{Level, debug, error, info, span, trace, warn};

pub mod client;
pub mod error;
pub mod server;

pub use client::{
    completion::*,
    detokenize::*,
    embeddings::*,
    infill::*,
    props::*,
    tokenize::*,
    types::{completion::*, generation_settings::*},
};
pub use error::{LmcppError, LmcppResult};
pub use server::{builder::*, handle::*, process::*, toolchain::*, types::*};
