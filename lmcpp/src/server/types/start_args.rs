//! Server Types – Server CLI Arguments
//! ===========================
//!
//! This module defines **all** command‑line configuration for the `lmcpp` HTTP
//! inference server.  It centralises the parsing, validation and ergonomic
//! construction of server start‑up arguments so that the binary can be driven
//! both from the shell and programmatically.
//!
//! ## Key concepts
//! * **`ServerArgs`** – a single struct representing every CLI flag / option
//!   understood by the server.  It derives [`cmdstruct::Command`] so it can
//!   populate the server start Command directly, **and** [`bon::Builder`] so it
//!   can be built fluently in code.
//! * **Type‑state builder** – `ServerArgsBuilder` encodes invariants (e.g. “at
//!   least one *model source* **must** be provided”) at compile time.  The
//!   generic `State` parameter tracks whether each required field is set;
//!   calling `.build()` is only possible when the state implements
//!   `IsComplete`.
//! * **CLI ↔ runtime symmetry** – every field is serialisable with `serde` so a
//!   parsed `ServerArgs` can be logged, persisted, or forwarded to another
//!   process without loss of information.
//!
//! ## Why a monolithic struct?
//! Keeping every flag in one place guarantees that:
//! * **Help output stays consistent** – `cmdstruct` renders `--help` directly
//!   from this definition.
//! * **Docs never drift** – field‑level doc comments double as CLI help, Rust
//!   docs and JSON schema descriptions.
//! * **Validation is local** – invariants such as mutual exclusivity of
//!   `cpu_mask` and `cpu_range`, or compile‑time enforcement of *exactly one*
//!   model source, are encoded right here instead of scattered checks.
//!
//! ## Feature highlights
//! * **Builder ergonomics** with automatic `into()` on `String` inputs,
//!   dramatically reducing boilerplate in higher‑level code.
//! * **Compile‑time safety**: attempting to call `.build()` without setting any
//!   model source fails to compile – a misuse caught before the program runs.
//! * **Display impl alignment rules**: downstream code that formats enums uses
//!   `writeln!` with left‑aligned keys for human‑friendly logs.

use std::{
    fmt::{self, Display},
    ops::Not,
};

use bon::Builder;
use cmdstruct::{Arg, Command};
use serde::{Deserialize, Serialize};

use super::model::*;
use crate::{
    client::types::generation_settings::Pooling,
    error::{LmcppError, LmcppResult},
};

pub const DEFAULT_MODEL_URL: &str = "https://huggingface.co/bartowski/google_gemma-3-1b-it-qat-GGUF/resolve/main/google_gemma-3-1b-it-qat-Q4_K_M.gguf";

pub const DEFAULT_HF_REPO: &str = "bartowski/google_gemma-3-1b-it-qat-GGUF:q4_k_m";

pub const DEFAULT_HF_FILE: &str = "bartowski/google_gemma-3-1b-it-qat-Q4_K_M.gguf";

#[derive(Serialize, Deserialize, Debug, Clone, Command, Builder)]
#[builder(derive(Debug, Clone), on(String, into), finish_fn(vis = "", name = build_internal))]
#[command(executable = "overwritten_at_runtime")]
pub struct ServerArgs {
    // ──────────────── Model selection & HF auth ─────────────────
    // Denotes whether the model source is set (local file, URL, or Hugging Face).
    #[allow(dead_code)]
    #[serde(skip_serializing)]
    #[builder(default, setters(vis = "pub", name = has_model_source_internal), getter)]
    has_model_source: bool,

    /// Human readable identifier for the model instance. Used to set the alias. If none supplied,
    /// the alias is derived from the model's path in the required `model` field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,

    /// Path to the local model file (GGUF or GGML format). If not set, you can use
    /// `hf_repo` or `model_url` to load a model from Hugging Face or a URL.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(setters(vis = "", name = model_internal))]
    #[arg(option = "-m")]
    pub model: Option<LocalModelPath>,

    /// Direct URL to download the model file. If provided, the server will fetch the
    /// model from this URL instead of a local file.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(setters(vis = "", name = model_url_internal))]
    #[arg(option = "-mu")]
    pub model_url: Option<ModelUrl>,

    /// Hugging Face model repository in the form `<user>/<model>[:quant]`. If set,
    /// the model is downloaded from Hugging Face (with optional quantization suffix).
    /// By default, if a multimodal projector (mmproj) is associated with the model,
    /// it will be downloaded as well unless `no_mmproj` is specified.  
    /// example: unsloth/phi-4-GGUF:q4_k_m
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(setters(vis = "", name = hf_repo_internal))]
    #[arg(option = "-hf")]
    pub hf_repo: Option<HfRepo>,

    /// Specific model file name to use from the Hugging Face repository. If set, this
    /// file (e.g., a particular quantized GGUF file) will be used instead of the
    /// default file in the repo.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(setters(vis = "", name = hf_file_internal))]
    #[arg(option = "-hff")]
    pub hf_file: Option<HfFile>,

    /// Hugging Face repository for the *draft* model (used in speculative decoding).
    /// Format is the same as `hf_repo` but points to a smaller, faster model that
    /// generates draft tokens ahead of the main model.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "-hfd")]
    pub hf_repo_draft: Option<String>,

    /// Hugging Face repository for the vocoder model (used for text-to-speech audio
    /// output). Provide this if you want the server to load a vocoder for TTS (not
    /// used unless audio output is needed).  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "-hfv")]
    pub hf_repo_v: Option<String>,

    /// Specific file name for the vocoder model from the Hugging Face repo. If set,
    /// this exact file will be used for the vocoder (overrides default selection).  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "-hffv")]
    pub hf_file_v: Option<String>,

    /// Hugging Face access token for downloading private models. If not provided,
    /// the `HF_TOKEN` environment variable is used by default.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "-hft")]
    pub hf_token: Option<String>,

    // ────────────────── Server identity / binding ───────────────────
    /// Custom alias to identify the model instance. This name may appear in the
    /// REST API (for example, in health or metadata endpoints) to tag which model is
    /// running.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "-a")]
    pub alias: Option<String>,

    /// Host address or UNIX socket path to bind the HTTP server to. Use an IP (like
    /// `"127.0.0.1"` or `"0.0.0.0"`) or a socket path ending in `.sock` for a UNIX
    /// domain socket. Defaults to `127.0.0.1` (localhost).  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--host")]
    pub host: Option<String>,

    /// TCP port for the HTTP server to listen on (default is 8080).  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--port")]
    pub port: Option<u16>,

    /// Filesystem path to serve static files from. If set, the server will serve
    /// files in this directory at the root URL (useful for a custom web UI). If not
    /// set, only the API endpoints and built-in UI (if enabled) are served.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--path")]
    pub static_path: Option<String>,

    /// Disable the built-in web interface. By default, the server provides a basic
    /// Web UI at the root URL; if this flag is set, only the API endpoints (e.g.,
    /// `/completion`, `/chat`) will be available.  
    #[serde(skip_serializing_if = "<&bool>::not")]
    #[builder(default = true)]
    #[arg(flag = "--no-webui")]
    pub no_webui: bool,

    // ─────────────────── Endpoint feature flags ─────────────────────
    /// Run in embedding-only mode. This enables only the embedding generation
    /// endpoint (e.g., `/embeddings`) and should be used with models intended for
    /// embedding tasks (no text completion will be served).  
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    #[arg(flag = "--embedding")]
    pub embeddings_only: bool,

    /// Enable the reranking endpoint. When set, the server provides an endpoint to
    /// score or rerank inputs (e.g., `/rerank`), using a model fine-tuned for ranking
    /// tasks.  
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    #[arg(flag = "--reranking")]
    pub reranking: bool,

    /// Enable the Prometheus-compatible metrics endpoint (`/metrics`). This allows
    /// scraping runtime metrics (like request rates, memory usage) for monitoring
    /// purposes.  
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    #[arg(flag = "--metrics")]
    pub metrics_endpoint: bool,

    /// Enable the slots monitoring endpoint. The `/slots` endpoint shows active
    /// conversation slots (parallel sessions) and their statuses. By default this
    /// is off; use this flag to expose it.  
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    #[arg(flag = "--slots")]
    pub slots_endpoint: bool,

    /// Enable the properties endpoint. When on, a `/props` endpoint is available to
    /// change global generation settings via POST (e.g., adjusting default parameters
    /// at runtime).  
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    #[arg(flag = "--props")]
    pub props_endpoint: bool,

    /// Directory path where slot state (KV cache for each conversation slot) will be
    /// saved. If specified, the server can persist and restore session states from
    /// this location (default is not to save slots).  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--slot-save-path")]
    pub slot_save_path: Option<String>,

    // ───────────────────── Logging & verbosity ─────────────────────
    /// Completely disable logging. No log output will be produced (either to console
    /// or file) when this flag is set.  
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    #[arg(flag = "--log-disable")]
    pub log_disable: bool,

    /// Path to a log file. If set, log messages will be written to this file. (If not
    /// set, logs are output to standard error or as configured).  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--log-file")]
    pub log_file: Option<String>,

    /// Enable colored log output. This will use ANSI color codes in log messages for
    /// easier reading (typically only effective in console output).  
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default = true)]
    #[arg(flag = "--log-colors")]
    pub log_colors: bool,

    /// Shorthand for maximum verbosity logging. Using `-v` (or `--verbose`) will
    /// set the log verbosity to the highest level, causing all debug and trace
    /// messages to be emitted.  
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    #[arg(flag = "-v")]
    pub verbose: bool,

    /// Set a specific verbosity level (log level threshold). Messages more verbose
    /// than this level will be suppressed. For example, `verbosity = 1` might show
    /// only warnings and errors if higher levels are used for debug/info.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "-lv")]
    pub verbosity: Option<u8>,

    /// Include a prefix in each log message (for instance, a timestamp or module
    /// tag). This can help identify where logs are coming from in the code.  
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default = true)]
    #[arg(flag = "--log-prefix")]
    pub log_prefix: bool,

    /// Prepend timestamps to each log line. When enabled, each log entry will start
    /// with a timestamp indicating when the message was logged.  
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default = true)]
    #[arg(flag = "--log-timestamps")]
    pub log_timestamps: bool,

    /// Print version information and exit. When this flag is used, the server will
    /// output its version/build info and terminate immediately (no serving).  
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    #[arg(flag = "--version")]
    pub version: bool,

    /// Output a Bash shell completion script to stdout. Use this to generate an
    /// autocompletion script that can be sourced in your shell for convenient CLI
    /// tab-completion.  
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    #[arg(flag = "--completion-bash")]
    pub completion_bash: bool,

    /// Print the full prompt context before generation. This includes any system or
    /// role prompts and the user's input after template processing. Useful for
    /// debugging to see exactly what the model is given.  
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    #[arg(flag = "--verbose-prompt")]
    pub verbose_prompt: bool,

    /// Number of threads to use for model inference (text generation). Setting this
    /// can optimize performance: e.g., use more threads for better throughput.
    /// `-1` means use an automatic default (likely the number of CPU cores).  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "-t")]
    pub threads: Option<u64>,

    /// Number of threads to use for the prompt processing and batching stage. This
    /// typically covers tokenization and initial prompt evaluation. Defaults to the
    /// same as `threads` if not set.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "-tb")]
    pub threads_batch: Option<u64>,

    // ────────────────── CPU affinity & priority ───────────────────
    /// CPU affinity mask for pinning threads, given as a hexadecimal bitmask. This
    /// lets you specify exact CPU cores to use (e.g., `"0x3f"` to allow cores 0–5).
    /// It complements `cpu_range` for fine-grained control.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "-C")]
    pub cpu_mask: Option<String>,

    /// CPU affinity range for threads, specified as `low-high`. Example: `0-3` to
    /// restrict threads to CPUs 0 through 3. Complements `cpu_mask` (only one or the
    /// other is typically used).  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "-Cr")]
    pub cpu_range: Option<String>,

    /// Enforce strict CPU placement according to the mask/range. If enabled, threads
    /// will be bound strictly to the specified cores and not scheduled elsewhere
    /// (useful for performance isolation).  
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    #[arg(flag = "--cpu-strict")]
    pub cpu_strict: bool,

    /// Set the process/threads scheduling priority (niceness level). Values:
    /// 0 = normal, 1 = medium, 2 = high, 3 = realtime. Higher values give the server
    /// more CPU time (use with care, realtime may require privileges).  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--prio")]
    pub prio: Option<u8>,

    /// Polling level for threads (0–100). This controls how actively threads spin
    /// waiting for work: 0 means no polling (yielding to OS), 100 means continuous
    /// busy-wait. The default is 50 (balanced).  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--poll")]
    pub poll: Option<u8>,

    /// CPU affinity mask for batch/prompt threads, in hex format. Overrides the main
    /// `cpu_mask` for the threads handling batching. Defaults to the same mask as
    /// general threads if not set.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "-Cb")]
    pub cpu_mask_batch: Option<String>,

    /// CPU affinity range for batch threads. Specifies allowed CPUs (e.g., `4-7` for
    /// batch threads to run on cores 4–7). If not set, uses the same range as normal
    /// threads.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "-Crb")]
    pub cpu_range_batch: Option<String>,

    /// Strict CPU affinity for batch threads. If true, batch/prompt threads will
    /// strictly adhere to `cpu_mask_batch`/`cpu_range_batch` settings. By default,
    /// it mirrors the `cpu_strict` setting.  
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    #[arg(flag = "--cpu-strict-batch")]
    pub cpu_strict_batch: bool,

    /// Scheduling priority for batch threads. Uses the same scale as `prio` (0=normal
    /// up to 3=realtime). If not set, inherits the main priority setting.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--prio-batch")]
    pub prio_batch: Option<u8>,

    /// Whether batch threads should use polling (busy-wait) or not. Accepts 0 or 1.
    /// If not specified, batch threads follow the same polling behavior as main
    /// threads (`poll` setting).  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--poll-batch")]
    pub poll_batch: Option<u8>,

    // ─────────────────── Context & batching params ─────────────────
    /// The context size (in tokens) for the model. This is the maximum combined length
    /// of prompt and generated output. Default is 4096, or 0 to use the model’s
    /// native context length from its metadata.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--ctx-size")]
    pub ctx_size: Option<u64>,

    /// Logical maximum batch size (number of tokens processed in one forward pass).
    /// This affects how many tokens are evaluated simultaneously. A higher batch size
    /// can improve throughput but uses more memory (default 2048).  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--batch-size")]
    pub batch_size: Option<u64>,

    /// Physical micro-batch size. This is the chunk size for splitting the batch into
    /// smaller parts for actual computation. A smaller `ubatch_size` reduces peak
    /// memory usage by processing the batch in pieces (default 512).  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--ubatch-size")]
    pub ubatch_size: Option<u64>,

    // ───────────────────── Feature toggles ────────────────────────
    /// Enable Flash Attention, an optimized attention mechanism. Improves speed and
    /// memory for supported models/hardware (requires that the model/backends support
    /// it). By default this is off.  
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    #[arg(flag = "--flash-attn")]
    pub flash_attn: bool,

    /// Disable internal performance measurements. Normally, the server may record
    /// timings (for prompts, generation, etc.) for logging or debugging; this flag
    /// turns off those measurements.  
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    #[arg(flag = "--no-perf")]
    pub no_perf: bool,

    /// Do not process escape sequences in prompt input. By default, sequences like
    /// `\\n` or `\\t` in the prompt text are interpreted as newline or tab; with this
    /// flag, they are left as literal backslash characters.  
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    #[arg(flag = "--no-escape")]
    pub no_escape: bool,

    // ─────────────────────── RoPE scaling ────────────────────────
    /// Method for RoPE (Rotary Position Embedding) scaling to extend context length.
    /// Options are: `"none"` (no scaling, use model default), `"linear"` (NTK linear
    /// scaling), or `"yarn"` (YaRN scaling). By default, if not specified, linear
    /// scaling is used unless the model defines otherwise.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--rope-scaling")]
    pub rope_scaling: Option<RopeScaling>,

    /// RoPE context scaling factor. A value > 1 increases the effective context
    /// length by that factor (for linear scaling). For example, 2.0 would double the
    /// context length if using linear NTK scaling.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--rope-scale")]
    pub rope_scale: Option<f32>,

    /// Base frequency for RoPE. This is used in NTK-aware scaling methods to adjust
    /// how positional frequencies are initialized. If not set, the model's built-in
    /// base frequency is used.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--rope-freq-base")]
    pub rope_freq_base: Option<f32>,

    /// Frequency scaling factor for RoPE (used in NTK-aware context extension). This
    /// effectively divides the rotation frequency by the given factor (making angles
    /// change more slowly), extending context by factor 1/`rope_freq_scale`.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--rope-freq-scale")]
    pub rope_freq_scale: Option<f32>,

    // ─────────────────────── YaRN tuning ─────────────────────────
    /// Original context size of the model for YaRN scaling. Set this to the model's
    /// trained context length if you want to override automatic detection. 0 means
    /// use the model's default training context size.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--yarn-orig-ctx")]
    pub yarn_orig_ctx: Option<u64>,

    /// YaRN extrapolation mix factor. Controls how new positional embeddings are
    /// interpolated vs extrapolated. `-1.0` uses the model's default. `0.0` means full
    /// interpolation (no true extrapolation), and higher values blend in more
    /// extrapolation.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--yarn-ext-factor")]
    pub yarn_ext_factor: Option<f32>,

    /// YaRN attention scaling factor. Adjusts the magnitude of attention for extended
    /// context (e.g., scaling by √t). Default 1.0 means standard; other values tweak
    /// attention strength in the extrapolated context.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--yarn-attn-factor")]
    pub yarn_attn_factor: Option<f32>,

    /// YaRN beta slow (α) parameter. Corresponds to the high-dimension correction
    /// term for extended context. Default 1.0 (no change); adjusting this affects how
    /// the slower components of positional encoding are corrected.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--yarn-beta-slow")]
    pub yarn_beta_slow: Option<f32>,

    /// YaRN beta fast (β) parameter. Controls the low-dimension correction term for
    /// extended context. Default 32.0; tuning this can affect how the faster-changing
    /// positional components are handled.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--yarn-beta-fast")]
    pub yarn_beta_fast: Option<f32>,

    #[builder(default)]
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    /// Dump the contents of the key-value cache to the logs. This is a verbose debug
    // ─────────────── KV-cache & memory / NUMA ────────────────────
    /// option that will print the entire KV cache (all stored attention keys/values),
    /// which can be very large.  
    #[arg(flag = "--dump-kv-cache")]
    pub dump_kv_cache: bool,

    #[builder(default)]
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    /// Keep the KV cache in main memory and do not offload it. Normally, if running

    /// on GPU, the KV (key/value) cache might be offloaded to CPU RAM to save VRAM.
    /// This flag prevents that, keeping the cache on the primary device.  
    #[arg(flag = "--no-kv-offload")]
    pub no_kv_offload: bool,

    /// Data type to use for the **key** part of the KV cache. Options include `f32`,
    /// `f16`, `bf16` (and various quantized types like `q4_0`, `q5_1`, etc.). Using
    /// lower precision or quantized types can save memory at some accuracy cost.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--cache-type-k")]
    pub cache_type_k: Option<String>,

    /// Data type for the **value** part of the KV cache. Supports the same options as
    /// `cache_type_k` (floating point or quantized types). Default is `f16` unless
    /// changed. Using quantized values reduces memory usage.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--cache-type-v")]
    pub cache_type_v: Option<String>,

    /// Threshold for KV cache defragmentation (fraction of free space). If the KV
    /// memory fragmentation exceeds this fraction, the cache will be defragmented.
    /// For example, 0.1 (10%) by default; set to a negative value to disable
    /// auto-defragmentation.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--defrag-thold")]
    pub defrag_thold: Option<f32>,

    /// Number of parallel sequences (requests) to decode simultaneously. This allows
    /// the model to handle N requests in parallel by dividing the context among them
    /// (e.g., for 2 parallel with context 8192, each gets ~4096 tokens).  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--parallel")]
    pub parallel: Option<u64>,

    /// Lock the model's memory into RAM (prevent it from being paged out or compressed
    /// by the OS). This can improve consistency of performance by avoiding OS swap,
    /// at the cost of using pinned memory.  
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    #[arg(flag = "--mlock")]
    pub mlock: bool,

    /// Do not memory-map the model file. Normally, memory-mapping allows on-demand
    /// loading of model parts. Disabling it forces the model to be fully loaded into
    /// memory upfront (slower startup, but can avoid certain paging issues).  
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    #[arg(flag = "--no-mmap")]
    pub no_mmap: bool,

    /// Strategy for NUMA (Non-Uniform Memory Access) optimizations on multi-node
    /// systems. Options: `"distribute"` (spread threads across NUMA nodes evenly),
    /// `"isolate"` (bind all threads to the initial NUMA node), `"numactl"` (use CPU
    /// placement from `numactl`). Use to improve performance on multi-socket servers.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--numa")]
    pub numa: Option<NumaStrategy>,

    // ────────────────── GPU off-load control ─────────────────────
    /// Specify the device(s) for GPU offloading. Provide a comma-separated list of
    /// device identifiers (backend-specific, e.g., `"cuda:0"` or `"metal:0"`). Use
    /// `--list-devices` to see available options. `"none"` means use CPU only.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--device")]
    pub device: Option<String>,

    /// List all available accelerator devices (GPUs, NPUs, etc.) and exit. Use this
    /// to get the identifiers for use with the `device` option (no model will be
    /// loaded when this flag is present).  
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    #[arg(flag = "--list-devices")]
    pub list_devices: bool,

    /// **Advanced:** Override the storage type of certain model tensors. Accepts a
    /// comma-separated list of overrides (format may be model-specific). For example,
    /// this could be used to force certain layers to use CPU memory. (*Note:* Use only
    /// if you understand the model's internals; not commonly needed).  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--override-tensor")]
    pub override_tensor: Option<Vec<String>>,

    /// Number of transformer layers to offload to the GPU. For example, `gpu_layers = 20`
    /// will place the first 20 layers on the GPU (or spread across GPUs as specified)
    /// and keep the rest on CPU. Use this to fit large models partially on GPU.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--gpu-layers")]
    pub gpu_layers: Option<u64>,

    /// Mode for splitting the model across multiple GPUs. Options: `"none"` (all on
    /// one GPU), `"layer"` (divide model layers among GPUs, splitting the KV cache as
    /// needed), `"row"` (split tensors within layers by rows across GPUs). Default is
    /// `layer` when multi-GPU is used.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--split-mode")]
    pub split_mode: Option<SplitMode>,

    /// Specify fractions of the model to allocate to each GPU when using multiple
    /// GPUs. Provide a comma-separated list of relative proportions for each GPU.
    /// E.g., `"3,1"` to allocate 75% of layers to GPU0 and 25% to GPU1 (total ratio).  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--tensor-split")]
    pub tensor_split: Option<String>,

    /// Index of the primary GPU. When `split_mode` is "none", this GPU is used for the
    /// entire model. When using row-splitting, this GPU also handles intermediate
    /// results and the KV cache. Default is 0 (the first GPU).  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--main-gpu")]
    pub main_gpu: Option<u32>,

    /// Verify the model's tensor values after loading. If enabled, the server will
    /// check for invalid values (NaNs, infinities) in model tensors and report any
    /// anomalies. Useful to detect corrupted model files.  
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    #[arg(flag = "--check-tensors")]
    pub check_tensors: bool,

    /// **Advanced:** Override a model metadata key-value pair. Format:
    /// `key=type:value`. For example, `--override-kv tokenizer.ggml.add_bos_token=bool:false`
    /// can override whether a beginning-of-stream token is added. This
    /// can be repeated for multiple overrides.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--override-kv")]
    pub override_kv: Option<Vec<String>>,

    // ─────────────── LoRA / control-vector loading ────────────────
    /// Path to a LoRA adapter file to load. LoRA adapters apply learned weight
    /// deltas to the model to fine-tune it. This option can be used multiple times to
    /// load and merge several LoRA adapters at startup.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--lora")]
    pub lora: Option<Vec<String>>,

    /// Path to a control vector file to add to the model. Control vectors are like
    /// trainable prompts or biases that can steer model behavior (for example, style
    /// or content filtering cues). Multiple control vectors can be added.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--control-vector")]
    pub control_vector: Option<Vec<String>>,

    // ─────────────────── Inference-mode toggles ───────────────────
    /// Disable context shifting in infinite generation scenarios. Normally, if the
    /// generated text exceeds the context length, the oldest tokens are dropped (shifted
    /// out) to make room for new ones. With this flag, the model will stop instead of
    /// shifting the context.  
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    #[arg(flag = "--no-context-shift")]
    pub no_context_shift: bool,

    /// Allow special tokens to be generated in the output. Special tokens (like
    /// end-of-sequence or control codes) are usually prevented from appearing in
    /// normal text output. Enabling this allows them to appear (if the model would
    /// generate them).  
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    #[arg(flag = "-sp")]
    pub special_tokens: bool,

    /// Skip the initial warm-up run. By default, the server might run a dummy
    /// inference on startup to allocate buffers and optimize cache usage. Using
    /// `--no-warmup` prevents this extra initial pass, reducing startup time.  
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    #[arg(flag = "--no-warmup")]
    pub no_warmup: bool,

    /// Enable SPM infill mode (Suffix-Prefix-Middle pattern). In this mode, if the
    /// prompt is structured for fill-in-the-middle tasks, the model will interpret it
    /// as such. This flag switches the expected prompt format from the default
    /// Prefix...Suffix to Suffix...Prefix around a middle gap.  
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    #[arg(flag = "--spm-infill")]
    pub spm_infill: bool,

    /// Pooling strategy for embedding vectors.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--pooling")]
    pub pooling: Option<Pooling>,

    /// Disable continuous (dynamic) batching of incoming requests. Normally, the
    /// server can aggregate multiple prompt requests and process them together to
    /// improve throughput. Using `--no-cont-batching` forces each request to be
    /// handled independently (may reduce latency for single queries).  
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    #[arg(flag = "-nocb")]
    pub no_cont_batching: bool,

    // ─────────────── Multimodal projector flags ───────────────────
    /// Path to a multimodal projector file (usually a GGUF file) for enabling
    /// vision or audio input. This is required for multimodal models that accept
    /// images or audio. If using `-hf` to load a model, the corresponding projector
    /// may be auto-downloaded (so this can often be omitted).  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--mmproj")]
    pub mmproj: Option<String>,

    /// Direct URL to a multimodal projector file. Use this if the projector is not
    /// bundled with the model or you want to provide a custom projector from a remote
    /// location. The file will be downloaded from the URL on startup.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--mmproj-url")]
    pub mmproj_url: Option<String>,

    /// Explicitly disable loading a multimodal projector. Even if a model has an
    /// associated projector (e.g., a vision model), this flag forces the server to
    /// ignore it and operate in text-only mode.  
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    #[arg(flag = "--no-mmproj")]
    pub no_mmproj: bool,

    /// Do not offload the multimodal projector to the GPU. By default, if a projector
    /// is used and a GPU is available, the projector model runs on the GPU. This flag
    /// keeps the projector on the CPU (which may be needed on GPUs with limited VRAM).  
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    #[arg(flag = "--no-mmproj-offload")]
    pub no_mmproj_offload: bool,

    // ───────────────── Chat template & reasoning ──────────────────
    /// JSON string of parameters to pass to the chat template system. Use this to
    /// provide additional context or options for a Jinja-based chat template. For
    /// example: `--chat-template-kwargs '{\"enable_thinking\":false}'` can disable
    /// certain template features if the template supports that parameter.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--chat-template-kwargs")]
    pub chat_template_kwargs: Option<String>,

    /// Enable Jinja2 templating for chat prompts. If this is set, custom chat
    /// templates (via `chat_template` or `chat_template_file`) will be processed as
    /// Jinja templates, allowing advanced logic in prompt construction.  
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    #[arg(flag = "--jinja")]
    pub jinja: bool,

    /// Select how the server handles `<think> … </think>` tags in model
    /// output.  
    ///  
    /// * **`deepseek`** (default) – extracts the text inside the tags and
    ///   places it in `message.reasoning_content`, leaving the visible
    ///   reply in `message.content` (tags remain when streaming).  
    /// * **`none`** – leaves the tags intact in `message.content` and skips
    ///   the extra field.  
    ///
    /// Corresponds to the CLI flag `--reasoning-format <value>`.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--reasoning-format")]
    pub reasoning_format: Option<ReasoningFormat>,

    /// Controls the amount of thinking allowed; currently only one of: -1 for
    /// unrestricted thinking budget, or 0 to disable thinking (default: -1)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--reasoning-budget")]
    pub reasoning_budget: Option<ReasoningBudget>,

    /// Specify a built-in chat template or provide a custom template string to format
    /// chat conversations. If the name matches a known template (e.g., "vicuna" or
    /// "chatml"), that template is used. If `--jinja` is enabled, you can supply a
    /// full Jinja template here. Overrides the model's default template.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--chat-template")]
    pub chat_template: Option<String>,

    /// Path to a file containing a chat template (generally in Jinja format). The
    /// file's content will be used to structure chat prompts. If `--jinja` is not set,
    /// the file should contain a supported template format or a simple prompt template.
    /// This overrides any built-in template from the model metadata.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--chat-template-file")]
    pub chat_template_file: Option<String>,

    /// Do not automatically prefill an assistant response. Normally, if the last turn
    /// in a conversation is by the assistant and is incomplete, the server might
    /// continue generating (prefill) that assistant answer on the next request. This
    /// flag treats an unfinished assistant turn as complete, so the next request will
    /// not append to it.  
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    #[arg(flag = "--no-prefill-assistant")]
    pub no_prefill_assistant: bool,

    // ───────────────────── Networking & I/O ───────────────────────
    /// Timeout for network read/write operations, in seconds. This applies to how
    /// long the server will wait on a client connection for data or for sending data
    /// before giving up. Default is 600 seconds (10 minutes).  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "-to")]
    pub timeout: Option<u64>,

    /// Number of threads dedicated to handling HTTP requests. This is separate from
    /// the inference threads. `-1` means use an automatic default (usually based on
    /// number of CPU cores or a fixed small number).  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--threads-http")]
    pub threads_http: Option<u64>,

    /// Enable prompt cache reuse with a minimum prefix length. If set to N > 0 and
    /// the new prompt starts with at least N tokens in common with the previous
    /// prompt, the server will reuse the previous inference's KV cache for that prefix
    /// (skipping re-computation). Default 0 disables this feature.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--cache-reuse")]
    pub cache_reuse: Option<u64>,

    // ─────────────────────── Slot management ──────────────────────
    /// Similarity threshold for reusing an existing slot (conversation state). When
    /// a new request comes in, its prompt is compared to existing slots' prompts; if
    /// the similarity is above this value, the server may use that slot to continue
    /// the conversation. Ranges 0.0 (disable) to 1.0 (must be identical). Default is
    /// 0.5 (50% similar).  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "-sps")]
    pub slot_prompt_similarity: Option<f32>,

    // ─────────────────────── LoRA staging flags ───────────────────
    /// Load any provided LoRA adapters into memory but do not apply them to the model
    /// weights initially. This allows the server to start with LoRAs loaded (so they
    /// can be quickly toggled on via the API) but the model runs in its base form
    /// until a POST request applies the LoRA changes.  
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    #[arg(flag = "--lora-init-without-apply")]
    pub lora_init_without_apply: bool,

    // ───────────── Speculative-decoding (draft model) ─────────────
    /// Maximum number of tokens to generate with the draft model for speculative
    /// decoding. The draft model will predict up to this many tokens ahead; the main
    /// model then validates or refines them. A higher value can speed up generation
    /// but might introduce more divergence (default 16).  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--draft-max")]
    pub draft_max: Option<u64>,

    /// Minimum number of draft tokens to use in speculative decoding. This ensures
    /// the draft model contributes at least this many tokens before potentially
    /// handing back to the main model. Default 0 means no minimum (it could stop
    /// using draft tokens sooner if conditions dictate).  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--draft-min")]
    pub draft_min: Option<u64>,

    /// Probability threshold for draft token acceptance (for speculative decoding).
    /// The draft model's predicted token is accepted greedily if its probability is
    /// above this value. Default 0.8 means the draft token must be fairly likely; if
    /// below, the main model will generate instead.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "--draft-p-min")]
    pub draft_p_min: Option<f32>,

    /// Context size for the draft model. If non-zero, overrides the default context
    /// length of the draft model. Usually this can remain 0 (use model's own context
    /// size) unless you need to limit or extend it explicitly.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "-cd")]
    pub ctx_size_draft: Option<u64>,

    /// Device specification for the draft model. Similar format to `--device`, you
    /// can specify which GPU(s) (or CPU) to use for the smaller draft model. This
    /// allows running the draft model on a different device than the main model if
    /// desired.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "-devd")]
    pub device_draft: Option<String>,

    /// Number of layers of the draft model to offload to GPU. If the draft model is
    /// also large, you can decide how many of its layers to keep in VRAM. Default is
    /// to offload as configured similarly to the main model, or 0 for CPU-only.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "-ngld")]
    pub gpu_layers_draft: Option<u32>,

    /// Path to a draft model file (GGUF/GGML) to use for speculative decoding. The
    /// draft model should be a smaller, faster model related to the main model (e.g.,
    /// distilled version) used to generate candidate tokens ahead of the main model.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "-md")]
    pub model_draft: Option<String>,

    /// Data type for the **key** cache of the draft model. Similar to `cache_type_k`
    /// but applies to the draft model's KV cache. Adjusting this can save memory for
    /// the draft model (default f16).  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "-ctkd")]
    pub cache_type_k_draft: Option<String>,

    /// Data type for the **value** cache of the draft model. Use this to quantize or
    /// change precision of the draft model's value cache, analogous to `cache_type_v`
    /// for the main model (default f16).  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "-ctvd")]
    pub cache_type_v_draft: Option<String>,

    // ───────────────────────── Audio / TTS ──────────────────────────
    /// Path to a vocoder model file for text-to-speech (TTS). If provided along with
    /// an LLM that outputs phonemes or audio tokens, the server can generate audio
    /// waveforms from text. Typically used with multi-modal models that support
    /// speech.  
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(option = "-mv")]
    pub model_vocoder: Option<String>,

    /// When using a TTS vocoder, enable guided tokens to improve word recall. Guided
    /// tokens provide hints or anchor points to the vocoder to ensure it pronounces
    /// all words clearly and in order. This can help with TTS accuracy and consistency.  
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    #[arg(flag = "--tts-use-guide-tokens")]
    pub tts_use_guide_tokens: bool,
}

impl Default for ServerArgs {
    fn default() -> Self {
        let model = llm_models::llm::local::gguf::gguf_model::GgufModel::default();
        ServerArgs::builder()
            .model(model.local_model_path())
            .expect("Default model should always be available")
            .build()
    }
}

use server_args_builder::{
    IsComplete, IsSet, IsUnset, SetHasModelSource, SetHfFile, SetHfRepo, SetModel, SetModelUrl,
    State,
};

impl<S: State> ServerArgsBuilder<S> {
    #[cfg(test)]
    pub fn default_model(self) -> LmcppResult<ServerArgsBuilder<SetModel<SetHasModelSource<S>>>>
    where
        S::Model: IsUnset,
        S::HasModelSource: IsUnset,
    {
        use crate::server::types::file::ValidFile;

        let model = llm_models::llm::local::gguf::gguf_model::GgufModel::default();

        Ok(self
            .has_model_source_internal(true)
            .model_internal(LocalModelPath(ValidFile::new(model.local_model_path())?)))
    }

    /// Set the [ServerArgs::model] field.
    pub fn model<V>(
        self,
        value: V,
    ) -> LmcppResult<ServerArgsBuilder<SetModel<SetHasModelSource<S>>>>
    where
        V: TryInto<crate::server::types::file::ValidFile, Error = LmcppError>,
        S::Model: IsUnset,
        S::HasModelSource: IsUnset,
    {
        let path = value.try_into()?;

        Ok(self
            .has_model_source_internal(true)
            .model_internal(LocalModelPath(path)))
    }

    /// Set the [ServerArgs::model_url] field.
    pub fn model_url<V>(
        self,
        value: V,
    ) -> LmcppResult<ServerArgsBuilder<SetModelUrl<SetHasModelSource<S>>>>
    where
        V: TryInto<url::Url, Error: std::fmt::Display>,
        S::ModelUrl: IsUnset,
        S::HasModelSource: IsUnset,
    {
        let url: url::Url = value.try_into().map_err(|e| LmcppError::InvalidConfig {
            field: "model_url",
            reason: format!("failed to parse model URL: {}", e),
        })?;
        Ok(self
            .has_model_source_internal(true)
            .model_url_internal(ModelUrl(url)))
    }

    /// Set the [ServerArgs::hf_repo] field.
    pub fn hf_repo<V>(
        self,
        value: V,
    ) -> LmcppResult<ServerArgsBuilder<SetHfRepo<SetHasModelSource<S>>>>
    where
        V: TryInto<HfRepo, Error = LmcppError>,
        S::HfRepo: IsUnset,
        S::HasModelSource: IsUnset,
    {
        let repo = value.try_into()?;
        Ok(self.has_model_source_internal(true).hf_repo_internal(repo))
    }

    /// Set the [ServerArgs::hf_file] field.
    pub fn hf_file<V>(self, value: V) -> LmcppResult<ServerArgsBuilder<SetHfFile<S>>>
    where
        V: TryInto<HfFile, Error = LmcppError>,
        S::HfFile: IsUnset,
        S::HfRepo: IsSet,
    {
        let file = value.try_into()?;
        Ok(self.hf_file_internal(file))
    }
}

impl<State: server_args_builder::State> ServerArgsBuilder<State>
where
    State: IsComplete,
    State::HasModelSource: IsSet,
{
    pub fn build(self) -> ServerArgs {
        let args = self.build_internal();
        args
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
#[serde(rename_all = "lowercase")] // "none", "linear", "yarn"
pub enum RopeScaling {
    None,
    Linear,
    Yarn,
}

impl Arg for RopeScaling {
    fn append_arg(&self, command: &mut std::process::Command) {
        command.arg(self.to_string().to_lowercase());
    }
}

impl Display for RopeScaling {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use RopeScaling::*;
        f.write_str(match self {
            None => "none",
            Linear => "linear",
            Yarn => "yarn",
        })
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum NumaStrategy {
    Distribute,
    Isolate,
    Numactl,
}

impl Arg for NumaStrategy {
    fn append_arg(&self, command: &mut std::process::Command) {
        command.arg(self.to_string().to_lowercase());
    }
}

impl Display for NumaStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self).map(|_| ())
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum SplitMode {
    None,
    Layer,
    Row,
}

impl Arg for SplitMode {
    fn append_arg(&self, command: &mut std::process::Command) {
        command.arg(self.to_string().to_lowercase());
    }
}

impl Display for SplitMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self).map(|_| ())
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningFormat {
    None,
    Deepseek,
    Auto,
}

impl Arg for ReasoningFormat {
    fn append_arg(&self, command: &mut std::process::Command) {
        command.arg(self.to_string().to_lowercase());
    }
}

impl Display for ReasoningFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self).map(|_| ())
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningBudget {
    None,      // 0
    Unlimited, // -1
}

impl Arg for ReasoningBudget {
    fn append_arg(&self, command: &mut std::process::Command) {
        command.arg(self.to_string().to_lowercase());
    }
}

impl Display for ReasoningBudget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use ReasoningBudget::*;
        f.write_str(match self {
            None => "0",
            Unlimited => "-1",
        })
    }
}
