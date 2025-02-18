//! # llm_devices: Device management and build system for LLM inference
//! [![API Documentation](https://docs.rs/llm_devices/badge.svg)](https://docs.rs/llm_devices)
//!
//! The llm_devices crate is a workspace member of the [llm_client](https://github.com/ShelbyJenkins/llm_client) project.
//! It is used as a dependency by the [llm_interface](https://github.com/ShelbyJenkins/llm_client/tree/master/llm_interface) crate for building llama.cpp.
//!
//! ## Features
//!
//! * Automated building of llama.cpp with appropriate platform-specific optimizations
//! * Device detection and configuration for CPU, RAM, CUDA (Linux/Windows), and Metal (macOS)
//! * Manages memory by detecting available VRAM/RAM, estimating model fit, and distributing layers across devices
//! * Logging tools

// Internal modules
mod devices;
mod llm_binary;
mod logging;
mod target_dir;

// Internal imports
#[allow(unused_imports)]
use anyhow::{anyhow, bail, Error, Result};
#[allow(unused_imports)]
use tracing::{debug, error, info, span, trace, warn, Level};

// Public exports
pub use self::{
    devices::{CpuConfig, DeviceConfig},
    llm_binary::build_or_install,
    llm_binary::{get_bin_dir, get_bin_path, LLAMA_CPP_SERVER_EXECUTABLE},
    logging::{i_ln, i_lns, i_nln, i_nlns, LoggingConfig, LoggingConfigTrait},
    target_dir::get_target_directory,
};

// Platform-specific exports
#[cfg(target_os = "macos")]
pub use devices::MetalConfig;
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub use devices::{init_nvml_wrapper, CudaConfig};
