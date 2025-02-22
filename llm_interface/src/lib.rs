//! # llm_interface: The backend for the llm_client crate
//! [![API Documentation](https://docs.rs/llm_interface/badge.svg)](https://docs.rs/llm_interface)
//!
//! The llm_interface crate is a workspace member of the [llm_client](https://github.com/ShelbyJenkins/llm_client) project.
//!
//! This crate contains the build.rs, data types, and behaviors for LLMs.
//!
//! ## Features
//!
//! * Integration with Llama.cpp (through llama-server)
//!     * Repo cloning and building
//!     * Managing Llama.cpp server
//! * Support for various LLM APIs including generic OpenAI format LLMs
//!
//! This crate enables running local LLMs and making requests to LLMs, designed
//! for easy integration into other projects.
//!
//! ## Examples
//!
//! See the various `Builders` implemented in the [integration tests](./tests/it/main.rs) for examples
//! of using this crate.
//!
//! For a look at a higher level API and how it implements this crate, see the
//! [llm_client](https://github.com/ShelbyJenkins/llm_client) crate.

// Public modules
pub mod llms;
pub mod requests;

// Internal imports
#[allow(unused_imports)]
use anyhow::{anyhow, bail, Error, Result};
#[allow(unused_imports)]
use tracing::{debug, error, info, span, trace, warn, Level};
