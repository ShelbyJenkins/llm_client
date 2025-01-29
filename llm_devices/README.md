<!-- cargo-rdme start -->

# llm_devices: Device management and build system for LLM inference
[![API Documentation](https://docs.rs/llm_devices/badge.svg)](https://docs.rs/llm_devices)

The llm_devices crate is a workspace member of the [llm_client](https://github.com/ShelbyJenkins/llm_client) project.
It is used as a dependency by the [llm_interface](https://github.com/ShelbyJenkins/llm_client/tree/master/llm_interface) crate for building llama.cpp.

## Features

* Automated building of llama.cpp with appropriate platform-specific optimizations
* Device detection and configuration for CPU, RAM, CUDA (Linux/Windows), and Metal (macOS)
* Manages memory by detecting available VRAM/RAM, estimating model fit, and distributing layers across devices
* Logging tools

<!-- cargo-rdme end -->
