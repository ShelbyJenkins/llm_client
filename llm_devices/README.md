# llm_devices: Managing Devices and Builds for LLMs

This crate is part of the [llm_client](https://github.com/ShelbyJenkins/llm_client) crate.

The [llm_interface](https://github.com/ShelbyJenkins/llm_client/tree/master/llm_interface) crate uses it as a dependency for building llama.cpp.

It's functionality includes:

* Cloning the specified tag, and building llama.cpp.

* Checking for device availabilty (CUDA, MacOS) to determine what platform to build for.

* Fetching available VRAM or system RAM for estimating the correct model to load.

* Offloading model layers to memory.

* Logging tools.

See the [build documentation](../docs/build.md) for more notes.