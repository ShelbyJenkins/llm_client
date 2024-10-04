
# Llama.cpp Backend

The current default feature is `llama_cpp_backend`. Llama.cpp is pulled from the repo and built with llm_interface's [build.rs](../llm_interface/build.rs). The [llm_device](../llm_devices/) crate contains the behavior for building.

### Linux / Mac / Windows CPU Build (Default)

* Supports all platforms
* Install time: ~30 seconds

### Linux / Windows CUDA Build 

* If CUDA is available on your system, it will build for CUDA automatically.
* Also requires Nvidia specific [dependencies listed here](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md) or you can run the [dev container](../.devcontainer/devcontainer.json) in this repo.
* Install time: ~120 seconds

### Mac GPU Build (Default)

* Nothing is required, as llama.cpp builds for mac by default.
* Tested as working.

# Mistral.rs Backend

Requires the `mistral_rs_backend` feature flag.

### Linux / Mac / Windows CPU Build (Default)

* Supports all platforms
* Install time: ~30 seconds

### Linux / Windows CUDA Build 

* Requires the `cuda` feature flag
    * `cudnn` is also available
* Also requires Nvidia specific [dependencies listed here](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md) or you can run the [dev container](../.devcontainer/devcontainer.json) in this repo.
* Install time: ~60 seconds

### Mac GPU Build

* Requires the `metal` feature flag
* Not tested.

