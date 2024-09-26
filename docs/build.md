
# Llama.cpp Backend

The current default feature is `llama_cpp_backend`. It is pulled from the repo and built with the [build.rs](../llm_interface/build.rs). 

### Linux / Mac / Windows CPU Build (Default)

* Supports all platforms
* Install time: ~30 seconds

### Linux / Windows CUDA Build 

* Requires the `cuda` feature flag
* Also requires Nvidia specific [dependencies listed here](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md) or you can run the [dev container](../.devcontainer/devcontainer.json) in this repo.
* Install time: ~120 seconds

### Mac GPU Build (Default)

* In theory, llama.cpp builds for Mac GPU backends by default, but I have not tested.

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

