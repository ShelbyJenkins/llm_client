There are three options for loading GGUF models. There are [examples](../llm_models/examples/) in the llm_models crate.

In the [device config example](../llm_client/examples/device_config.rs), you can see how to use different GPU and CPU configurations. 

# Loading Models from Presets

I maintain some common presets for LLMs. We estimate the model size based on the metadata from these presets.

When using a GPU, your available VRAM (either automated from all available or from your device config) will be used to select the largest quant for the preset you've choosen. You can also choose to specify the amount of VRAM you wish to use.

Optionally, you can choose the max context size you want to use as it's an important part of memory estimates. Otherwise it defaults to ~8000 tokens.

The llm_utils crate then calculates the largest quant you can use, and downloads the model from a list of GGUF quants on Hugging Face.The presets include config files and tokenizers from the original repos, so you do not have to login to Hugging Face and get Meta's or Mistral's approval to access the original model's repo. I've done some work in accessing the tokenizer from the GGUF file themselves, but this is not fully tested. I do retrieve the chat template from the GGUF though.

```rust
use llm_client::prelude::*;

let llm_client = LlmClient::llama_cpp()
    .mistral7b_instruct_v0_3() // Uses a preset model
    .init() // Downloads model from hugging face and starts the inference interface
    .await?;

let llm_client = LlmClient::llama_cpp()
    .llama3_1_8b_instruct()
    .hf_token(hf_token) // Add your hugging face API token
    .hf_token_env_var("HF_TOKEN_ENV_VAR") // Or the env var to access it
    .init() 
    .await?;
```

# Loading Models from Local

```rust
use llm_client::prelude::*;

let llm_client = LlmClient::llama_cpp()
    .local_quant_file_path("/root/.cache/huggingface/hub/models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF/blobs/9da71c45c90a821809821244d4971e5e5dfad7eb091f0b8ff0546392393b6283")
    .init().await?;
```

# Loading Models from Hugging Face

```rust
use llm_client::prelude::*;

let llm_client = LlmClient::llama_cpp()
    .hf_quant_file_url("https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/blob/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf")
    .init().await?;
```
