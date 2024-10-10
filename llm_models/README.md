# llm_models: Load and Download LLM Models, Metadata, and Tokenizers

This crate is part of the [llm_client](https://github.com/ShelbyJenkins/llm_client) crate.

* GGUFs from local storage or Hugging Face
    * Parses model metadata from GGUF file
    * Includes limited support for tokenizer from GGUF file
    * Also supports loading Metadata and Tokenizer from their respective files

### LocalLlmModel

Everything you need for GGUF models. The `GgugLoader` wraps the loaders for convience. All loaders return a `LocalLlmModel` which contains the tokenizer, metadata, chat template, and anything that can be extract from the GGUF. 


#### GgufPresetLoader

* Presets for popular models like Llama 3, Phi, Mistral/Mixtral, and more
* Loads the best quantized model by calculating the largest quant that will fit in your VRAM

```rust
let model: LocalLlmModel = GgufLoader::default()
    .llama3_1_8b_instruct()
    .preset_with_available_vram_gb(48) // Load the largest quant that will fit in your vram
    .load()?;
```

#### GgufHfLoader

GGUF models from Hugging Face.

```rust
let model: LocalLlmModel = GgufLoader::default()
    .hf_quant_file_url("https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/blob/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf")
    .load()?;
```

#### GgufLocalLoader

GGUF models for local storage.

```rust
let model: LocalLlmModel = GgufLoader::default()
    .local_quant_file_path("/root/.cache/huggingface/hub/models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF/blobs/9da71c45c90a821809821244d4971e5e5dfad7eb091f0b8ff0546392393b6283")
    .load()?;
```

#### ApiLlmModel

* Supports openai, anthropic, perplexity, and adding your own API models
* Supports prompting, tokenization, and price estimation

```rust
    assert_eq!(ApiLlmModel::gpt_4_o(), ApiLlmModel {
        model_id: "gpt-4o".to_string(),
        context_length: 128000,
        cost_per_m_in_tokens: 5.00,
        max_tokens_output: 4096,
        cost_per_m_out_tokens: 15.00,
        tokens_per_message: 3,
        tokens_per_name: 1,
        tokenizer: Arc<LlmTokenizer>,
    })
```

### LlmTokenizer

* Simple abstract API for encoding and decoding allows for abstract LLM consumption across multiple architechtures.
*Hugging Face's Tokenizer library for local models and Tiktoken-rs for OpenAI and Anthropic ([Anthropic doesn't have a publically available tokenizer](https://github.com/javirandor/anthropic-tokenizer).)

```rust
    let tok = LlmTokenizer::new_tiktoken("gpt-4o"); // Get a Tiktoken tokenizer
    let tok = LlmTokenizer::new_from_tokenizer_json("path/to/tokenizer.json"); // From local path
    let tok = LlmTokenizer::new_from_hf_repo(hf_token, "meta-llama/Meta-Llama-3-8B-Instruct"); // From repo
    // From LocalLlmModel or ApiLlmModel
    let tok = model.model_base.tokenizer;
```

### Setter Traits
* All setter traits are public, so you can integrate into your own projects if you wish. 
* For example: `OpenAiModelTrait`,`GgufLoaderTrait`,`AnthropicModelTrait`, and `HfTokenTrait` for loading models 