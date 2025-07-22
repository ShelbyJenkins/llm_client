Adding a new GGUF preset model:
1. If it's a new "Organization":
    - Add a directory to `llm_client/llm_models_macros/src/gguf_models/data`
    - Add a `organization.json`
2. Create directory for the new model in `llm_client/llm_models_macros/src/gguf_models/data/<organization>`
    - Note: I don't think the naming format for the directory matters but we currently use the `model_id` derived from the HuggingFace repo for the model.
3. Add model_macro_data.json to the new model's directory
4. (Optional) Add the model's config.json to the new model's directory
5. (Optional) Add the model's tokenizer.json to the new model's directory
    - Extracting a tokenizer from a GGUF is still in testing, so we may need to do this for some models.
6. Run `llm_client/llm_models_macros/src/bin/generate_models.rs` to update llm_models crate.


Adding an API model:
1. If it's a new "Provider":
    - Add a module to `llm_client/llm_models_macros/src/api_models/data`
    - Add a `mod.rs`
    - Add provider to `llm_client/llm_models_macros/src/api_models/provider.rs`
2. Add model to `llm_client/llm_models_macros/src/api_models/data/<provider>/models.json`
3. Run `llm_client/llm_models_macros/src/bin/generate_models.rs` to update llm_models crate.