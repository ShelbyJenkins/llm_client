# llm_prompt: Low Level Prompt System for API LLMs (OpenAI) and local LLMs (Chat Template)

This crate is part of the [llm_client](https://github.com/ShelbyJenkins/llm_client) project.

[API Docs](https://docs.rs/llm_prompt/latest/llm_prompt/)

__Local LLM Support__ 
* Uses the LLM's chat template to properly format the prompt.
* Wider support than Llama.cpp with Jinja chat templates and raw tokens
    * Llama.cpp attempts to build a prompt from string input using community implementations of chat templates and matching via model ids. It does not always work nor does it support all models.
    * Llama.cpp performs no manipulation of the input when sending just tokens, so using this crate's `get_built_prompt_as_tokens` function is safer.
* Build with generation prefixes
    * Supports all local models. Even those that don't explicitly support it.

__API LLM Support__ 
* OpenAI formatted prompts (OpenAI, Anthropic, Etc.)
    *  Outputs System/User/Assistant keys and content strings.

__Accurate Token Counts__ 
* Accurately count prompt tokens 
    * Ensures a prompt is within model limits.
    * Handles unique rules for API and Local models.


__User friendly__ 
* A single struct with thread safe interior mutability for ergonomics.
* Will fail gracefully via result if the prompt does not match turn ordering rules.

__Save and load from file__ 
* Serde implemented for PromptMessages




### Use
This llm_models crate from the [llm_client](https://github.com/ShelbyJenkins/llm_client) project is used here for example purposes. It is not required. 

```rust
use llm_prompt::*;
use llm_models::api_model::ApiLlmModel;
use llm_models::local_model::LocalLlmModel;

// OpenAI Format
let model = ApiLlmModel::gpt_3_5_turbo();
let prompt = LlmPrompt::new_api_prompt(
    model.model_base.tokenizer.clone(),
    Some(model.tokens_per_message),
    model.tokens_per_name,
);

// Chat Template
let model = LocalLlmModel::default();
let prompt = LlmPrompt::new_local_prompt(
    model.model_base.tokenizer.clone(),
    &model.chat_template.chat_template,
    model.chat_template.bos_token.as_deref(),
    &model.chat_template.eos_token,
    model.chat_template.unk_token.as_deref(),
    model.chat_template.base_generation_prefix.as_deref(),
);
// There are three types of 'messages'

// Add system messages
prompt.add_system_message()?.set_content("You are a nice robot");

// User messages
prompt.add_user_message()?.set_content("Hello");

// LLM responses
prompt.add_assistant_message()?.set_content("Well, how do you do?");

// Builds with generation prefix. The llm will complete the response: 'Don't you think that is... cool?'
// Only Chat Template format supports this
prompt.set_generation_prefix("Don't you think that is...");



// Access (and build) the underlying prompt topography
let local_prompt: &LocalPrompt = prompt.local_prompt()?;
let api_prompt: &ApiPrompt = prompt.api_prompt()?;

// Get chat template formatted prompt
let local_prompt_as_string: String = prompt.local_prompt()?.get_built_prompt()?;
let local_prompt_as_tokens: Vec<u32> = prompt.local_prompt()?.get_built_prompt_as_tokens()?;

// Openai formatted prompt (Openai and Anthropic format)
let api_prompt_as_messages: Vec<HashMap<String, String>> = prompt.api_prompt()?.get_built_prompt()?;



// Get total tokens in prompt
let total_prompt_tokens: u64 = prompt.local_prompt()?.get_total_prompt_tokens();
let total_prompt_tokens: u64 = prompt.api_prompt()?.get_total_prompt_tokens();

// Validate requested max_tokens for a generation. If it exceeds the models limits, reduce max_tokens to a safe value
let actual_request_tokens = check_and_get_max_tokens(
    model.context_length,
    Some(model.max_tokens_output), // If using a GGUF model use either model.context_length or the ctx_size of the server
    total_prompt_tokens,
    Some(10), // Safety tokens
    requested_max_tokens,
)?;
```

`LlmPrompt` requires a tokenizer. You can use the [llm_models](https://github.com/ShelbyJenkins/llm_client/tree/master/llm_models/src/tokenizer.rs) crate's tokenizer, or implement the [PromptTokenizer](./src/lib.rs) trait on your own tokenizer.

```rust
impl PromptTokenizer for LlmTokenizer {
    fn tokenize(&self, input: &str) -> Vec<u32> {
        self.tokenize(input)
    }

    fn count_tokens(&self, str: &str) -> u32 {
        self.count_tokens(str)
    }
}
```
