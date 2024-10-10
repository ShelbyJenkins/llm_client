# llm_prompt: Low Level LLM API for OpenAI and Chat Template Formatted Prompts

This crate is part of the [llm_client](https://github.com/ShelbyJenkins/llm_client) crate.

* OpenAI formatted prompts (OpenAI, Anthropic, Etc.)
    * Hashmap
* Chat template formatted prompts (Local LLMs)
    * Strings
    * Tokens 
* Uses the GGUF's chat template and Jinja templates to format the prompt to model spec. 
* Build with generation prefixes on all chat template models. Even those that don't explicitly support it.
* Count prompt tokens and check to ensure it's within model limits

```rust
use llm_prompt::*;

// OpenAI Format
let prompt: LlmPrompt = LlmPrompt::new_openai_prompt(
    Some(model.tokens_per_message),
    Some(model.tokens_per_name),
    model.model_base.tokenizer.clone(),
);

// Chat Template
let prompt: LlmPrompt = LlmPrompt::new_chat_template_prompt(
    &model.chat_template.chat_template,
    &model.chat_template.bos_token,
    &model.chat_template.eos_token,
    model.chat_template.unk_token.as_deref(),
    model.chat_template.base_generation_prefix.as_deref(),
    model.model_base.tokenizer.clone(),
);
// There are three types of 'messages'

// Add system messages
prompt.add_system_message().set_content("You are a nice robot");

// User messages
prompt.add_user_message().set_content("Hello");

// LLM responses
prompt.add_assistant_message().set_content("Well how do you do?");

// Messages all share the same functions see prompting::PromptMessage for more
prompt.add_system_message().append_content(final_rule_set);
prompt.add_system_message().prepend_content(starting_rule_set);

// Builds with generation prefix. The llm will complete the response: 'Don't you think that is... cool?'
// Only Chat Template format supports this
prompt.set_generation_prefix("Don't you think that is...");

// Get total tokens in prompt
let total_prompt_tokens: u32 = prompt.get_total_prompt_tokens();

// Get chat template formatted prompt
let chat_template_prompt: String = prompt.get_built_prompt_string();
let chat_template_prompt_as_tokens: Vec<u32> = prompt.get_built_prompt_as_tokens()

// Openai formatted prompt (Openai and Anthropic format)
let openai_prompt: Vec<HashMap<String, String>> = prompt.get_built_prompt_hashmap()

// Validate requested max_tokens for a generation. If it exceeds the models limits, reduce max_tokens to a safe value
let actual_request_tokens = check_and_get_max_tokens(
    model.context_length,
    model.max_tokens_output, // If using a GGUF model use either model.context_length or the ctx_size of the server
    total_prompt_tokens,
    10, // Safety tokens
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
